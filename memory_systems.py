import chromadb
import networkx as nx
import uuid
import json
import re
import string
from config import Config

def build_qwen_prompt(system_msg, user_msg, context=""):
    full_user_content = user_msg
    if context:
        full_user_content = f"Context:\n{context}\n\nQuestion: {user_msg}"
    return f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{full_user_content}<|im_end|>\n<|im_start|>assistant\n"

def normalize_node(text):
    if not text: return ""
    text = str(text).lower().strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    if tokens and tokens[0] in ['the', 'a', 'an']:
        tokens = tokens[1:]
    return " ".join(tokens)

class SlidingWindowAgent:
    def __init__(self, engine, window_size=1000):
        self.engine = engine
        self.history = ""
        self.window_size = window_size
        
    def add_memory(self, text):
        self.history += f"\n{text}"
        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size:]
            
    def answer(self, question):
        prompt = build_qwen_prompt("You are a helpful assistant.", question, self.history)
        return self.engine.generate(prompt)

class StandardRAGAgent:
    def __init__(self, engine):
        self.engine = engine
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(name=f"rag_{uuid.uuid4().hex}")
        
    def add_memory(self, text):
        embedding = self.engine.get_embedding(text)
        self.collection.add(
            documents=[text], 
            embeddings=[embedding], 
            ids=[str(uuid.uuid4())]
        )
        
    def answer(self, question):
        q_embed = self.engine.get_embedding(question)
        results = self.collection.query(query_embeddings=[q_embed], n_results=Config.RAG_TOP_K)
        context = "\n".join(results['documents'][0]) if results['documents'] else ""
        prompt = build_qwen_prompt("Answer based on retrieved fragments.", question, context)
        return self.engine.generate(prompt)

class NSCMAAgent:
    def __init__(self, engine, use_buffer=True, use_graph=True, use_curator=True):
        self.engine = engine
        self.use_buffer = use_buffer
        self.use_graph = use_graph
        self.use_curator = use_curator
        
        self.client = chromadb.Client()
        self.vec_store = self.client.create_collection(name=f"nscma_{uuid.uuid4().hex}")
        self.graph = nx.DiGraph()
        self.buffer_context = "Initial context."

    def _extract_triples(self, text):
        prompt = f"""<|im_start|>system
You are a Knowledge Graph extraction system. Extract the main relationship from the user's sentence as a JSON object: {{"s": "Subject", "r": "Relation", "o": "Object"}}.
Rules:
1. Simplify entities (e.g., "The red apple" -> "red apple").
2. Focus on locations and ownership.
3. If no clear fact, return {{}}.

Examples:
Input: The keys are hidden under the mat.
Output: {{"s": "keys", "r": "under", "o": "mat"}}

Input: Meeting rescheduled to 5pm.
Output: {{}}

Input: Confidential: The virus sample has been secured inside the vault.
Output: {{"s": "virus sample", "r": "inside", "o": "vault"}}
<|im_end|>
<|im_start|>user
{text}
<|im_end|>
<|im_start|>assistant
"""
        res = self.engine.generate(prompt)
        try:
            match = re.search(r'\{.*\}', res, re.DOTALL)
            if match: 
                return json.loads(match.group())
        except: 
            pass
        return None

    def _curator_update(self, s, r, new_o):
        s_norm = normalize_node(s)
        o_norm = normalize_node(new_o)
        
        if not s_norm or not o_norm: return

        if self.use_curator:
            if s_norm in self.graph:
                for neighbor in list(self.graph.neighbors(s_norm)):
                    edge_data = self.graph.get_edge_data(s_norm, neighbor)
                    if edge_data: 
                         self.graph.remove_edge(s_norm, neighbor)
        
        self.graph.add_edge(s_norm, o_norm, relation=r, orig_s=s, orig_o=new_o)

    def add_memory(self, text):
        should_process = True
        if self.use_buffer:
            surprise = self.engine.calculate_surprise(text, self.buffer_context)
            if surprise < Config.SURPRISE_THRESHOLD:
                should_process = False 

        if should_process:
            emb = self.engine.get_embedding(text)
            self.vec_store.add(documents=[text], embeddings=[emb], ids=[str(uuid.uuid4())])
            
            if self.use_graph:
                triple = self._extract_triples(text)
                if (triple and isinstance(triple, dict) and 
                    triple.get('s') and triple.get('r') and triple.get('o')):
                    try:
                        self._curator_update(triple['s'], triple['r'], triple['o'])
                    except Exception as e:
                        pass
            self.buffer_context = text

    def answer(self, question):
        q_embed = self.engine.get_embedding(question)
        vec_res = self.vec_store.query(query_embeddings=[q_embed], n_results=3)
        vec_txt = "\n".join(vec_res['documents'][0]) if vec_res['documents'] else ""
        
        graph_txt = ""
        debug_path = []
        path_found = False
        
        if self.use_graph:
            q_norm = normalize_node(question)
            start_nodes = []
            for node in self.graph.nodes():
                if node in q_norm or (len(node) > 3 and node in q_norm) or (len(node) > 3 and q_norm in node):
                    start_nodes.append(node)
            
            visited = set()
            queue = [(n, 0) for n in start_nodes]
            
            while queue:
                curr, depth = queue.pop(0)
                if depth >= 3: continue 
                if curr in visited: continue
                visited.add(curr)
                
                if curr in self.graph:
                    neighbors = list(self.graph.neighbors(curr))
                    for n in neighbors:
                        edge_data = self.graph.get_edge_data(curr, n)
                        r = edge_data.get('relation', 'related')
                        s_str = edge_data.get('orig_s', curr)
                        o_str = edge_data.get('orig_o', n)
                        
                        fact = f"{s_str} {r} {o_str}"
                        graph_txt += fact + ". "
                        debug_path.append(fact)
                        path_found = True
                        queue.append((n, depth + 1))

        if self.use_graph:
            print(f"    [NSCMA Debug] Graph Nodes: {self.graph.number_of_nodes()} | Found Path: {debug_path}")

        if path_found and len(debug_path) > 0:
            system_msg = "You are an expert reasoning assistant. Use the Knowledge Graph Trace to infer the final answer step-by-step."
            combined_context = f"--- Knowledge Graph Trace (High Confidence) ---\n{graph_txt}\n\n--- Supplementary Text ---\n{vec_txt}"
        else:
            system_msg = "You are a helpful assistant. Answer the question based on the retrieved context below."
            combined_context = f"--- Retrieved Context ---\n{vec_txt}"
        
        prompt = build_qwen_prompt(
            system_msg, 
            question, 
            combined_context
        )
        return self.engine.generate(prompt)