import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from config import Config

class ModelEngine:
    def __init__(self):
        print(f"Initializing Engine...")
        print("Loading Embedding Model to CPU to save VRAM...")
        try:
            self.embedder = SentenceTransformer(Config.EMBEDDING_MODEL_ID, device="cpu")
        except Exception as e:
            print(f"Embedding model load failed: {e}")
            raise e
        target_device = "cuda:0" if Config.DEVICE == "cuda" else "cpu"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print(f"Loading Tokenizer ({Config.LLM_MODEL_ID})...")
        self.tokenizer = AutoTokenizer.from_pretrained(Config.LLM_MODEL_ID)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"Loading LLM to {target_device}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            Config.LLM_MODEL_ID,
            quantization_config=bnb_config if Config.LOAD_IN_4BIT else None,
            device_map=target_device, 
            trust_remote_code=True
        )
        self.model.eval()

    def get_embedding(self, text):
        return self.embedder.encode(text, convert_to_tensor=False).tolist()

    def generate(self, prompt):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024 
            ).to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=Config.MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        except Exception as e:
            print(f"Generation Error: {e}")
            return ""

    def calculate_surprise(self, text, history_summary):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        input_text = f"{history_summary}\n{text}"
        try:
            encodings = self.tokenizer(
                input_text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024
            ).to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**encodings, labels=encodings["input_ids"])
                loss = outputs.loss
            return torch.exp(loss).item()
        except Exception as e:
            print(f"Surprise Calc Error: {e}")
            return 10.0