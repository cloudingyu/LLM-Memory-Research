import time
import json
import numpy as np
import traceback
from tqdm import tqdm
from config import Config
from models import ModelEngine
from data_loader import DataLoader
from memory_systems import SlidingWindowAgent, StandardRAGAgent, NSCMAAgent

def chunk_text_stream(stream, chunk_size=15):
    """
    将碎片的句子流合并成更大的块，减少 LLM 调用次数，提升速度。
    """
    chunks = []
    current_chunk = []
    for sentence in stream:
        current_chunk.append(sentence)
        if len(current_chunk) >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def check_answer(prediction, ground_truth):
    """
    宽松匹配逻辑
    """
    if not prediction or not ground_truth:
        return False
    pred_norm = prediction.lower().strip()
    truth_norm = str(ground_truth).lower().strip()
    if truth_norm in pred_norm:
        return True
    return False

def run_experiment_1_variable_noise(engine):
    print("\n" + "="*70)
    print(">>> Experiment 1: Knowledge Update Robustness (Variable Noise) <<<")
    print("="*70)
    noise_lengths = [10, 50, 100, 200]
    final_stats = {
        "SlidingWindow": [],
        "StandardRAG": [],
        "NSCMA": []
    }
    systems = {
        "SlidingWindow": lambda: SlidingWindowAgent(engine, window_size=1000), 
        "StandardRAG":   lambda: StandardRAGAgent(engine),
        "NSCMA":         lambda: NSCMAAgent(engine)
    }
    for length in noise_lengths:
        print(f"\n--- Testing Noise Length: {length} ---")
        try:
            data = DataLoader.generate_synthetic_update(
                limit=Config.TEST_SAMPLE_LIMIT, 
                noise_length=length
            )
        except Exception as e:
            print(f"❌ Generation Error: {e}"); continue
        current_scores = {k: 0 for k in systems}
        for item in tqdm(data, desc=f"L={length}"):
            agents = {name: constructor() for name, constructor in systems.items()}
            chunks = chunk_text_stream(item['stream'], chunk_size=15)
            for name, agent in agents.items():
                for text in chunks:
                    agent.add_memory(text)
                ans = agent.answer(item['question'])
                if check_answer(ans, item['ground_truth']):
                    current_scores[name] += 1
        for name, score in current_scores.items():
            acc = (score / len(data)) * 100
            final_stats[name].append(acc)
            print(f"  {name:<15}: {acc:.1f}%")
    with open("exp1_results.json", "w") as f:
        json.dump({"x_axis": noise_lengths, "y_axis": final_stats}, f)
    print("\n" + "-"*60)
    print(">>> Experiment 1 Summary Table (Accuracy %)")
    print("-"*60)
    print(f"{'Length':<10} | {'Sliding':<15} | {'RAG':<15} | {'NSCMA':<15}")
    print("-"*60)
    for i, l in enumerate(noise_lengths):
        print(f"{l:<10} | {final_stats['SlidingWindow'][i]:<15.1f} | {final_stats['StandardRAG'][i]:<15.1f} | {final_stats['NSCMA'][i]:<15.1f}")
    print("-"*60)

def run_experiment_2_multihop(engine):
    print("\n" + "="*70)
    print(">>> Experiment 2: Multi-hop Reasoning (Synthetic Data) <<<")
    print("="*70)
    try:
        data = DataLoader.generate_synthetic_multihop(
            limit=Config.TEST_SAMPLE_LIMIT,
            noise_length=50
        )
    except Exception as e:
        print(f"❌ Error: {e}"); return
    systems = {
        "StandardRAG": lambda: StandardRAGAgent(engine),
        "NSCMA":       lambda: NSCMAAgent(engine)
    }
    results = {k: 0 for k in systems}
    print(f"Evaluating {len(data)} samples...")
    for item in tqdm(data, desc="Exp2 Progress"):
        raw_stream = [s.strip() for s in item['context'].split('.') if len(s.strip()) > 3]
        chunks = chunk_text_stream(raw_stream, chunk_size=15)
        agents = {name: constructor() for name, constructor in systems.items()}
        for name, agent in agents.items():
            for text in chunks:
                agent.add_memory(text)
            ans = agent.answer(item['question'])
            if check_answer(ans, item['ground_truth']):
                results[name] += 1
    print("\n--- Results (Exp 2: Multi-hop) ---")
    for k, v in results.items():
        acc = (v / len(data)) * 100
        print(f"{k:<15}: {v}/{len(data)} ({acc:.1f}%)")

def run_experiment_3_ablation(engine):
    print("\n" + "="*70)
    print(">>> Experiment 3: Ablation Study (Component Analysis) <<<")
    print("="*70)
    limit = min(5, Config.TEST_SAMPLE_LIMIT)
    try:
        data_update = DataLoader.generate_synthetic_update(limit=limit, noise_length=40)
        data_multihop = DataLoader.generate_synthetic_multihop(limit=limit, noise_length=40)
    except Exception as e:
        print(f"❌ Data Error: {e}"); return
    variants = {
        "Full NSCMA":         {"use_buffer": True,  "use_graph": True,  "use_curator": True},
        "w/o Semantic Graph": {"use_buffer": True,  "use_graph": False, "use_curator": False}, 
        "w/o Curator":        {"use_buffer": True,  "use_graph": True,  "use_curator": False},
        "w/o Neural Buffer":  {"use_buffer": False, "use_graph": True,  "use_curator": True}
    }
    print("-" * 90)
    print(f"{'Variant':<20} | {'Acc (Multi-hop)':<18} | {'Acc (Update)':<15} | {'Avg Time (s)':<15}")
    print("-" * 90)
    for name, config in variants.items():
        score_update = 0
        for item in data_update:
            agent = NSCMAAgent(engine, **config)
            chunks = chunk_text_stream(item['stream'], chunk_size=15)
            for text in chunks:
                agent.add_memory(text)
            if check_answer(agent.answer(item['question']), item['ground_truth']):
                score_update += 1
        acc_update = (score_update / len(data_update)) * 100
        score_multihop = 0
        time_costs = []
        for item in data_multihop:
            agent = NSCMAAgent(engine, **config)
            raw_stream = [s.strip() for s in item['context'].split('.') if len(s.strip()) > 3]
            chunks = chunk_text_stream(raw_stream, chunk_size=15)
            start_t = time.time()
            for text in chunks:
                agent.add_memory(text) 
            ans = agent.answer(item['question']) 
            end_t = time.time()
            time_costs.append(end_t - start_t)
            if check_answer(ans, item['ground_truth']):
                score_multihop += 1
        acc_multihop = (score_multihop / len(data_multihop)) * 100
        avg_time = np.mean(time_costs) if time_costs else 0.0
        print(f"{name:<20} | {acc_multihop:<6.1f}%            | {acc_update:<6.1f}%         | {avg_time:.2f}s")

if __name__ == "__main__":
    try:
        print("Initializing Engine...")
        engine = ModelEngine()
        run_experiment_3_ablation(engine)
        print("\nAll Experiments Completed Successfully! 🎉")
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        traceback.print_exc()