#!/usr/bin/env python3
"""
Benchmark for SemanticMatcher
This script measures the performance of various operations in the SemanticMatcher class.
"""

import sys
import os
import time
import random
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Tuple
from tqdm import tqdm

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.semantic_matcher import SemanticMatcher, ModelConfig

def generate_test_skills(count: int = 1000) -> List[str]:
    """Generate test skills for benchmarking."""
    domains = [
        "programming", "data science", "machine learning", "web development",
        "mobile development", "devops", "cloud computing", "cybersecurity",
        "database", "project management", "design", "marketing"
    ]
    
    skills = []
    for _ in range(count):
        domain = random.choice(domains)
        skill = f"{domain} {random.randint(1, 100)}"
        skills.append(skill)
    
    return skills

def benchmark_embedding_batch_vs_individual(skills: List[str]) -> Dict[str, Any]:
    """Benchmark batch embedding vs individual embedding."""
    results = {"batch": {}, "individual": {}}
    
    # Warmup
    print("Warming up...")
    SemanticMatcher.get_model()
    
    # Benchmark batch embedding
    print("Benchmarking batch embedding...")
    batch_sizes = [10, 50, 100, 500]
    
    for batch_size in batch_sizes:
        if batch_size > len(skills):
            continue
            
        sample_skills = skills[:batch_size]
        
        # Time batch processing
        start_time = time.time()
        embeddings = SemanticMatcher.get_embeddings_batch(sample_skills)
        end_time = time.time()
        
        batch_time = end_time - start_time
        results["batch"][batch_size] = {
            "time": batch_time,
            "per_skill": batch_time / batch_size
        }
        
        print(f"Batch size {batch_size}: {batch_time:.2f}s ({batch_time/batch_size:.4f}s per skill)")
        
        # Time individual processing
        start_time = time.time()
        for skill in sample_skills:
            SemanticMatcher.get_embedding(skill)
        end_time = time.time()
        
        individual_time = end_time - start_time
        results["individual"][batch_size] = {
            "time": individual_time,
            "per_skill": individual_time / batch_size
        }
        
        print(f"Individual for {batch_size} skills: {individual_time:.2f}s ({individual_time/batch_size:.4f}s per skill)")
        print(f"Speedup: {individual_time/batch_time:.2f}x")
        
    return results

def benchmark_model_warmup() -> Dict[str, Any]:
    """Benchmark model warmup vs cold start."""
    results = {"with_warmup": {}, "without_warmup": {}}
    
    # Generate a small test set
    test_skills = generate_test_skills(count=50)
    
    # Test with warmup_on_init=False
    print("Testing without warmup...")
    SemanticMatcher._model = None  # Reset model
    SemanticMatcher._embedding_cache = {}  # Reset cache
    SemanticMatcher.configure_model(ModelConfig(warmup_on_init=False))
    
    # First-time model load (cold start)
    start_time = time.time()
    for skill in test_skills:
        SemanticMatcher.get_embedding(skill)
    end_time = time.time()
    
    cold_start_time = end_time - start_time
    results["without_warmup"]["first_run"] = cold_start_time
    print(f"Cold start time: {cold_start_time:.2f}s")
    
    # Second run with model already loaded
    start_time = time.time()
    for skill in test_skills:
        SemanticMatcher.get_embedding(skill)
    end_time = time.time()
    
    second_run_time = end_time - start_time
    results["without_warmup"]["second_run"] = second_run_time
    print(f"Second run time: {second_run_time:.2f}s")
    
    # Test with warmup_on_init=True
    print("Testing with warmup...")
    SemanticMatcher._model = None  # Reset model
    SemanticMatcher._embedding_cache = {}  # Reset cache
    SemanticMatcher.configure_model(ModelConfig(warmup_on_init=True))
    
    # First-time model load (with warmup)
    start_time = time.time()
    for skill in test_skills:
        SemanticMatcher.get_embedding(skill)
    end_time = time.time()
    
    warmup_first_run_time = end_time - start_time
    results["with_warmup"]["first_run"] = warmup_first_run_time
    print(f"First run with warmup: {warmup_first_run_time:.2f}s")
    
    # Second run with model already loaded
    start_time = time.time()
    for skill in test_skills:
        SemanticMatcher.get_embedding(skill)
    end_time = time.time()
    
    warmup_second_run_time = end_time - start_time
    results["with_warmup"]["second_run"] = warmup_second_run_time
    print(f"Second run with warmup: {warmup_second_run_time:.2f}s")
    
    return results

def benchmark_matching_methods() -> Dict[str, Any]:
    """Benchmark semantic vs fuzzy matching."""
    results = {"semantic": {}, "fuzzy": {}}
    
    # Generate test skills
    skills = generate_test_skills(count=100)
    reference_skills = generate_test_skills(count=1000)
    
    # Make sure model is loaded
    SemanticMatcher.get_model()
    
    # Benchmark semantic matching
    print("Benchmarking semantic matching...")
    start_time = time.time()
    for skill in tqdm(skills):
        SemanticMatcher.match_skill(skill, reference_skills, use_semantic=True)
    end_time = time.time()
    
    semantic_time = end_time - start_time
    results["semantic"]["time"] = semantic_time
    results["semantic"]["per_skill"] = semantic_time / len(skills)
    
    print(f"Semantic matching: {semantic_time:.2f}s ({semantic_time/len(skills):.4f}s per skill)")
    
    # Benchmark fuzzy matching
    print("Benchmarking fuzzy matching...")
    start_time = time.time()
    for skill in tqdm(skills):
        SemanticMatcher.match_skill(skill, reference_skills, use_semantic=False)
    end_time = time.time()
    
    fuzzy_time = end_time - start_time
    results["fuzzy"]["time"] = fuzzy_time
    results["fuzzy"]["per_skill"] = fuzzy_time / len(skills)
    
    print(f"Fuzzy matching: {fuzzy_time:.2f}s ({fuzzy_time/len(skills):.4f}s per skill)")
    print(f"Semantic vs Fuzzy: {semantic_time/fuzzy_time:.2f}x")
    
    return results

def plot_results(
    batch_results: Dict[str, Any], 
    warmup_results: Dict[str, Any], 
    matching_results: Dict[str, Any]
) -> None:
    """Plot benchmark results."""
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot batch vs individual
    batch_sizes = list(batch_results["batch"].keys())
    batch_times = [batch_results["batch"][size]["per_skill"] for size in batch_sizes]
    individual_times = [batch_results["individual"][size]["per_skill"] for size in batch_sizes]
    
    ax1.plot(batch_sizes, batch_times, 'o-', label='Batch')
    ax1.plot(batch_sizes, individual_times, 'o-', label='Individual')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Time per Skill (s)')
    ax1.set_title('Batch vs Individual Processing')
    ax1.grid(True)
    ax1.legend()
    
    # Plot warmup vs cold start
    labels = ['First Run', 'Second Run']
    no_warmup_times = [
        warmup_results["without_warmup"]["first_run"], 
        warmup_results["without_warmup"]["second_run"]
    ]
    warmup_times = [
        warmup_results["with_warmup"]["first_run"], 
        warmup_results["with_warmup"]["second_run"]
    ]
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax2.bar(x - width/2, no_warmup_times, width, label='Without Warmup')
    ax2.bar(x + width/2, warmup_times, width, label='With Warmup')
    ax2.set_xlabel('Run')
    ax2.set_ylabel('Time (s)')
    ax2.set_title('Effect of Model Warmup')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.grid(True)
    ax2.legend()
    
    # Plot semantic vs fuzzy
    methods = ['Semantic', 'Fuzzy']
    times = [
        matching_results["semantic"]["per_skill"],
        matching_results["fuzzy"]["per_skill"]
    ]
    
    ax3.bar(methods, times)
    ax3.set_xlabel('Matching Method')
    ax3.set_ylabel('Time per Skill (s)')
    ax3.set_title('Semantic vs Fuzzy Matching')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    print("Saved benchmark plot to benchmark_results.png")

def main():
    """Run all benchmarks."""
    print("Initializing SemanticMatcher...")
    SemanticMatcher.configure_model(ModelConfig(
        model_name="small",
        warmup_on_init=True,
        enable_progress_bars=True
    ))
    
    # Generate test skills
    print("Generating test skills...")
    test_skills = generate_test_skills(count=1000)
    
    # Run benchmarks
    print("\n=== Benchmarking Embedding Methods ===")
    batch_results = benchmark_embedding_batch_vs_individual(test_skills)
    
    print("\n=== Benchmarking Model Warmup ===")
    warmup_results = benchmark_model_warmup()
    
    print("\n=== Benchmarking Matching Methods ===")
    matching_results = benchmark_matching_methods()
    
    # Plot results
    print("\n=== Generating Plots ===")
    plot_results(batch_results, warmup_results, matching_results)
    
    # Save all results to JSON
    print("\n=== Saving Results ===")
    all_results = {
        "batch_vs_individual": batch_results,
        "model_warmup": warmup_results,
        "matching_methods": matching_results
    }
    
    with open("benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("Saved benchmark results to benchmark_results.json")
    print("\nBenchmark complete!")

if __name__ == "__main__":
    main() 