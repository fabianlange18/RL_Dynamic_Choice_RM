"""
Memory consumption monitoring for BiogemeEstimator.

Tracks peak and average memory usage during estimation methods.
"""

import os
import gc
import psutil
import time
import tracemalloc
from contextlib import contextmanager

import config as c
import constants as C

_pytensor_flags = os.environ.get("PYTENSOR_FLAGS", "")
if "cxx=" not in _pytensor_flags:
    os.environ["PYTENSOR_FLAGS"] = f"{_pytensor_flags},cxx=".strip(",")

from estimation_biogeme import BiogemeEstimator, collect_transaction_data
from env_example_2 import TalluriExample2


@contextmanager
def memory_tracker(label: str):
    """Context manager to track memory usage of a block of code.
    
    Prints peak memory, average memory, and execution time.
    """
    gc.collect()
    tracemalloc.start()
    process = psutil.Process(os.getpid())
    
    mem_start = process.memory_info().rss / 1024 / 1024  # MB
    time_start = time.perf_counter()
    
    peak_mem = mem_start
    try:
        yield
    finally:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        time_elapsed = time.perf_counter() - time_start
        mem_end = process.memory_info().rss / 1024 / 1024  # MB
        peak_mem_traced = peak / 1024 / 1024  # Convert bytes to MB
        
        mem_delta = mem_end - mem_start
        
        print(f"\n{'=' * 70}")
        print(f"Memory Monitor: {label}")
        print(f"{'=' * 70}")
        print(f"Memory at start:    {mem_start:>10.2f} MB")
        print(f"Memory at end:      {mem_end:>10.2f} MB")
        print(f"Delta:              {mem_delta:>10.2f} MB")
        print(f"Peak (tracemalloc): {peak_mem_traced:>10.2f} MB")
        print(f"Time elapsed:       {time_elapsed:>10.4f} seconds")
        print(f"{'=' * 70}\n")


def estimate_with_monitoring():
    """Run all estimations and monitor memory consumption."""
    
    print("\n" + "="*70)
    print("BIOGEME ESTIMATOR - MEMORY MONITORING")
    print("="*70)
    
    # Collect data
    print("\n[1/5] Collecting transaction data...")
    with memory_tracker("Data Collection"):
        _env = TalluriExample2(efficient_sets=None)
        observations = collect_transaction_data(_env)
        _env.close()
    
    n_observations = len(observations)
    print(f"Collected {n_observations} observations")
    
    # Initialize estimator
    print("\n[2/5] Initializing BiogemeEstimator...")
    with memory_tracker("Estimator Initialization (database + preprocessing)"):
        estimator = BiogemeEstimator(observations)
    
    print(f"Database created with {len(estimator._dataframe)} rows")
    
    # Run each estimation
    print("\n[3/5] Running MNL estimation...")
    with memory_tracker("MNL Estimation"):
        mnl_result = estimator.estimate_mnl()
    print(f"MNL Beta: {mnl_result['beta']:.6f}, Lambda: {mnl_result['lambda']:.6f}")
    
    print("\n[4/5] Running MMNL (5-point) estimation...")
    with memory_tracker("MMNL 5-Point Estimation"):
        mmnl_5pt_result = estimator.estimate_mmnl(K=5)
    print(f"MMNL Segments: {mmnl_5pt_result['n_segments']}, Lambda: {mmnl_5pt_result['lambda']:.6f}")
    
    print("\n[5/5] Running MMNL (continuous) estimation...")
    with memory_tracker("MMNL Continuous Estimation"):
        mmnl_cont_result = estimator.estimate_mmnl_continuous()
    print(f"MMNL mu_b: {mmnl_cont_result['mu_b']:.6f}, sigma_b: {mmnl_cont_result['sigma_b']:.6f}")
    
    # Summary
    print("\n" + "="*70)
    print("ESTIMATION SUMMARY")
    print("="*70)
    print(f"MNL   LL: {mnl_result['final_log_likelihood']:>12.4f}")
    print(f"MMNL  LL: {mmnl_5pt_result['final_log_likelihood']:>12.4f}")
    print(f"Cont  LL: {mmnl_cont_result['final_log_likelihood']:>12.4f}")
    print("="*70 + "\n")
    
    del observations, estimator
    gc.collect()


if __name__ == "__main__":
    estimate_with_monitoring()
