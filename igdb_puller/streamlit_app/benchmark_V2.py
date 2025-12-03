"""
Benchmark V2: Compare Option A (pre-computed) vs Option B (real-time) recommendations.

Run this locally to decide which approach to use for production.

Usage:
    1. First run data_pipeline_V2.py to generate processed data
    2. Set USE_LOCAL_DATA=true environment variable
    3. Run: python benchmark_V2.py

Results will help you decide:
- If real-time recommendations are fast enough (<500ms), you can skip storing recommendations.parquet
- If pre-computed is significantly faster, stick with the full pipeline
"""

import time
import random
import statistics
import os
import sys

# Ensure we use local data for benchmarking
os.environ["USE_LOCAL_DATA"] = "true"

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_benchmarks():
    """Run all benchmarks."""
    
    # Import after setting environment
    from s3_loader_V2 import (
        load_games, 
        load_recommendations, 
        load_embeddings,
        search_games
    )
    from content_recommender_V2 import get_recommendations
    
    print("="*70)
    print("BENCHMARK V2: Pre-computed vs Real-time Recommendations")
    print("="*70)
    
    # =========================================================================
    # 1. Data Loading Benchmark
    # =========================================================================
    print("\n" + "="*70)
    print("1. DATA LOADING BENCHMARK")
    print("="*70)
    
    # Clear any cached data
    load_games.clear()
    load_recommendations.clear()
    load_embeddings.clear()
    
    # Games
    start = time.time()
    games = load_games()
    games_time = time.time() - start
    print(f"\nGames ({len(games):,} rows)")
    print(f"  Load time: {games_time:.2f}s")
    print(f"  Memory: ~{games.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    
    if games.empty:
        print("\n❌ No games data found. Run data_pipeline_V2.py first.")
        return
    
    # Recommendations
    start = time.time()
    recs = load_recommendations()
    recs_time = time.time() - start
    print(f"\nRecommendations ({len(recs):,} rows)")
    print(f"  Load time: {recs_time:.2f}s")
    if not recs.empty:
        print(f"  Memory: ~{recs.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    else:
        print("  ⚠️ Not available")
    
    # Embeddings
    start = time.time()
    embeddings, id_map = load_embeddings()
    emb_time = time.time() - start
    print(f"\nEmbeddings")
    if embeddings.size > 0:
        print(f"  Shape: {embeddings.shape}")
        print(f"  Load time: {emb_time:.2f}s")
        print(f"  Memory: ~{embeddings.nbytes / 1024 / 1024:.1f} MB")
    else:
        print("  ⚠️ Not available")
    
    # =========================================================================
    # 2. Search Benchmark
    # =========================================================================
    print("\n" + "="*70)
    print("2. SEARCH BENCHMARK")
    print("="*70)
    
    n_queries = 50
    sample_names = games["name"].dropna().sample(min(n_queries, len(games))).tolist()
    queries = [str(name)[:5] for name in sample_names]  # Use first 5 chars
    
    # Clear cache before benchmark
    search_games.clear()
    
    times = []
    for query in queries:
        start = time.time()
        results = search_games(query, limit=20)
        elapsed = time.time() - start
        times.append(elapsed)
    
    print(f"\nQueries tested: {n_queries}")
    print(f"  Mean time:   {statistics.mean(times)*1000:.2f} ms")
    print(f"  Median time: {statistics.median(times)*1000:.2f} ms")
    print(f"  Min time:    {min(times)*1000:.2f} ms")
    print(f"  Max time:    {max(times)*1000:.2f} ms")
    print(f"  Std dev:     {statistics.stdev(times)*1000:.2f} ms")
    
    if statistics.mean(times) < 0.1:
        print("  ✅ Search is very fast (<100ms)")
    elif statistics.mean(times) < 0.5:
        print("  ✅ Search is acceptable (<500ms)")
    else:
        print("  ⚠️ Search is slow (>500ms)")
    
    # =========================================================================
    # 3. Recommendation Benchmark
    # =========================================================================
    print("\n" + "="*70)
    print("3. RECOMMENDATION BENCHMARK")
    print("="*70)
    
    n_games = 30
    
    # Get sample game IDs
    if not recs.empty:
        valid_ids = recs["game_id"].tolist()
        sample_ids = random.sample(valid_ids, min(n_games, len(valid_ids)))
    else:
        sample_ids = games["id"].sample(min(n_games, len(games))).tolist()
    
    # Option A: Pre-computed
    if not recs.empty:
        print(f"\n--- Option A: Pre-computed Recommendations ---")
        times_a = []
        for game_id in sample_ids:
            _, elapsed = get_recommendations(game_id, top_n=10, method="precomputed")
            times_a.append(elapsed)
        
        print(f"Games tested: {n_games}")
        print(f"  Mean time:   {statistics.mean(times_a)*1000:.2f} ms")
        print(f"  Median time: {statistics.median(times_a)*1000:.2f} ms")
        print(f"  Min time:    {min(times_a)*1000:.2f} ms")
        print(f"  Max time:    {max(times_a)*1000:.2f} ms")
    else:
        print("\n⚠️ Skipping Option A (no pre-computed recommendations)")
        times_a = None
    
    # Option B: Real-time
    if embeddings.size > 0:
        print(f"\n--- Option B: Real-time Recommendations ---")
        times_b = []
        for game_id in sample_ids:
            try:
                _, elapsed = get_recommendations(game_id, top_n=10, method="realtime")
                times_b.append(elapsed)
            except Exception as e:
                print(f"  Error for game {game_id}: {e}")
        
        if times_b:
            print(f"Games tested: {len(times_b)}")
            print(f"  Mean time:   {statistics.mean(times_b)*1000:.2f} ms")
            print(f"  Median time: {statistics.median(times_b)*1000:.2f} ms")
            print(f"  Min time:    {min(times_b)*1000:.2f} ms")
            print(f"  Max time:    {max(times_b)*1000:.2f} ms")
    else:
        print("\n⚠️ Skipping Option B (no embeddings)")
        times_b = None
    
    # =========================================================================
    # 4. Summary & Recommendations
    # =========================================================================
    print("\n" + "="*70)
    print("4. SUMMARY & RECOMMENDATIONS")
    print("="*70)
    
    print("\nData Loading:")
    print(f"  Games:           {games_time:.2f}s")
    print(f"  Recommendations: {recs_time:.2f}s")
    print(f"  Embeddings:      {emb_time:.2f}s")
    print(f"  Total cold start: {games_time + recs_time:.2f}s (with caching)")
    
    print("\nRecommendation Methods:")
    if times_a:
        mean_a = statistics.mean(times_a) * 1000
        print(f"  Option A (pre-computed): {mean_a:.2f} ms average")
    if times_b:
        mean_b = statistics.mean(times_b) * 1000
        print(f"  Option B (real-time):    {mean_b:.2f} ms average")
    
    if times_a and times_b:
        speedup = statistics.mean(times_b) / statistics.mean(times_a)
        print(f"\n  Pre-computed is {speedup:.1f}x faster than real-time")
    
    print("\n" + "-"*70)
    print("RECOMMENDATION:")
    print("-"*70)
    
    if times_b and statistics.mean(times_b) < 0.3:
        print("""
✅ Real-time recommendations are fast enough (<300ms)!

You can SKIP uploading recommendations.parquet to S3.
Just upload:
  - games_enriched.parquet
  - embeddings.npy
  - game_id_map.parquet

Set the app to use 'realtime' method by default.
        """)
    elif times_a:
        print("""
✅ Use pre-computed recommendations for best performance.

Upload all files to S3:
  - games_enriched.parquet
  - recommendations.parquet
  - embeddings.npy (optional, for benchmarking)
  - game_id_map.parquet (optional, for benchmarking)

The app will use 'precomputed' method by default.
        """)
    else:
        print("""
⚠️ Could not benchmark recommendations.

Make sure you've run data_pipeline_V2.py first to generate:
  - processed_data/games_enriched.parquet
  - processed_data/recommendations.parquet
  - processed_data/embeddings.npy
  - processed_data/game_id_map.parquet
        """)


if __name__ == "__main__":
    run_benchmarks()
