"""
Lab 3 of 8 - Search Engineering and Benchmarking
Objective: benchmark all five search algorithms (BFS, DFS, UCS, GBFS, A*) on
the airline graph in a reproducible way, and check the results make sense.

What this script does
    1. Builds the list of test queries: all of search_queries.csv plus extra
       random origin/destination pairs (fixed seed, so the same pairs every run).
    2. Runs every algorithm on every query and saves one row per run to
       results/lab03/benchmark.csv.
    3. Sanity-checks the results (valid routes, A* cost == UCS cost, BFS uses
       the fewest hops).
    4. Summarises per algorithm in results/lab03/benchmark_summary.csv and
       draws results/lab03/benchmark_expansions.png.

Companion tools:
    python -m utils --algorithm astar --start KUL --goal KEF    (command-line route finder)
    python -m pytest utils -q                                   (unit tests, run inside all_experiments/)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")                     # save figures to file, no pop-up window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import route_is_valid
from utils.search_runner import ALGORITHM_NAMES, QUERIES_FILE, load_default_graph, run_algorithm

# ---------------------------------------------------------------------------
# Settings. Change N_RANDOM_PAIRS or SEED to try a different benchmark.
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent            # .../AI/all_experiments
ROOT = HERE.parent                                # .../AI
RESULTS = ROOT / "results" / "lab03"

SEED = 2023                                       # fixed seed -> same random pairs every run
N_RANDOM_PAIRS = 10                               # extra pairs on top of the 10 in the query file
COST_TOLERANCE_KM = 1e-6                          # floating-point slack when comparing costs


def heading(text: str) -> None:
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def build_queries(graph: dict, n_random: int, seed: int) -> pd.DataFrame:
    """The query file plus n_random reproducible random pairs."""
    fixed = pd.read_csv(QUERIES_FILE).assign(source="query_file")

    # Only pick airports that actually have flights: a start needs an outgoing
    # route and a goal needs an incoming route, otherwise the pair is pointless.
    has_outgoing = sorted(a for a, neighbours in graph.items() if neighbours)
    has_incoming = sorted({b for neighbours in graph.values() for b, _ in neighbours})

    rng = np.random.default_rng(seed)
    already = set(zip(fixed["start"], fixed["goal"]))
    pairs = []
    while len(pairs) < n_random:
        start = has_outgoing[rng.integers(len(has_outgoing))]
        goal = has_incoming[rng.integers(len(has_incoming))]
        if start != goal and (start, goal) not in already:
            already.add((start, goal))
            pairs.append({"query_id": f"R{len(pairs) + 1:02d}", "start": start,
                          "goal": goal, "source": "random"})
    return pd.concat([fixed, pd.DataFrame(pairs)], ignore_index=True)


def run_benchmark(graph: dict, coords: dict, queries: pd.DataFrame) -> pd.DataFrame:
    """One row per (query, algorithm) run."""
    rows = []
    for query in queries.itertuples(index=False):
        for short_name, display_name in ALGORITHM_NAMES.items():
            result = run_algorithm(short_name, graph, coords, query.start, query.goal)
            rows.append({
                "query_id": query.query_id,
                "source": query.source,
                "start": query.start,
                "goal": query.goal,
                "algorithm": display_name,
                "found": result.found,
                "path": " -> ".join(result.path),
                "hops": result.hops if result.found else np.nan,
                "cost_km": result.path_cost_km if result.found else np.nan,
                "expanded": result.expanded,
                "max_frontier": result.max_frontier,
                "runtime_ms": result.runtime_ms,
            })
    runs = pd.DataFrame(rows)

    # gap_pct: how much longer than the optimum (UCS) each route is.
    optimum = runs[runs["algorithm"] == "UCS"].set_index("query_id")["cost_km"]
    runs["gap_pct"] = (runs["cost_km"] / runs["query_id"].map(optimum) - 1) * 100
    return runs


def sanity_checks(runs: pd.DataFrame, graph: dict) -> None:
    """Engineering checks: if any of these fail, something is wrong with the code."""
    found = runs[runs["found"]]

    # 1. Every route only uses real flights.
    for row in found.itertuples():
        assert route_is_valid(row.path.split(" -> "), graph), f"{row.algorithm} {row.query_id}: invalid route"
    print(f"[ok] all {len(found)} routes use real flights only")

    # 2. All algorithms agree on whether a route exists (they are all complete on a finite graph).
    agreement = runs.groupby("query_id")["found"].nunique().eq(1).all()
    assert agreement, "algorithms disagree about whether a route exists"
    print("[ok] all algorithms agree on which queries are solvable")

    by_query = found.pivot(index="query_id", columns="algorithm", values=["cost_km", "hops"])

    # 3. A* with an admissible heuristic returns the optimal (UCS) cost.
    cost_gap = (by_query["cost_km"]["A*"] - by_query["cost_km"]["UCS"]).abs()
    assert (cost_gap < COST_TOLERANCE_KM).all(), "A* cost differs from UCS cost"
    print("[ok] A* cost equals UCS cost on every solvable query")

    # 4. BFS finds the route with the fewest flights.
    fewest = by_query["hops"].min(axis=1)
    assert (by_query["hops"]["BFS"] == fewest).all(), "another algorithm used fewer hops than BFS"
    print("[ok] BFS uses the fewest hops on every solvable query")


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per algorithm. Cost/hops are averaged over the successful runs only."""
    summary = runs.groupby("algorithm", sort=False).agg(
        runs=("found", "size"),
        success_rate=("found", "mean"),
        mean_cost_km=("cost_km", "mean"),
        mean_hops=("hops", "mean"),
        mean_expanded=("expanded", "mean"),
        median_runtime_ms=("runtime_ms", "median"),
        mean_gap_pct=("gap_pct", "mean"),
        optimal_rate=("gap_pct", lambda gap: (gap.dropna().abs() < 1e-9).mean()),
    )
    return summary.reset_index()


def plot_expansions(runs: pd.DataFrame, path: Path) -> None:
    """Box plot of nodes expanded per algorithm (one point per query)."""
    order = list(ALGORITHM_NAMES.values())
    data = [runs.loc[runs["algorithm"] == name, "expanded"] for name in order]

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.boxplot(data, showmeans=True)
    axis.set_xticks(range(1, len(order) + 1), order)
    for i, values in enumerate(data, start=1):          # show each run as a dot too
        jitter = np.random.default_rng(i).uniform(-0.12, 0.12, len(values))
        axis.scatter(np.full(len(values), i) + jitter, values, s=12, alpha=0.5)
    axis.set_yscale("log")
    axis.set_ylabel("Nodes expanded per query (log scale)")
    axis.set_title(f"Lab 3: expansions over {runs['query_id'].nunique()} queries "
                   f"(triangle = mean)")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    heading("Question 1 - Build a reproducible query set")
    coords, graph, _ = load_default_graph()
    queries = build_queries(graph, N_RANDOM_PAIRS, SEED)
    print(f"{len(queries)} queries ({N_RANDOM_PAIRS} random, seed={SEED}):")
    print(queries.to_string(index=False))

    heading("Question 2 - Run every algorithm on every query")
    runs = run_benchmark(graph, coords, queries)
    runs.to_csv(RESULTS / "benchmark.csv", index=False)
    print(f"{len(runs)} runs saved to benchmark.csv")
    print(runs.pivot(index="query_id", columns="algorithm", values="expanded")
              [list(ALGORITHM_NAMES.values())].loc[queries["query_id"]].to_string())

    heading("Question 3 - Sanity checks")
    sanity_checks(runs, graph)

    heading("Question 4 - Per-algorithm summary")
    summary = summarise(runs)
    summary.to_csv(RESULTS / "benchmark_summary.csv", index=False)
    print(summary.round(2).to_string(index=False))

    plot_expansions(runs, RESULTS / "benchmark_expansions.png")
    print(f"\nAll files saved in {RESULTS}")


if __name__ == "__main__":
    main()
