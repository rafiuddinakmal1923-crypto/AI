
from __future__ import annotations

from heapq import heappop, heappush
from math import inf
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                     # save figures to file, no pop-up window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import astar, gbfs, ucs
from utils.search_runner import QUERIES_FILE, load_default_graph, make_heuristic

# ---------------------------------------------------------------------------
# Paths are built from this file's location, so it works from any terminal folder.
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent            # .../AI/all_experiments
ROOT = HERE.parent                                # .../AI
RESULTS = ROOT / "results" / "lab02"

WEIGHT = 1.5                                      # inflation factor for weighted A*
TOLERANCE_KM = 1e-6                               # floating-point slack for "h <= true cost"


def heading(text: str) -> None:
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Part 1 - run the four searches
# ---------------------------------------------------------------------------
def run_searches(graph: dict, coords: dict, start: str, goal: str) -> dict:
    """Run UCS, GBFS, A* and weighted A* for one query. Returns {name: SearchResult}."""
    h = make_heuristic(coords, goal)                       # h(n) = great-circle km to goal
    h_weighted = make_heuristic(coords, goal, WEIGHT)      # 1.5 x h(n)
    return {
        "UCS": ucs(graph, start, goal),                    # f = g        (no heuristic)
        "GBFS": gbfs(graph, start, goal, h),               # f = h        (ignores cost so far)
        "A*": astar(graph, start, goal, h),                # f = g + h
        f"Weighted A* (w={WEIGHT})": astar(graph, start, goal, h_weighted),  # f = g + 1.5h
    }


def results_to_rows(query_id: str, start: str, goal: str, results: dict) -> list[dict]:
    """Turn the SearchResults of one query into table rows, adding gap_pct vs UCS."""
    optimum = results["UCS"].path_cost_km                  # UCS is guaranteed cheapest
    rows = []
    for name, result in results.items():
        rows.append({
            "query_id": query_id,
            "start": start,
            "goal": goal,
            "algorithm": name,
            "path": " -> ".join(result.path),
            "hops": result.hops,
            "cost_km": result.path_cost_km,
            "expanded": result.expanded,
            "max_frontier": result.max_frontier,
            "runtime_ms": result.runtime_ms,
            # 0 % means the route is optimal; 9 % means 9 % more kilometres than needed.
            "gap_pct": (result.path_cost_km - optimum) / optimum * 100,
        })
    return rows


# ---------------------------------------------------------------------------
# Part 2 - admissibility check
# ---------------------------------------------------------------------------
def reverse_graph(graph: dict) -> dict:
    """Flip every edge: A -> B (cost c) becomes B -> A (cost c)."""
    reversed_edges = {state: [] for state in graph}
    for source, neighbours in graph.items():
        for destination, cost in neighbours:
            reversed_edges[destination].append((source, cost))
    return reversed_edges


def true_cost_to_goal(graph: dict, goal: str) -> dict:
    """Dijkstra from the goal on the REVERSED graph.

    A shortest path X -> goal in the real graph is a shortest path goal -> X in
    the reversed graph, so one Dijkstra run gives the true cheapest cost h*(n)
    from every airport n to the goal. Airports that cannot reach the goal are
    simply missing from the returned dictionary.
    """
    backwards = reverse_graph(graph)
    distance = {goal: 0.0}
    frontier = [(0.0, goal)]
    done = set()
    while frontier:
        d, state = heappop(frontier)
        if state in done:
            continue
        done.add(state)
        for neighbour, cost in backwards[state]:
            new_d = d + cost
            if new_d < distance.get(neighbour, inf):
                distance[neighbour] = new_d
                heappush(frontier, (new_d, neighbour))
    return distance


def check_heuristic(graph: dict, coords: dict, query_id: str, goal: str):
    """Compare h(n) with the true cost h*(n) for every airport that can reach the goal.

    Returns (summary_row, violations_dataframe).
    """
    h = make_heuristic(coords, goal)
    true_cost = true_cost_to_goal(graph, goal)

    rows = []
    for airport, h_star in true_cost.items():
        if airport == goal:
            continue                                      # h = h* = 0 at the goal itself
        h_value = h(airport)
        rows.append({
            "airport": airport,
            "h_km": h_value,
            "true_km": h_star,
            "ratio": h_value / h_star,                    # 1.0 = perfect, close to 0 = weak
            "admissible": h_value <= h_star + TOLERANCE_KM,
        })
    table = pd.DataFrame(rows)

    # Consistency (a stronger property): h(n) <= cost(n, m) + h(m) for every edge.
    # The search code never re-opens a closed state, so A* needs this to stay optimal.
    inconsistent_edges = sum(
        1
        for n, neighbours in graph.items()
        for m, cost in neighbours
        if h(n) > cost + h(m) + TOLERANCE_KM
    )

    summary = {
        "query_id": query_id,
        "goal": goal,
        "reachable_airports": len(table),
        "violations": int((~table["admissible"]).sum()),
        "mean_h_over_true": table["ratio"].mean(),
        "median_h_over_true": table["ratio"].median(),
        "min_h_over_true": table["ratio"].min(),
        "max_h_over_true": table["ratio"].max(),
        "exact_airports": int(np.isclose(table["ratio"], 1.0).sum()),  # h equals true cost
        "inconsistent_edges": inconsistent_edges,
    }
    violations = table[~table["admissible"]].assign(query_id=query_id, goal=goal)
    return summary, violations


# ---------------------------------------------------------------------------
# Part 3 - chart
# ---------------------------------------------------------------------------
def plot_expanded(metrics: pd.DataFrame, path: Path) -> None:
    """Grouped bar chart: nodes expanded per algorithm, one group per query."""
    table = metrics.pivot(index="query_id", columns="algorithm", values="expanded")
    table = table[metrics["algorithm"].unique()]          # keep the run order in the legend

    figure, axis = plt.subplots(figsize=(9, 5))
    x = np.arange(len(table.index))
    width = 0.8 / len(table.columns)
    for i, algorithm in enumerate(table.columns):
        bars = axis.bar(x + i * width, table[algorithm], width, label=algorithm)
        axis.bar_label(bars, fontsize=8)
    labels = [f"{q}\n{s}->{g}" for q, s, g in
              metrics.drop_duplicates("query_id")[["query_id", "start", "goal"]].itertuples(index=False)]
    axis.set_xticks(x + width * (len(table.columns) - 1) / 2, labels)
    axis.set_yscale("log")             # UCS expands thousands, GBFS only a handful
    axis.set_ylabel("Nodes expanded (log scale)")
    axis.set_title("Lab 2: how much work each search does")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    heading("Question 1 - Load the graph and the queries")
    coords, graph, stats = load_default_graph()
    queries = pd.read_csv(QUERIES_FILE).head(3)
    print(pd.Series(stats, name="count").to_frame())
    print(queries.to_string(index=False))

    heading("Question 2 - UCS vs GBFS vs A* vs weighted A*")
    rows = []
    for query in queries.itertuples(index=False):
        results = run_searches(graph, coords, query.start, query.goal)
        rows += results_to_rows(query.query_id, query.start, query.goal, results)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "metrics.csv", index=False)
    print(metrics.drop(columns=["start", "goal", "path"]).round(2).to_string(index=False))

    heading("Question 3 - Is the haversine heuristic admissible?")
    summaries, all_violations = [], []
    for query in queries.itertuples(index=False):
        summary, violations = check_heuristic(graph, coords, query.query_id, query.goal)
        summaries.append(summary)
        all_violations.append(violations)
    check = pd.DataFrame(summaries)
    check.to_csv(RESULTS / "heuristic_check.csv", index=False)
    print(check.round(3).to_string(index=False))

    violations = pd.concat(all_violations, ignore_index=True)
    if violations.empty:
        print("\nNo violations: h(n) <= true cost for every airport that can reach each goal.")
    else:
        violations.to_csv(RESULTS / "heuristic_violations.csv", index=False)
        print(f"\n{len(violations)} violations found - saved to heuristic_violations.csv")
        print(violations.head(10).to_string(index=False))

    heading("Question 4 - Chart of nodes expanded")
    plot_expanded(metrics, RESULTS / "expanded_nodes.png")
    print(f"Saved {RESULTS / 'expanded_nodes.png'}")

    heading("Summary")
    for query_id, group in metrics.groupby("query_id", sort=False):
        ucs_row = group[group["algorithm"] == "UCS"].iloc[0]
        astar_row = group[group["algorithm"] == "A*"].iloc[0]
        saving = 1 - astar_row["expanded"] / ucs_row["expanded"]
        print(f"{query_id}: A* found the same {astar_row['cost_km']:.0f} km route as UCS "
              f"but expanded {saving:.0%} fewer airports "
              f"({astar_row['expanded']} vs {ucs_row['expanded']}).")
    print(f"\nAll files saved in {RESULTS}")


if __name__ == "__main__":
    main()
