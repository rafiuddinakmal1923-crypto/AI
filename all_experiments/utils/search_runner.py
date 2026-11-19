"""Small helpers shared by Lab 2, Lab 3 and the command-line tool (python -m utils).

They sit on top of the existing search code in core.py and openflights.py:
    * load_default_graph()  - load the OpenFlights files from all_datasets/Lab 1-3/
    * make_heuristic()      - straight-line (great-circle) distance to the goal
    * run_algorithm()       - run one search by name, e.g. run_algorithm("astar", ...)
"""
from __future__ import annotations

from pathlib import Path

from .core import astar, bfs, dfs, gbfs, ucs
from .openflights import haversine_km, load_openflights

# Paths are built from this file's location, so they work from any terminal folder.
UTILS_DIR = Path(__file__).resolve().parent                   # .../AI/all_experiments/utils
PROJECT_ROOT = UTILS_DIR.parent.parent                        # .../AI
DEFAULT_DATA_DIR = PROJECT_ROOT / "all_datasets" / "Lab 1-3"
QUERIES_FILE = UTILS_DIR / "search_queries.csv"

# Display names used in every results table, in the order we want them listed.
ALGORITHM_NAMES = {
    "bfs": "BFS",
    "dfs": "DFS",
    "ucs": "UCS",
    "gbfs": "GBFS",
    "astar": "A*",
}


def load_default_graph(data_dir: str | Path = DEFAULT_DATA_DIR):
    """Load the airline graph. Returns (coords, graph, stats).

    coords = {"KUL": {"latitude": 2.7, "longitude": 101.7}, ...}
    graph  = {"KUL": [("AMS", 10236.7), ...], ...}   (edge cost = km)
    """
    data_dir = Path(data_dir)
    _, _, coords, graph, stats = load_openflights(
        data_dir / "lab01-03_airports.csv",
        data_dir / "routes.csv",
    )
    return coords, graph, stats


def make_heuristic(coords: dict, goal: str, weight: float = 1.0):
    """Return h(state) = weight x great-circle distance from state to goal (km).

    With weight = 1 this never overestimates: no chain of flights can be shorter
    than the straight line over the Earth's surface, so A* stays optimal.
    With weight > 1 (weighted A*) we trade that guarantee for fewer expansions.
    """
    goal_lat = coords[goal]["latitude"]
    goal_lon = coords[goal]["longitude"]
    cache = {}                        # each airport's distance is only computed once

    def h(state):
        if state not in cache:
            here = coords[state]
            cache[state] = haversine_km(here["latitude"], here["longitude"], goal_lat, goal_lon)
        return weight * cache[state]

    return h


def run_algorithm(name: str, graph: dict, coords: dict, start: str, goal: str, weight: float = 1.0):
    """Run one search by its short name ('bfs', 'dfs', 'ucs', 'gbfs' or 'astar')."""
    name = name.lower()
    if name == "bfs":
        return bfs(graph, start, goal)
    if name == "dfs":
        return dfs(graph, start, goal)
    if name == "ucs":
        return ucs(graph, start, goal)
    if name == "gbfs":
        return gbfs(graph, start, goal, make_heuristic(coords, goal))
    if name == "astar":
        return astar(graph, start, goal, make_heuristic(coords, goal, weight))
    raise ValueError(f"Unknown algorithm {name!r}. Choose from: {', '.join(ALGORITHM_NAMES)}")
