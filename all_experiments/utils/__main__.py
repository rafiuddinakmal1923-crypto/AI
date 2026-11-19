"""Command-line route finder (Lab 3).

Run from the all_experiments folder:
    python -m utils --algorithm astar --start KUL --goal KEF
    python -m utils --algorithm bfs --start SIN --goal EZE
    python -m utils --help

Python runs this file when you type `python -m utils`, because it is the
package's __main__.py.
"""
from __future__ import annotations

import argparse
import difflib
import sys

from .search_runner import ALGORITHM_NAMES, DEFAULT_DATA_DIR, load_default_graph, run_algorithm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m utils",
        description="Find a flight route between two airports on the OpenFlights graph.",
    )
    parser.add_argument("--algorithm", "-a", default="astar", choices=list(ALGORITHM_NAMES),
                        help="search algorithm to use (default: astar)")
    parser.add_argument("--start", "-s", required=True, help="3-letter IATA code, e.g. KUL")
    parser.add_argument("--goal", "-g", required=True, help="3-letter IATA code, e.g. KEF")
    parser.add_argument("--weight", type=float, default=1.0,
                        help="heuristic weight for A* (1.0 = optimal, >1 = weighted A*)")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                        help="folder holding lab01-03_airports.csv and routes.csv")
    return parser


def check_airport(parser: argparse.ArgumentParser, code: str, coords: dict, role: str) -> str:
    """Return the upper-case code, or stop with a clear message if it is not in the graph."""
    code = code.strip().upper()
    if len(code) != 3 or not code.isalnum():
        parser.error(f"{role} airport {code!r} is not a 3-letter IATA code (example: KUL).")
    if code not in coords:
        suggestions = difflib.get_close_matches(code, coords.keys(), n=3, cutoff=0.6)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        parser.error(f"unknown {role} airport code {code!r}: it is not in the airport list.{hint}")
    return code


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.weight < 1.0:
        parser.error("--weight must be at least 1.0")

    coords, graph, _ = load_default_graph(args.data_dir)
    start = check_airport(parser, args.start, coords, "start")
    goal = check_airport(parser, args.goal, coords, "goal")

    result = run_algorithm(args.algorithm, graph, coords, start, goal, weight=args.weight)

    name = ALGORITHM_NAMES[args.algorithm]
    if args.algorithm == "astar" and args.weight != 1.0:
        name += f" (weight {args.weight})"
    print(f"Algorithm : {name}")
    if not result.found:
        print(f"No route found from {start} to {goal} ({result.expanded} airports expanded).")
        return 1
    print(f"Route     : {' -> '.join(result.path)}")
    print(f"Hops      : {result.hops}")
    print(f"Distance  : {result.path_cost_km:,.1f} km")
    print(f"Expanded  : {result.expanded} airports")
    print(f"Frontier  : {result.max_frontier} (largest size)")
    print(f"Runtime   : {result.runtime_ms:.2f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
