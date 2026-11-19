"""Lab 3 tests on the REAL airline graph, plus the command-line tool.

Run from the all_experiments folder:   python -m pytest utils -q
"""
import pandas as pd
import pytest

from utils.__main__ import main as cli_main
from utils.search_runner import ALGORITHM_NAMES, QUERIES_FILE, load_default_graph, run_algorithm

QUERIES = pd.read_csv(QUERIES_FILE)[["start", "goal"]].itertuples(index=False, name=None)
QUERIES = list(QUERIES)


@pytest.fixture(scope="module")
def airline():
    """Load the graph once for the whole file (it takes about a second)."""
    coords, graph, _ = load_default_graph()
    return coords, graph


@pytest.mark.parametrize("start, goal", QUERIES)
def test_astar_cost_equals_ucs_cost(airline, start, goal):
    coords, graph = airline
    a_star = run_algorithm("astar", graph, coords, start, goal)
    uniform = run_algorithm("ucs", graph, coords, start, goal)
    assert a_star.found and uniform.found
    assert a_star.path_cost_km == pytest.approx(uniform.path_cost_km, abs=1e-6)
    # ...and the heuristic should have saved work.
    assert a_star.expanded <= uniform.expanded


@pytest.mark.parametrize("start, goal", QUERIES)
def test_bfs_gives_fewest_hops(airline, start, goal):
    coords, graph = airline
    hops = {name: run_algorithm(name, graph, coords, start, goal).hops for name in ALGORITHM_NAMES}
    assert hops["bfs"] == min(hops.values())


def test_cli_finds_a_route(capsys):
    assert cli_main(["--algorithm", "astar", "--start", "kul", "--goal", "KEF"]) == 0
    output = capsys.readouterr().out
    assert "KUL -> HKT -> LED -> HEL -> KEF" in output


@pytest.mark.parametrize("bad_args, message", [
    (["--start", "XXX", "--goal", "KEF"], "unknown start airport code 'XXX'"),
    (["--start", "KUL", "--goal", "QQQ"], "unknown goal airport code 'QQQ'"),
    (["--start", "KUALA", "--goal", "KEF"], "not a 3-letter IATA code"),
    (["--algorithm", "magic", "--start", "KUL", "--goal", "KEF"], "invalid choice"),
])
def test_cli_rejects_bad_input(capsys, bad_args, message):
    with pytest.raises(SystemExit) as stop:
        cli_main(bad_args)
    assert stop.value.code == 2                     # argparse's "bad usage" exit code
    assert message in capsys.readouterr().err
