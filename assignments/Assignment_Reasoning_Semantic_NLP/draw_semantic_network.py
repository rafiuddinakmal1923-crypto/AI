
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

HERE = Path(__file__).resolve().parent

# Node positions (x, y) chosen by hand so no arrows cross labels.
NODES = {
    "Airline": (0.0, 2.0),
    "Flight": (3.0, 2.0),
    "Passenger": (3.0, 4.2),
    "Destination": (6.3, 3.4),
    "Departure Time": (6.3, 0.8),
    "Delayed": (1.8, 0.0),
    "Cancelled": (4.2, 0.0),
    "Passenger Complaint": (3.0, -2.0),
    "Customer Support": (6.6, -2.0),
}
EDGES = [  # (from, label, to, given_in_question)
    ("Airline", "operates", "Flight", True),
    ("Passenger", "books", "Flight", True),
    ("Flight", "has", "Destination", True),
    ("Flight", "has", "Departure Time", True),
    ("Flight", "can be", "Delayed", True),
    ("Flight", "can be", "Cancelled", True),
    ("Delayed", "may cause", "Passenger Complaint", True),
    ("Cancelled", "may cause", "Passenger Complaint", False),
    ("Passenger Complaint", "is handled by", "Customer Support", False),
]
CURVE = {}                         # all arrows are straight in this layout


def box_edge(centre, towards, name):
    """Point where the line from a box's centre towards another point leaves the box.

    The box size is estimated from the label length (in plot units for this figure).
    """
    (cx, cy), (tx, ty) = centre, towards
    dx, dy = tx - cx, ty - cy
    half_width = 0.044 * len(name) + 0.10
    half_height = 0.22
    t = min(half_width / abs(dx) if dx else float("inf"),
            half_height / abs(dy) if dy else float("inf"))
    return cx + dx * t, cy + dy * t


figure, axis = plt.subplots(figsize=(11, 7.5))
for source, label, target, given in EDGES:
    (x1, y1), (x2, y2) = NODES[source], NODES[target]
    bend = CURVE.get((source, target), 0.0)
    colour = "#1f4e79" if given else "#c55a11"
    # Start and end each arrow exactly on the edge of its box (boxes are wider for
    # longer names), so every arrowhead stays visible.
    start = box_edge((x1, y1), (x2, y2), source)
    end = box_edge((x2, y2), (x1, y1), target)
    axis.add_patch(FancyArrowPatch(start, end, connectionstyle=f"arc3,rad={bend}",
                                   arrowstyle="-|>", mutation_scale=18, color=colour, lw=1.6,
                                   shrinkA=4, shrinkB=4, linestyle="-" if given else "--"))
    # label at the middle of the (possibly curved) arrow
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    mx, my = mx + bend * (y2 - y1) / 2, my - bend * (x2 - x1) / 2
    axis.text(mx, my, label, fontsize=10, ha="center", va="center", color=colour,
              bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))

for name, (x, y) in NODES.items():
    axis.text(x, y, name, fontsize=11, fontweight="bold", ha="center", va="center",
              bbox=dict(boxstyle="round,pad=0.55", fc="#deebf7", ec="#1f4e79", lw=1.5))

axis.plot([], [], color="#1f4e79", label="relationship given in the question")
axis.plot([], [], color="#c55a11", linestyle="--", label="additional relationship (my own)")
axis.legend(loc="lower left", fontsize=9, frameon=False)
axis.set_title("Semantic network: airline knowledge", fontsize=14, fontweight="bold")
axis.set_xlim(-1.2, 8.0)
axis.set_ylim(-2.8, 4.9)
axis.axis("off")
figure.tight_layout()
figure.savefig(HERE / "semantic_network.png", dpi=160)
print("Saved", HERE / "semantic_network.png")
