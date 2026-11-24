"""Q2(b) - Semantic network stored as (subject, relation, object) triples (REFERENCE DRAFT)."""

semantic_network = [
    ("Airline", "operates", "Flight"),
    ("Passenger", "books", "Flight"),
    ("Flight", "has", "Destination"),
    ("Flight", "has", "Departure Time"),
    ("Flight", "can be", "Delayed"),
    ("Flight", "can be", "Cancelled"),
    ("Delayed", "may cause", "Passenger Complaint"),
    # My own additional relationships
    ("Cancelled", "may cause", "Passenger Complaint"),
    ("Passenger Complaint", "is handled by", "Customer Support"),
]

for relationship in semantic_network:
    print(relationship)

# A small query: follow the links out of "Flight"
print("\nEverything linked from Flight:")
for subject, relation, obj in semantic_network:
    if subject == "Flight":
        print(f"  Flight --{relation}--> {obj}")
