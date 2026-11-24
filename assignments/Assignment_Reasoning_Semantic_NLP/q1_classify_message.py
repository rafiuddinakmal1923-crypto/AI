"""Q1(c) - Rule-based classification of airline messages (REFERENCE DRAFT)."""


def classify_message(tweet):
    """Rule 1 / Rule 3: decide the category from keywords in the tweet."""
    tweet = tweet.lower()                      # so "Cancelled" and "cancelled" both match
    if "cancelled" in tweet or "delayed" in tweet:
        return "Service Complaint"
    elif "what time" in tweet or "departure" in tweet:
        return "Information Request"
    else:
        return "Other"


def should_prioritise(category):
    """Rule 2: every service complaint is prioritised for support."""
    return category == "Service Complaint"


tweet_a = "My flight has been cancelled"
tweet_b = "What time does flight AB123 depart?"

print(classify_message(tweet_a))
print(classify_message(tweet_b))

# Extra: apply Rule 2 to both tweets
for name, tweet in [("A", tweet_a), ("B", tweet_b)]:
    category = classify_message(tweet)
    print(f"Tweet {name}: {category:<20} prioritise = {should_prioritise(category)}")
