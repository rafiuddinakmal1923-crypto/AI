# Study guide: Lab 7 (tweet sentiment) and Lab 8 (genetic algorithm)

Both scripts are the lecturer's versions with small fixes, listed at the top of each file.
Every number below comes from `results/lab07/` and `results/lab08/`. They match the
reference values on the lecture slides, apart from runtime and one small difference noted in Lab 8.

```powershell
python all_experiments/lab07_airline_sentiment.py
python all_experiments/lab08_genetic_algorithm_regression.py
```

---

## Lab 7: Airline tweet sentiment with Naive Bayes

**Goal:** read a tweet to an airline and predict whether it is *negative*, *neutral* or
*positive*. Then judge the model honestly: not just "accuracy looks good".

### The pipeline in one picture
```
raw tweet ──clean──► "user flight cancelled again! url" ──count words──► [flight:1, cancelled:1, again:1, ...]
                                                                              │
                                    Multinomial Naive Bayes ◄─────────────────┘
                                              │
                               negative / neutral / positive (+ confidence)
```

### Step by step (the script's Q1–Q7)

| Step | What the code does | Why |
|---|---|---|
| **Q1 Clean + deduplicate** | `clean_text()`: links → `URL`, `@handles` → `USER`, lowercase, squeeze spaces. Then drop tweets whose *cleaned* text is a copy. | 213 tweets are exact copies and 355 are copies once links/handles are hidden (mostly retweets). If one copy lands in training and another in test, the model is tested on a tweet it has already seen. That's **data leakage**, and it makes the score look better than it really is. 14,640 → **14,285** tweets. |
| **Q2 Split 70/15/15** | `train_test_split` twice, `stratify=` sentiment, seed 42, saved to `split.csv`. | *Stratified* keeps the 63/21/16 % class mix in every part. A fixed seed means the same split every run. |
| **Q3 Bag of words** | `CountVectorizer(stop_words="english", min_df=2, max_features=20000)`, **fitted on training tweets only**. | It turns text into word counts. Unigrams give **4,671** features; unigrams + bigrams give **11,850**. Fitting on train only means the vocabulary never "peeks" at validation/test words. |
| **Q4 Majority baseline** | Always predict the most common training class (negative). | It sets the bar any real model must beat: **63.5 % accuracy for free**. |
| **Q5 Tune on validation** | Grid: n-grams {(1,1), (1,2)} × smoothing α {0.3, 0.5, 1, 2}. Pick the best **validation macro F1**. | Choose the settings without touching the test set. Best: **(1,2), α = 0.3, validation macro F1 0.697**. |
| **Q6 Test once** | Refit with the chosen settings, then predict the test set **once**. | This gives an honest estimate of performance on new tweets. |
| **Q7 Evidence** | Per-class report, per-airline macro F1, and the 4 most confident mistakes per class. | Find *where* the model fails, not just how often. |

### Key ideas

**Bag of words:** a tweet becomes a vector of word counts, and word *order* is lost.
Bigrams (pairs of adjacent words such as "not good" or "never again") bring back a little local order.
That is why (1,2) beat (1,1) on every α.

**Multinomial Naive Bayes:** for each class *c* it computes
`score(c) = log P(c) + Σ count(w) · log P(w | c)`, then picks the class with the highest score.
- `P(c)` is the **prior**: how common the class is in training.
- `P(w | c)` is the **likelihood**: how often word *w* appears in tweets of class *c*.
- "Naive" means it assumes the words are independent given the class. That's not true, but it works surprisingly well.

**Smoothing α (Laplace):** a word never seen in, say, positive training tweets would get
`P(w | positive) = 0`, and one zero wipes out the whole class score. Adding α to every count
prevents this. A small α stays close to the data; a large α flattens the probabilities.
Here α = 0.3 was best, and α = 2 was clearly worse (0.652).

**Why accuracy alone misleads:** with 63.5 % negative tweets, the "always negative" model
already gets 63.5 % accuracy but finds **zero** neutral or positive tweets.
- **Precision** = of the tweets I *called* class X, how many really are X?
- **Recall** = of the tweets that really *are* X, how many did I find?
- **F1** = harmonic mean of precision and recall.
- **Macro F1** = average F1 over the 3 classes, **each class counting equally**, so the small
  neutral/positive classes can't be ignored. The baseline's macro F1 is only 0.259.

### Results (`results/lab07/`)

| Model (test set, 2,143 tweets) | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---|---|---|
| Majority baseline | 0.635 | 0.212 | 0.333 | 0.259 |
| Multinomial NB | **0.770** | 0.704 | 0.693 | **0.698** |

Per class (`per_class_metrics.csv`): negative F1 **0.857**, positive **0.673**, neutral **0.564**.

Confusion matrix (rows = true, columns = predicted):

|  | negative | neutral | positive |
|---|---|---|---|
| **negative** | 1185 | 115 | 61 |
| **neutral** | 158 | 239 | 50 |
| **positive** | 63 | 46 | 226 |

- **Neutral is the hardest class.** Only 239 of 447 neutral tweets are found. Neutral tweets
  (questions, requests) use the same words as complaints ("flight", "hold", "delayed") but
  have no strong emotion words, and the negative class is three times bigger. So the
  prior pulls them towards "negative" (158 mistakes).
- **Per airline** (`per_airline_f1.csv`): macro F1 ranges from 0.659 (Delta) to 0.782 (Virgin America).
  But Virgin America has only 63 test tweets against United's 594, and every airline has a
  different class mix. Treat this as a diagnostic, **not a ranking of airlines**.

### Error analysis (`error_analysis.csv`, 12 tweets)
The 4 most confident mistakes per true class are tagged negation (3), question/request (2) or
ambiguous/other (7). Things to notice:
- **Confidence is not correctness.** All 12 mistakes have confidence ≥ 0.9999. Naive Bayes
  multiplies many word probabilities, so its scores are usually extreme.
- **Label noise:** the dataset has 150 retweets of a JetBlue joke, "Our fleet's on fleek".
  Annotators labelled them 107 neutral, 29 negative and 14 positive. All 150 are also filed
  under the wrong airline (Delta). Some "errors" are really the label's fault.
- **Bag of words can't read negation or sarcasm:** "I was **not** able to check in" after
  "great flight… excited to fly again" was labelled neutral, but the model sees "great",
  "excited" and says positive. Worse, scikit-learn's `stop_words="english"` list **removes
  "not", "no", "never" and "cannot"**, so the model never sees the negation at all. Even the
  bigram "not good" becomes just "good". (The slides warn about this on "Why word order and
  negation matter". Keeping negations in would be a sensible next experiment.)

### Changes I made to the lecturer's script
1. Paths now point to `all_datasets/Lab 7/Tweets.csv` and are resolved from the script's location.
2. The headings printed a literal `\n` instead of a new line (`"\\n"` in the source).
3. The `www.` part of the URL regex had a doubled backslash in a raw string, so it never
   matched. Only 1 tweet is affected, and all the results above are unchanged.
4. Output is forced to UTF-8. Without this, a tweet containing an emoji (😒) crashed the
   script on Windows when the output was redirected.

---

## Lab 8: Genetic algorithm for regression

**Goal:** fit `charges ≈ p0 + p1·age + p2·bmi + p3·children + p4·smoker` using a
**genetic algorithm (GA)**, then check it against ordinary least squares (OLS), the exact answer.
The point is *not* that the GA wins: it's to verify a stochastic optimiser and learn how to report it.

### GA vocabulary
| Term | In this lab |
|---|---|
| **Chromosome** | One candidate model: 5 numbers `[p0, p1, p2, p3, p4]` (intercept + 4 coefficients) |
| **Population** | 100 chromosomes (40 in Q6) |
| **Fitness** | Training **MSE** (mean squared error). Lower is better. The test set is never used here. |
| **Selection** | **Tournament of 3:** pick 3 at random, the lowest MSE becomes a parent |
| **Crossover** | **Blend:** `child = α·A + (1−α)·B` with α drawn from [−0.1, 1.1], so a child can land slightly outside its parents |
| **Mutation** | Each gene has a 15 % chance of Gaussian noise added (scale 0.25 × 2000 for the intercept, 0.25 × 1000 for the coefficients) |
| **Elitism** | The best 4 chromosomes are copied unchanged, so best-so-far fitness can never get worse |
| **Generation** | One round of evaluate → select → crossover → mutate → keep elites (160 rounds) |

### Step by step (Q1–Q6)
1. **Q1 Sanity check on `f(x) = (x−3)²`.** The answer is known (x = 3), so this tests the GA
   loop before using it on real data. Result: **x = 2.999989**, f(x) ≈ 0 ✔
2. **Q2 Prepare data.** 1,338 rows; smoker yes/no → 1/0; 80/20 split with seed 42
   (**1,070 / 268**). **StandardScaler fitted on training rows only.** Age (18–64) and smoker
   (0/1) are on very different scales, and standardising makes one mutation step size sensible for every gene.
3. **Q3 One GA run (seed 1).** Coefficients start around 0 with SD 5000 (charges are in the
   thousands of dollars), and the intercept starts near the mean charge. Result: train RMSE
   6,113.07, **test RMSE 5,829.38**.
4. **Q4 Ten seeds (1–10).** One random run proves little, so report the **median and spread**:
   median test RMSE **5,829.38**, SD **0.0019**. All ten seeds land on the same answer.
5. **Q5 Compare with OLS.** `LinearRegression` on the identical data gives test RMSE **5,829.38**,
   and the coefficients match the GA's median to one decimal place:

   | | age | bmi | children | smoker |
   |---|---|---|---|---|
   | OLS | 3616.3 | 1978.4 | 519.2 | 9559.3 |
   | GA (median of 10) | 3616.3 | 1978.4 | 519.2 | 9559.3 |

   The predictors are standardised, so each coefficient means "change in charges for +1 standard deviation".
   Being a smoker matters most (about +$9,560 per SD of the smoker column), then age.
   `convergence.png` shows every seed reaching the OLS line within about 30 of the 160 generations.
6. **Q6 Population 40 vs 100** (`summary.csv`):

   | Method | Median test RMSE | SD across seeds | Median runtime |
   |---|---|---|---|
   | GA pop = 100 | 5,829.379 | 0.0019 | 0.068 s |
   | GA pop = 40 | 5,829.379 | 0.0174 | 0.036 s |
   | OLS | 5,829.379 | 0 | < 0.001 s |

   A smaller population is about **twice as fast** but about **9× more variable** across seeds.
   That's the speed vs reliability trade-off.

### How to interpret it
- The GA **matches** OLS; it doesn't beat it. Linear least squares has one exact solution,
  and OLS computes it directly, much faster. Matching OLS is evidence the GA is implemented correctly.
- Use a GA when no exact or gradient method fits: a black-box or non-differentiable objective,
  discrete choices (routes, schedules, feature subsets), or many local optima.
  The GA assignment (delivery routes) is exactly that kind of problem.
- Always report several seeds (median + spread), never just the best seed.

### Differences from the slides
- Runtimes (0.068 s vs ≈ 0.04 s) depend on the machine.
- Population-40 SD is **0.017** here vs **≈ 0.026** on the slides. The seeds are fixed, but the
  numbers are tiny (hundredths of a dollar on an RMSE of 5,829), so different numpy/BLAS
  builds can round them differently. The slide itself says "small machine-dependent differences are expected".
  Every other value matches.

### Change I made to the lecturer's script
Only the data path: `all_datasets/Lab 8/insurance.csv`, resolved from the script's location.
