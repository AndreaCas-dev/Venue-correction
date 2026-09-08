# Environmental Correction & Predictive Modeling of 100m Sprint Performance

A case study developed in cooperation with the **Athletics Integrity Unit (AIU)**, aimed at improving the comparability of elite 100m sprint performances and using that corrected data to help identify athletes who may warrant inclusion in doping testing programs.

> Project by: **Andrea Cascone**, Vinzent Aschir, Hugo Thessieu, Ahmad Msheik — Université Côte d'Azur
> Supervised by: Alejandro Lozano, Miriam Stucker

The main repository of the project is hosted here: https://github.com/Aschir/Athlete_performance_monitoring.git

---

## 1. The Problem

Deciding which athletes should be included in a **Registered Testing Pool (RTP)** — the list of athletes subject to regular, unannounced doping controls — requires continuously comparing performance data across athletes and over time. This is harder than it sounds, for two reasons:

1. **Rankings change constantly.** Athletes compete at different venues, in different conditions, at different points in their careers.
2. **Raw times aren't directly comparable.** A 10.10s run into a headwind at sea level is a very different performance from a 10.10s run with a tailwind at altitude — but on paper, they look identical.

If you feed raw, uncorrected race times into a machine learning model, the model risks learning to recognize *favorable weather and venues* rather than *genuine athletic talent*. This project tackles that problem in two stages:

- **Stage 1 — Neutralize the environment.** Correct every recorded time for the effects of wind and venue, so performances become comparable regardless of where or under what conditions they happened.
- **Stage 2 — Predict elite performance.** Use the corrected, comparable data to train a time-series deep learning model that flags athletes likely to reach an "elite" performance level, as a decision-support signal for the AIU.

The dataset covers **elite 100m sprint results from 2015–2024**, including personal info (name, nationality), environmental info (wind, venue, date, indoor/outdoor), performance info (time, score, place), and competition info (round, importance).

---

## 2. My main Contribution — Venue Correction (Statistical Approach)

### Why venues matter

Beyond wind, the **venue itself** can systematically bias performances. Lower air density at high-altitude tracks reduces aerodynamic drag on a sprinter, and certain tracks (due to surface, altitude, or other factors) consistently produce times that are faster or slower than what an athlete's underlying ability would predict. Venues that consistently produce *faster-than-expected* times are called **Hot Venues**; venues that consistently produce *slower-than-expected* times are called **Cold Venues**.

The project explored two independent ways of detecting these venues: a statistical method and a machine learning method. Both were evaluated, and my statistical approach was ultimately the one **selected for the final downstream analysis** provided to the AIU, because it gives a conservative, interpretable correction that only flags venues with strong, consistent statistical evidence behind them.

### How the statistical method works, step by step

The core idea: **compare each individual race result to that same athlete's own average performance that season.** If, at a given venue, a large share of athletes consistently perform much better (or worse) than their own seasonal average, that's evidence the venue itself — not the athletes — is responsible for the difference.

1. **Compute each athlete's seasonal baseline.**
   For every athlete, in every year, I calculated their average wind-corrected performance across all their races that season. This baseline represents "how this athlete is running this year," independent of venue.

2. **Compute the "Venue Gap" for every race.**
   For each individual race, I calculated the difference between the athlete's seasonal average and the time they actually ran at that specific venue:

   `Venue Gap = Athlete's Seasonal Average − Time Run at This Venue`

   A large **positive** gap means the athlete ran much *faster* than usual at that venue (a hint the venue is "Hot"). A large **negative** gap means they ran much *slower* than usual (a hint the venue is "Cold").

3. **Find statistically unusual gaps using Tukey's fences.**
   Rather than picking an arbitrary cutoff, I used a standard outlier-detection technique (Tukey's fences, based on the interquartile range, IQR) to define what counts as an unusually large gap. In consultation with the AIU, I used a slightly stricter-than-default threshold (1.1×IQR instead of the classic 1.5×IQR) to catch more borderline cases while still avoiding false positives:

   - Upper fence ≈ 0.214 (unusually large *positive* gap → fast/"Hot" outlier)
   - Lower fence ≈ −0.205 (unusually large *negative* gap → slow/"Cold" outlier)

4. **Classify venues, not just individual races.**
   A single lucky or unlucky race isn't enough to blame the venue — it could just be a good or bad day. So instead, I looked at the venue as a whole: **a venue was only labeled Hot or Cold if more than 40% of the athletes who raced there** showed an outlier gap in the same direction. This threshold (agreed with the AIU) makes sure a venue label reflects a *systematic* pattern across many different athletes, not the form of one or two individuals. Venues that showed contradictory signals (meeting the criteria for both Hot and Cold) were treated as neutral and left uncorrected.

5. **Apply the correction.**
   Once a venue was confirmed as Hot or Cold, every recorded mark at that venue was adjusted by the venue's average gap size:

   - **Hot venue:** `Corrected Time = Original Time + |average venue gap|` (time is nudged slower, to cancel out the venue's advantage)
   - **Cold venue:** `Corrected Time = Original Time − |average venue gap|` (time is nudged faster, to cancel out the venue's disadvantage)

### Result of this step

Out of **4,069 venues** in the dataset, this method identified:
- **5 Hot Venues** (e.g. Yukon OK, Panama City, Lodi CA, Brest, Cannington)
- **34 Cold Venues** (e.g. Magglingen, Itajaí, Grimstad, Bonneville, Athens OH, Leicester, and others)

That's under 1% of all venues — intentionally so. The method is deliberately conservative: it only flags venues with strong, consistent statistical evidence, rather than trying to correct every small fluctuation. This was a conscious trade-off agreed with the AIU: false positives (wrongly flagging a normal venue) are more damaging to trust in the system than false negatives (missing a borderline one). These flagged venues were handed off for downstream manual review by the AIU, and this statistical correction was the version used to build the final corrected dataset for the predictive model described below.

*(The alternative, machine-learning-based venue correction — using an LGBM regression model and residual z-scores — is also in the repository for comparison; it identified a different, smaller, and less overlapping set of venues, which highlights how sensitive venue-bias detection is to methodology.)*

---

## 3. From Corrected Data to Predictions: The Time-Series Model

Once times were corrected for wind and venue, the project moved to the actual prediction task: **can we forecast which athletes are approaching an elite performance level, based on how their career has developed so far?**

### 3.1 Why a time-series model, and why by season

An athlete's performance isn't a single number — it's a trajectory that unfolds over years, with progression, plateaus, and sometimes setbacks. To capture that, the team didn't feed the model individual races. Instead, all of an athlete's races within a calendar year were summarized into **one "season" snapshot**, described by 17 features such as:

- their average and best score that season,
- how many competitions and finals they reached,
- how their performance changed compared to the previous year,
- their age and experience level,
- the quality/prestige of the venues they competed at.

This turns each athlete's career into a clean, regular sequence — one data point per season — which is exactly the kind of ordered, sequential data that recurrent neural networks are built to handle. The model looks at an athlete's **10 most recent seasons** and tries to predict their performance level going forward. Athletes with fewer than 5 seasons of history simply have the missing early seasons padded out and masked so the model knows to ignore them.

### 3.2 The model architecture: an LSTM with attention

The model used is a type of recurrent neural network called an **LSTM (Long Short-Term Memory)** network — a model designed specifically to learn patterns in sequences over time (originally famous for text and speech, but well-suited to any "story that unfolds over time," including an athlete's career). A regular LSTM reads a sequence in order and builds up a memory of what it has seen. This project used two refinements on top of the basic LSTM:

- **Bidirectional processing.** The model reads each athlete's 5-season sequence both forward (season 1 → 5) and backward (season 5 → 1), then combines both readings. This lets it use the full context of the sequence rather than only what came before each point.

- **Attention layer.** A plain LSTM tends to weigh recent information most heavily and can "forget" important things that happened earlier in a sequence. To fix this, an attention mechanism was added on top: it learns to automatically figure out which seasons in an athlete's history matter most for the final prediction (for example, a standout breakthrough season a few years back), and weighs those more heavily — regardless of where they sit in the timeline. Think of it as the model learning to highlight the most telling chapters of an athlete's career, instead of just reading the most recent page.

There's also a separate small branch of the network that processes **static, career-level facts** about the athlete (like their career-best score or total seasons of experience) that don't change season to season. The outputs of the season-by-season branch and the static branch are combined and passed through a final set of layers to produce one number: the athlete's predicted performance score for their next season.

To make the sequential input more efficient and less noisy, an **input projection layer** was added before the LSTM, compressing the 17 raw features per season into a more compact, denser representation before the network processes them.

### 3.3 What the model was trained to predict, and the loss function

Athletes who qualify for testing-pool consideration ("Top Athletes") are extremely rare in the dataset — about 0.05% of all athlete-seasons. Because of this massive imbalance, a straightforward "elite vs. not elite" classifier turned out to be impractical to train directly: with so few positive examples, a classifier can get very high accuracy just by *always* predicting "not elite."

So the team reframed the problem as a **regression task**: instead of directly predicting a yes/no label, the model predicts a continuous number — the athlete's expected `ResultScore` (a standardized performance score) for their most recent season, based on the 5 seasons before it. This continuous prediction is only turned into a "Top Athlete" yes/no decision afterward, by checking if the predicted score crosses a variable threshold across years (e.g., **1199 points**).

Two extra techniques were used to make sure the rare elite cases weren't ignored during training:
- **Oversampling** — seasons belonging to Top Athletes were shown to the model more often during training, so it wouldn't just learn to ignore them.
- **Extra penalty for elite misclassification** — the loss function specifically penalized the model more heavily when it got an elite athlete's prediction wrong, to force it to pay attention to that minority group.

The loss function itself was the **Huber loss** (rather than plain Mean Squared Error). Huber loss behaves like a mean squared error for small mistakes but like a more forgiving mean absolute error for very large mistakes. This makes training more stable and less thrown off by extreme outlier performances in the data.

$$L_{\delta}(e) = \begin{cases} \frac{1}{2} e^2 & \text{per } \vert{}e\vert{} \le \delta \\ \delta \cdot \left(\vert{}e\vert{} - \frac{1}{2} \delta\right) & \text{per } \vert{}e\vert{} > \delta \end{cases}$$

### 3.4 Results

**Regression performance** (predicting the continuous performance score):

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Baseline LSTM (no attention) | 58.25 | 76.09 | 0.34 |
| **Advanced LSTM (with attention + input projection)** | **23.38** | **33.50** | **0.87** |

In plain terms: the baseline model could only explain about a third of the variation in athlete performance, and it systematically failed for high-scoring athletes — for someone with a true score of 1200, it might predict 900 or 1050, essentially guessing wrong for exactly the athletes the AIU cares most about. Adding the attention mechanism roughly **halved the average prediction error** and pushed the model's explanatory power (R²) up to 0.87, with predictions for elite-level athletes finally tracking closely with their real scores.

**Binary classification performance** (converting predictions into an elite/not-elite decision at the 1199-point threshold, evaluated against a very imbalanced set of 2,725 non-elite vs. 7 elite athlete-seasons):

| Model | Accuracy | Macro F1 | Elite Precision | Elite Recall | Elite F1 |
|---|---|---|---|---|---|
| **athlete_lstm_balanced1.pt (best)** | 0.997 | **0.80** | 0.44 | **1.00** | 0.61 |
| athlete_lstm_balanced2.pt | 0.996 | 0.76 | 0.38 | 0.86 | 0.52 |

Because "not elite" is by far the dominant class, raw accuracy isn't a meaningful measure here — a model that predicted "not elite" for everyone would already score above 99.7% accuracy while being completely useless. That's why **macro F1-score** (which treats both classes as equally important, rather than being dominated by the majority class) was used as the main evaluation metric.

The best model correctly identified **all 7 elite athletes in the test set (100% recall)**, meaning it never missed an athlete who genuinely deserved flagging. Its precision was lower (44%), meaning it also flagged some non-elite athletes as potentially elite — a reasonable trade-off for this use case, since in a doping-monitoring context, it is far more costly to miss a genuinely elite athlete than to have a human reviewer double-check a handful of extra candidates flagged by the model.

---

## 4. Repository Structure & Reproducing the Results

The full code is organized into the following stages, matching the pipeline described above:

1. **Data preprocessing** — cleaning race records, standardizing round codes, filtering by activity level and age, removing duplicates.
2. **Wind correction** — physics-based neutralization of wind and altitude effects on race times (Mureika/Jonas model).
3. **Venue correction**
   - `Statistical approach` — Tukey's-fences-based Hot/Cold Venue detection (this repo's core contribution).
   - `Machine learning approach` — LGBM residual-based Hot/Cold Venue detection, for comparison.
4. **Season-level feature engineering** — aggregating race-level data into one feature vector per athlete per season.
5. **LSTM model** — the bidirectional, attention-based, dual-input (sequential + static) architecture used for the final predictions.
6. **Evaluation pipeline** — the tool provided to the AIU for practical, ongoing analysis of athlete performance predictions.

See the Appendix of the accompanying report for the full annotated code for each stage (wind correction, statistical venue correction, ML venue correction, season aggregation, sequence building, model definition, and training loop).

---

## 5. Limitations & Future Work

- The statistical venue correction is intentionally conservative (under 1% of venues flagged), and there is no ground-truth list of "true" Hot/Cold venues to validate against — so some real but weaker venue effects may be missed.
- The two venue-detection methods (statistical vs. machine learning) don't fully agree on which venues are biased, showing that venue-bias detection is sensitive to the chosen methodology.
- Missing birthdates required dropping a substantial number of rows, and imprecise venue elevation data limits how well the wind correction generalizes.
- The model currently has no way to distinguish a genuine performance decline from a one-off event like an injury, which can lead to misleading predictions in those cases.
- Future work includes combining the statistical and ML venue-correction methods, modeling at the individual-competition level instead of per season, and predicting performance for an athlete's *next competition* rather than their next season, for a more fine-grained monitoring signal.

---

## References

1. Mureika, J. et al. *A realistic quasi-physical model of the 100m dash.* Canadian Journal of Physics (2001).
2. Feriche, B. et al. *Resistance Training Using Different Hypoxic Training Strategies: a Basis for Hypertrophy and Muscle Power Development.* Sports Medicine - Open, 3 (2017).
3. Ke, G. et al. *LightGBM: A highly efficient gradient boosting decision tree.* NeurIPS 30 (2017).
4. Chen, Y. et al. *Hybrid Transformer-LSTM Model for Athlete Performance Prediction in Sports Training Management.* Informatica 49.24 (2025).

---

*AI-based tools were used for coding assistance, text improvements, and proofreading, in accordance with the disclosure in the original report.*
