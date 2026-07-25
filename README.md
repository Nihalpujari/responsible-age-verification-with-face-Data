# Responsible Age Classification for Age-Restricted Sales

**Course:** Responsible AI and Data Ethics — 2026
**Institution:** SRH University Heidelberg
**Supervisor:** Prof. Simon Geschwill

**Live:** https://responsible-age-verification.streamlit.app/

**Dataset:** https://www.kaggle.com/datasets/moritzm00/utkface-cropped

---

## 1. What this project is

A facial age-band classifier, built end to end, and then **audited as if it were going into production** — because the point of the course is that building the model is the easy half.

**Deployment scenario:** automated age verification at the point of sale for alcohol, under the German *Jugendschutzgesetz* (JuSchG § 9):

| Product | Legal minimum age |
|---|---|
| Beer, wine, sparkling wine | 16 |
| Spirits, liquor | 18 |

The system does **not** output "you are 24." It outputs a probability that the person is at or above the legal threshold, and then makes one of three decisions — clear the sale, refuse it, or hand it to a human cashier. That third option is the most important design decision in the project and is explained in §5.

**Dataset:** UTKFace (~23,700 in-the-wild face images, each labelled with age, gender and race in the filename).

**Task framing:** 9-class ordinal classification, not regression:

```
0-2 · 3-9 · 10-19 · 20-29 · 30-39 · 40-49 · 50-59 · 60-69 · 70+
```

---

## 2. Repository map

```
repo/Code/
├── Responsible_Age_Classifier_v7_FINAL.ipynb   ← THE deliverable. Weeks 1-4, runs top to bottom.
├── responsible_ai_utils.py                     ← pure logic, importable, no notebook state
├── test_responsible_ai.py                      ← pytest suite over the above
├── Regulatory_Analysis.docx                    ← Week 1 written deliverable
├── cloudflared                                 ← binary used to publish the demo app
│
├── Responsible_Age_Classifier_week1_3.ipynb    ← development history: model versions v1-v7
├── Responsible_Age_Classifier_CLEAN.ipynb      ← intermediate restructure
├── Age_Classifier_v7_Demo.ipynb                ← deployment experiments (gradio, ipywidgets)
├── age_app.py                                  ← earlier standalone app version
├── cf.log / st.log                             ← runtime logs from a live demo session
```

The three history notebooks are kept deliberately. They are the evidence that the version comparison in Week 3 was an actual experiment and not a story told afterwards.

**Model checkpoints and split CSVs** live outside the repo at
`/home/jovyan/vault/face_project/data/` (nine `.keras` files, plus `train_df.csv`, `val_df.csv`, `test_df.csv`).
Images: `.../data/UTKFace/UTKFace/`.

---

## 3. Week 1 — Data, cleaning, and the regulatory frame

### 3.1 Ingestion and parsing

UTKFace encodes its labels in the filename: `age_gender_race_timestamp.jpg`. Every file is parsed into a dataframe (`parse_utk_filename`). Malformed and non-numeric stems return `None` rather than raising — there are genuinely broken filenames in the archive, and crashing on them would have been the wrong failure mode.

### 3.2 Cleaning rules (`clean_utk`)

| Rule | Reason |
|---|---|
| Drop `age < 1` | age 0 is a placeholder in this dataset, not a real infant label |
| Drop `age > 100` | the tail is sparse and label-noisy |
| Drop `gender == Unknown` / `race == Unknown` | unparseable codes; keeping them pollutes the fairness slices |
| Add `age_group` column | bins continuous age into the 9 canonical bands |

Every one of these rules is covered by a unit test (Week 4), so the cleaning contract is enforced rather than assumed.

### 3.3 Exploratory analysis — what the data actually looks like

This is where the fairness problem announces itself before a single model is trained.

- **Age is severely non-uniform.** The 20-29 band dominates; 50-59 and 60-69 are thin. Not a sampling accident — it reflects who ends up in scraped web images.
- **Race is imbalanced.** White faces are the largest group by a wide margin.
- **The imbalances interact.** The intersectional cells (race × age band) are the real unit of concern, and some are extremely thin. A model can look acceptable on every marginal slice while failing badly on a specific cell.

### 3.4 Splitting

Stratified train / validation / test split, with the split integrity itself tested: `splits_disjoint()` asserts that no filepath appears in more than one split. Leakage between splits is the most common way a fairness audit produces numbers that mean nothing, so it is checked mechanically rather than trusted.

### 3.5 Regulatory analysis (`Regulatory_Analysis.docx`)

Written analysis covering:

- **EU AI Act** — where a biometric age-estimation system sits in the risk taxonomy, and specifically **Article 14 (human oversight)**, which directly motivates the abstention design in §5.
- **GDPR** — facial images as biometric data, purpose limitation, and the fact that UTKFace subjects never consented to this use.
- **Jugendschutzgesetz § 9** — the legal thresholds the system claims to enforce.
- Documented dataset provenance and the consent problem it carries.

---

## 4. Week 2 — Training

Two models are trained in the final notebook. This is a deliberate reduction: the development notebooks contain seven versions, but the final deliverable presents only the **baseline** and the **final selected model**, so the comparison is legible.

### 4.1 `v1` — Baseline

A straightforward MobileNetV2 transfer-learning setup. No augmentation, no staged unfreezing, no learning-rate schedule. Its job is not to be good; its job is to be the reference point that makes every later improvement measurable.

### 4.2 `v7` — Final model

Selected after a seven-version search across architectures, resolutions, augmentation and training recipes — a CNN trained from scratch, EfficientNetB0, and several MobileNetV2 configurations. The full series and its numbers are in §4.3.

```
Input 224×224×3, raw [0,255]
  └─ Augmentation: flip / rotation / zoom / contrast
  └─ Rescaling(1/127.5, offset=-1.0)        ← baked into the model
  └─ MobileNetV2 (ImageNet weights)
  └─ GlobalAveragePooling → Dropout → Dense(9, softmax)

Stage 1:  backbone frozen,   Adam 1e-3,   5 epochs
Stage 2:  full unfreeze,     Adam 1e-4,  35 epochs
          ReduceLROnPlateau(monitor=val_accuracy, factor=0.4, patience=3)
          EarlyStopping(monitor=val_accuracy, patience=7, restore_best_weights=True)
```

**On the rescaling layer.** MobileNetV2's ImageNet weights expect inputs in `[-1, 1]`. Baking `Rescaling` into the model instead of doing it in the data pipeline means the saved `.keras` file is self-contained — the deployed app cannot get the preprocessing wrong, because there is no preprocessing left for it to get wrong. The app still auto-detects this by recursively searching the loaded model for a `Rescaling` layer, so it stays correct under either convention.

**On class weighting.** Computed from the training distribution and passed to `fit()`, so the thin upper age bands are not simply ignored by the optimiser in favour of the dominant 20-29 band.

**On the data pipeline.** Images are held as `uint8` (~4.2 GB for both resolutions, versus ~34 GB as `float64`) and the `tf.data` tensor slices are pinned to CPU. Building the dataset on the GPU caused out-of-memory failures at this dataset size.

### 4.3 The version series — how v7 was arrived at

The final notebook shows two models. **Seven were trained.** The full series is in
`Code/extra coding notbooks/Responsible_Age_Classifier_week1_3.ipynb`, kept in the repository on
purpose: the fairness conclusions in §6.3 are only meaningful because they come from **controlled
pairs**, where one thing changed at a time. Without the series, those claims would be storytelling.

Test set: 3,553 images, 9 classes.

| Version | Backbone | px | Aug | Recipe | Exact acc | One-off | Race gap | Ratio (min/max) |
|---|---|---|---|---|---|---|---|---|
| **v1** | MobileNetV2 | 96 | no | A | 47.73% | 86.24% | 14.36 pp | 0.752 |
| **v2** | MobileNetV2 | 224 | no | A | 51.62% | 88.60% | 19.97 pp | 0.700 |
| **v3** | EfficientNetB0 | 224 | yes | A | 47.42% | 87.67% | 10.75 pp | 0.804 |
| **v4** | MobileNetV2 | 224 | **yes** | A | 49.54% | 87.05% | 11.29 pp | 0.805 |
| **v5** | MobileNetV2 | 224 | yes | A (longer) | 49.90% | 88.40% | 13.50 pp | 0.774 |
| **v6** | **custom CNN from scratch** | 224 | yes | B | 53.25% | 88.85% | 12.84 pp | 0.794 |
| **v7** | MobileNetV2 | 224 | yes | **B** | **53.93%** | **89.84%** | 17.91 pp | 0.734 |

**Recipe A** — two-stage transfer with a fixed learning rate.
**Recipe B** — two-stage transfer plus `ReduceLROnPlateau` on `val_accuracy` and a longer stage-2 budget.

Per-race accuracy for v1–v5 is logged in full in the notebook. Ratios are reported as measured; see §6.2
on why no pass/fail line is drawn against them.

**The three comparisons that carry the argument**

- **v2 → v4 · the augmentation ablation.** Identical architecture, identical recipe, identical
  resolution. Augmentation is the *only* difference. Accuracy drops slightly (51.62% → 49.54%) while the
  race gap nearly halves (**19.97 → 11.29 pp**). This is the cleanest result in the project: a change
  made for generalisation moved fairness substantially, and cost accuracy to do it.

- **v6 → v7 · pretrained vs. from scratch, same recipe.** A custom CNN with no ImageNet weights reaches
  **53.25%**; MobileNetV2 with ImageNet weights reaches **53.93%**. **0.68 pp apart.** Both jumped well
  above the Recipe-A models. The gain came from the *training recipe*, not from the pretrained backbone
  — on ~20k images the ImageNet prior is worth well under a point. This pair also supplies the Grad-CAM
  result in §6.4.

- **v1 → v2 · resolution.** 96px to 224px buys ~4 pp of accuracy and **widens** the race gap by 5.6 pp.
  Higher resolution was not neutral with respect to fairness.

**What the rest of the series shows**

- **v3 (EfficientNetB0)** is the clearest negative result. A larger, more modern backbone produced the
  *lowest* exact accuracy in the series (47.42%) alongside the *narrowest* race gap (10.75 pp) — it
  underfit fairly rather than performing well. Worth keeping precisely because it breaks the assumption
  that the accuracy ranking and the fairness ranking move together. They do not: across the series the
  correlation runs the *wrong* way, with the two most accurate models (v6, v7) carrying wider gaps than
  v3 and v4.
- **v5** tests whether v4 simply needed more epochs. It does not: +0.36 pp accuracy for roughly double
  the training, and the race gap moves back up to 13.50 pp.

**A caveat, stated plainly.** Seven versions were evaluated on the test set and one was selected on the
strength of those comparisons. That is **selection on the test set**, and the reported v7 figures are
therefore mildly optimistic. Publishing the whole series rather than only the winner is the partial
remedy — a reader can see exactly how much searching produced the headline number.

---

## 5. The abstention design — the core responsible-AI contribution

### 5.1 The problem

The `10-19` band **straddles the legal threshold of 18**. A confident, entirely correct prediction of "10-19" tells you nothing about whether this person may buy spirits. A system that treats "10-19" as a decision would be wrong in a way that is invisible in the accuracy metric.

### 5.2 The solution

Don't use the top-1 label. Use the **full softmax** to compute the probability of clearing the threshold:

```python
def p_at_least(probs, band_frac):
    return float(sum(probs[3:]) + probs[2] * band_frac)
```

All mass in `20-29` and above counts fully. The `10-19` band contributes only `band_frac` — the share of that band at or above the legal age, **derived from the training data's actual age distribution**, not assumed uniform. Fallbacks if the CSVs are unavailable: `0.40` for the 16+ threshold, `0.20` for 18+.

### 5.3 Three-way policy

```python
def decide(p_over, p_approve=0.95, p_reject=0.05):
    if p_over >= p_approve: return "auto-clear"
    if p_over <= p_reject:  return "auto-reject"
    return "human-review"
```

The band in the middle is the point. The model is permitted to say *I don't know*, and that answer routes to a human cashier. This is **selective prediction**, and it is the concrete implementation of EU AI Act Article 14 human oversight — not a paragraph in a document, but a branch in the deployed code.

**Safety property, enforced by tests:** all probability mass concentrated in `10-19` can never produce `auto-clear`. A teenager can be refused or escalated, never automatically approved.

---

## 6. Week 3 — Evaluation, fairness audit, and explainability

### 6.1 Accuracy metrics

Two are reported, and the difference matters:

- **Exact-band accuracy** — the strict metric.
- **One-off accuracy** — prediction within one band of truth. For an *ordinal* target this is the more honest measure of usefulness: confusing 30-39 with 40-49 is a materially different error from confusing it with 3-9. Roughly 90% one-off against roughly 54% exact says the model has learned real ordinal structure, and that the exact-band number understates it.

### 6.2 Fairness metric

**Accuracy parity gap** = best group accuracy − worst group accuracy, in percentage points.

A note on thresholds: the US EEOC "four-fifths rule" (a disparate-impact *ratio* below 0.8) is frequently cited in fairness work. It is an employment-screening convention from a different jurisdiction and a different decision type, and **this project does not adopt it as a pass/fail criterion.** Ratios and gaps are reported as measured. Where the line should sit for a retail age-check system is a question for the deployer, not something to hardcode into an audit script.

Gaps are reported with **bootstrap confidence intervals**, so a gap driven by a thin slice is visibly uncertain rather than quoted as a hard number.

### 6.3 Findings

**Augmentation is a fairness intervention, not just a regularisation trick.**
A controlled ablation (identical architecture, augmentation the only difference) cut the race accuracy gap from **19.97 pp to 11.29 pp**. Nothing about that change was fairness-motivated in intent — which is exactly why it needed to be measured.

**The training recipe mattered more than the pretrained backbone.**
A CNN trained from scratch (v6, 53.25%) landed **0.68 pp** below ImageNet-pretrained MobileNetV2 (v7, 53.93%) under the same Recipe B — and both sat 3-4 pp above every Recipe-A model. On a dataset of this size the ImageNet prior is worth under a point; the two-stage unfreeze and the LR schedule are worth several.

**Accuracy and fairness did not move together.**
Across the series the most accurate models carry the *widest* race gaps (v7: 53.93% / 17.91 pp) while the least accurate carries the narrowest (v3: 47.42% / 10.75 pp). Selecting on accuracy alone would have selected against fairness — which is the argument for auditing both rather than optimising one.

**The best-represented group is not the best-served group.**
White faces in the 20-29 band — the single largest cell in the dataset (n ≈ 307 in test) — score around **40%**, while every other race in that same band scores **59-60%**. More data did not mean better performance. This is the finding that most directly contradicts the intuitive "just collect more data" fix.

**The confusion matrix is U-shaped.** Accuracy is highest at the extremes (young children, elderly) and collapses in the middle. In the 50-59 band the *modal* prediction is wrong: 0.32 of that band is predicted as 60-69, exceeding the 0.27 predicted correctly. Middle-aged bands are flagged as low-reliability in the deployed app.

**Fairness through unawareness fails.** Race is never an input to the model. The disparities in the output are large and systematic regardless. Removing a protected attribute from the input does not remove it from the model's behaviour — it only removes your ability to measure what the model is doing with it.

### 6.4 Grad-CAM — the headline explainability result

Grad-CAM takes the gradient of the predicted class with respect to the final convolutional feature map, producing a heatmap of which facial regions drove the decision. Applied to **baseline v1 versus final v7**, and to controlled version pairs, with per-race attention profiles aggregated over thirds of the image.

**The result, replicated across two independent model pairs:**

> The model with the **more uniform** cross-race attention distribution has the **wider** accuracy gap.

**v7** (MobileNetV2): attention spread **6.67**, race gap **17.91 pp**.
**v6** (scratch CNN): attention spread **20.43**, race gap **12.84 pp**.

v7 looks *far* more even-handed under Grad-CAM — it attends to nearly the same facial regions regardless of race — and it is the model with the wider outcome disparity. The v2 → v4 pair moves in the same direction.

**Why this matters.** There is a comfortable assumption in applied XAI that if a model attends to the same regions across demographic groups, it is treating them equally. This project's data says that inference does not hold. **Spatial attention uniformity is not a sufficient fairness diagnostic.** An explainability visualisation can look reassuring while the outcome disparity gets worse. XAI and fairness auditing are complementary — one cannot substitute for the other.

*Implementation note:* the Grad-CAM function handles both flat models and models with a nested backbone sub-model, and unwraps the single-element output list Keras 3 returns for loaded models. Both cases occur across the version series.

---

## 7. Week 4 — Testing and deployment

### 7.1 Test suite

Logic that matters was extracted out of the notebook into `responsible_ai_utils.py`, so it can be imported and tested rather than living in cell state that dies on kernel restart.

```bash
pytest test_responsible_ai.py -v --cov=responsible_ai_utils --cov-report=term
```

Coverage:

| Group | What it pins down |
|---|---|
| `age_to_group` | every one of the 18 band boundaries, plus the invariant that all outputs are canonical labels |
| `parse_utk_filename` | valid parses, too-few parts, non-numeric, invalid gender/race codes → `Unknown` |
| `clean_utk` | each cleaning rule, and the exact surviving row set on a toy frame |
| `splits_disjoint` | clean splits pass; a deliberately leaked row is detected |
| `one_off_correct`, `accuracy_parity_gap`, `fairness_check` | metric arithmetic |
| `p_at_least` | all-adult → 1.0, all-child → 0.0, teen band counts partially, output stays in [0,1], monotone in threshold |
| **Safety block** | **a young child is never auto-cleared; the teen band alone never auto-clears; a child/teen mixture never auto-clears** |

That last block is the one worth pointing at. It is not testing that the code runs — it is testing that the *policy* cannot be violated, whatever the model outputs.

### 7.2 Streamlit application

`age_app.py`, served from the notebook.

- Product selector (16+ / 18+) — the threshold, and therefore the decision, changes with the product
- Upload or live camera capture
- Haar cascade face detection and crop before inference — mitigates the domain shift between UTKFace's tight crops and casual webcam framing
- Verdict card: **auto-clear / auto-reject / human review**, with P(age ≥ threshold) shown
- Full 9-band probability distribution, so the user sees the uncertainty rather than a single confident-looking label
- Low-reliability warning surfaced when the top band is 30-59
- Automatic detection of whether the loaded model expects `[0,255]` or `[0,1]`

### 7.3 Public demo link

The JupyterHub environment has no `jupyter-server-proxy`, so no port can be exposed conventionally. `cloudflared` (a standalone binary, zero Python dependencies) opens a quick tunnel to the Streamlit port and returns a public `trycloudflare.com` URL. The launcher cell starts both processes, tails the logs, and prints the link.

---

## 8. Known limitations

Stated plainly, because a limitations section that reads like marketing is worse than none.

1. **Accuracy is around 54% exact-band.** In line with published UTKFace 9-class results, but not a number that would justify unsupervised deployment. The abstention design exists precisely because the model is not good enough to be trusted alone.
2. **Middle-age bands (30-59) are unreliable** — per-class recall under 40%, and 50-59's modal prediction is a wrong band.
3. **Substantial race accuracy gaps remain** in the final model. They were reduced, not eliminated.
4. **UTKFace's race labels are coarse and externally assigned.** Five categories including "Other" is not a defensible taxonomy of human populations; it is the granularity the dataset happens to provide, and every fairness number here inherits that limitation.
5. **Consent.** UTKFace subjects did not consent to training an age-verification system. A real ethical problem with the dataset, not a footnote.
6. **Domain shift.** Training images are tight, frontal, reasonably lit crops. Retail-counter conditions are not. Face detection and cropping mitigate this; they do not solve it.
7. **`band_frac` is derived from UTKFace's age distribution**, which is not the age distribution of people buying alcohol in Germany. In a real deployment this parameter should be re-estimated from the deployment population.
8. **No adversarial or presentation-attack testing.** A printed photograph has not been evaluated against this system.

---

## 9. How to run

```bash
# 1. Tests
cd repo/Code
python -m pytest test_responsible_ai.py -v --cov=responsible_ai_utils --cov-report=term

# 2. Full pipeline
#    Open Responsible_Age_Classifier_v7_FINAL.ipynb, run top to bottom.
#    Week 2 retrains both models — allow roughly 45-60 minutes on GPU.

# 3. Live demo
#    Run the Week 4 launcher cell; it prints a public trycloudflare.com URL.
```

The Week 4 app requires the Week 2 training cells to have run, since it loads the checkpoint they produce.

---

## 10. What this project argues

Three claims, each supported by a measurement in the notebook rather than by assertion:

1. **A fairness-relevant intervention need not be fairness-motivated.** Augmentation was added for generalisation and cut the race gap by nearly half. The converse is also true, and more dangerous: routine engineering decisions can widen disparities without anyone intending it or noticing.

2. **Explainability is not a fairness guarantee.** Across two model pairs, more uniform Grad-CAM attention accompanied *wider* outcome gaps. A visualisation that looks fair is not evidence that the model is fair.

3. **The honest response to an uncertain model is to let it abstain.** Rather than tuning a 54%-accurate classifier until its confident-looking output could be shipped, the system routes its own uncertainty to a human. Human oversight is a code path, not a policy paragraph.
