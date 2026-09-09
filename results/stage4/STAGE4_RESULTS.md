# Stage 4 — i.i.d. Label-Shift Reproduction and Validation

## 1. Objective

Stage 4 establishes a trustworthy non-graph baseline before introducing graph dependence.

The purpose of this stage is to verify that the complete label-shift calibration pipeline behaves correctly in a controlled i.i.d. setting:

1. generate source and target samples under pure label shift;
2. train a non-graph probabilistic classifier on labeled source data;
3. estimate target class priors without target labels using BBSE and RLLS;
4. convert estimated target priors into importance weights;
5. estimate target classwise calibration error using LaSCal;
6. compare label-free estimates against supervised researcher-side target calibration error;
7. characterize finite-sample behavior across label-shift severity and source sample size;
8. validate the implementation against reference implementations;
9. perform additional probabilistic sanity checks using ECE, NLL, Brier score, and controlled temperature scaling.

Stage 4 intentionally contains no graph structure.

Its role is to establish the i.i.d. reference point required for the graph-dependent experiments beginning in Stage 5.

---

## 2. Experimental Setup

### 2.1 Synthetic pure label shift

Binary source and target datasets are generated from the same class-conditional feature distributions.

The intended shift is therefore:

\[
P_s(X \mid Y) = P_t(X \mid Y)
\]

while allowing

\[
P_s(Y) \neq P_t(Y).
\]

The configured source class priors are

\[
P_s(Y=0)=0.5,
\qquad
P_s(Y=1)=0.5.
\]

The principal target-prior conditions are

\[
P_t(Y=0) \in \{0.50,0.60,0.70,0.80\}.
\]

The corresponding class-1 prior is

\[
P_t(Y=1)=1-P_t(Y=0).
\]

Source and target samples use separate deterministic random seeds.

Unless otherwise specified:

- source sample size: 2000;
- target sample size: 2000;
- feature dimension: 10;
- source split: 60% train, 20% validation, 20% test.

The source-validation partition supplies the labeled source information required by the label-shift estimators and LaSCal.

Target labels are hidden from all label-free estimation procedures and are used only for researcher-side evaluation.

### 2.2 Experimental separation

Three types of information are deliberately separated throughout Stage 4.

**Training information**

- labeled source training samples.

**Label-free estimation information**

- labeled source-validation samples;
- unlabeled target predictions.

**Researcher-side evaluation information**

- target labels;
- empirical target priors;
- supervised target calibration error;
- target ECE, NLL, and Brier score;
- empirical-prior oracle diagnostics.

This separation is important because target labels must never leak into BBSE, RLLS, or LaSCal estimation.

---

## 3. Baseline Classifier

A logistic-regression classifier is trained exclusively on the source training split.

This deliberately simple non-graph classifier provides a controlled i.i.d. baseline without introducing message passing or graph-induced statistical dependence.

The classifier outputs class probabilities

\[
\hat p(y=k \mid x)
\]

for each class.

These probabilities are subsequently used for label-shift estimation, calibration-error estimation, and secondary probabilistic diagnostics.

The simplicity of the classifier is intentional. Stage 4 is an implementation and finite-sample validation stage rather than an attempt to maximize predictive performance.

---

## 4. Label-Shift Estimation

### 4.1 BBSE

Black Box Shift Estimation (BBSE) estimates the target label distribution from the source-validation confusion matrix and the observed distribution of target predictions.

The confusion-matrix convention used by the implementation is

\[
C_{ij}=P_s(\hat Y=i\mid Y=j),
\]

so rows correspond to predicted classes and columns correspond to true classes.

The estimated target prior is denoted

\[
\hat q_t.
\]

Importance weights are then

\[
\hat w(k)
=
\frac{\hat P_t(Y=k)}
     {\hat P_s(Y=k)}.
\]

Invalid BBSE probability vectors are diagnosed explicitly rather than silently clipped.

This preserves failures as observable experimental outcomes rather than hiding them through post-processing.

### 4.2 RLLS-hard

Regularized Learning under Label Shift (RLLS) is implemented using the hard-prediction formulation.

The implementation follows the reference formulation used by the LaSCal ecosystem and was cross-checked against `RLLSImbalanceAdapter` from `abstention`.

The Stage 4 configuration uses

\[
\alpha=0.01,
\qquad
\delta=0.05.
\]

These values are fixed rather than selected using target labels.

The optimization operates on source prediction/label moments and the target prediction distribution. The implementation does not clip the resulting weights merely to force desirable behavior.

### 4.3 Empirical-prior oracle diagnostic

For diagnostic purposes, an empirical-prior oracle weight is also computed:

\[
w_{\mathrm{oracle}}(k)
=
\frac{\hat P_t(Y=k)}
     {\hat P_s(Y=k)},
\]

where the numerator uses empirical target labels.

This is a researcher-side diagnostic only.

It is not a deployable label-free method and is not used by BBSE, RLLS, or the primary LaSCal estimator.

Because these weights use empirical rather than population priors, they are referred to throughout the project as **empirical-prior oracle weights**.

This distinction matters: the empirical-prior oracle still contains finite-sample variation and should not be interpreted as an error-free population oracle.

---

## 5. LaSCal Calibration-Error Estimation

The primary Stage 4 quantity is the classwise label-free calibration-error estimate following the LaSCal formulation.

For each class \(k\), the estimator uses:

- labeled source-validation predictions;
- unlabeled target predictions;
- an estimated importance weight for class \(k\);
- adaptive target bins.

The primary configuration is

\[
p=2,
\qquad
B=15.
\]

The result is retained as a classwise vector rather than prematurely collapsing all classes into a single scalar.

For reporting estimation reliability, the classwise absolute error is

\[
E_k
=
\left|
\widehat{CE}_{t,k}
-
CE_{t,k}
\right|.
\]

A macro summary is additionally reported as

\[
E_{\mathrm{macro}}
=
\frac{1}{K}
\sum_{k=1}^{K}E_k.
\]

The supervised target quantity \(CE_{t,k}\) uses target labels only for researcher-side evaluation.

The classwise vector remains scientifically important because a macro average can hide asymmetric behavior between classes.

---

## 6. Reference-Implementation Validation

Implementation correctness was checked independently from experimental performance.

### 6.1 RLLS reference comparison

The local RLLS-hard implementation was compared against the reference `RLLSImbalanceAdapter` implementation on a controlled example.

Local weights:

\[
[1.66666667,\;0.33333333]
\]

Reference weights:

\[
[1.66666667,\;0.33333333].
\]

The outputs agreed numerically within the comparison tolerance.

This establishes reference equivalence for the controlled test case; it is not claimed as proof of universal implementation equivalence.

### 6.2 LaSCal controlled reference comparison

The local LaSCal implementation was compared directly against the official LaSCal calibration-error implementation on controlled two-class data.

Local result:

\[
[0.0975,\;0.0512037]
\]

Official result:

\[
[0.0975,\;0.0512037].
\]

Maximum absolute difference:

\[
3.576\times10^{-9}.
\]

The comparison passed the specified numerical tolerance.

Again, this demonstrates numerical equivalence for the controlled comparison rather than claiming universal equivalence for every possible input.

### 6.3 Supervised calibration-error reference comparison

The supervised classwise target calibration-error implementation was also cross-checked against the corresponding reference behavior in a controlled non-sparse setting.

Local result:

\[
[0.06541667,\;0.06541667]
\]

Reference result:

\[
[0.06541666,\;0.06541666].
\]

Maximum absolute difference:

\[
2.78\times10^{-9}.
\]

Sparse-bin behavior is handled deliberately by the local implementation by skipping bins for which a leave-one-out estimate is not defined.

Therefore, the controlled comparison is intentionally performed in a setting where the local and reference bin semantics are directly comparable.

### 6.4 Generated-data LaSCal comparison

A further reference check was performed using the actual synthetic 70/30 Stage 4 data-generation and classifier pipeline.

With identical probabilities, BBSE weights, binning configuration, and inputs, the local LaSCal output and official implementation were numerically equivalent within the specified tolerance.

This provides a stronger end-to-end implementation check than the toy comparison alone while still avoiding a claim of universal equivalence.

---

## 7. Canonical 70/30 Label-Shift Run

For the canonical run:

- source prior: 50/50;
- configured target prior: 70/30;
- source size: 2000;
- target size: 2000;
- seed: 42.

The source-validation confusion matrix was

\[
C=
\begin{bmatrix}
0.856 & 0.166\\
0.144 & 0.834
\end{bmatrix}.
\]

Its condition number was

\[
\kappa(C)=1.4508,
\]

indicating that the matrix was not severely ill-conditioned in this run.

The empirical source-validation priors were approximately

\[
[0.502,\;0.498].
\]

The empirical target priors were

\[
[0.690,\;0.310].
\]

BBSE estimated

\[
[0.674,\;0.326],
\]

giving target-prior L2 error

\[
0.0233.
\]

The importance weights were approximately:

| Method | Class 0 | Class 1 |
|---|---:|---:|
| BBSE | 1.340 | 0.656 |
| RLLS-hard | 1.340 | 0.656 |
| Empirical-prior oracle | 1.373 | 0.623 |

The BBSE and RLLS weight L2 errors relative to the empirical-prior oracle were both approximately

\[
0.0466.
\]

In this controlled setting, BBSE and RLLS-hard produced effectively identical weights.

This is reported as an empirical observation, not as a general equivalence between the methods.

---

## 8. Canonical Calibration-Error Results

The supervised target classwise calibration error was

\[
[0.012783,\;0.012924].
\]

The label-free estimates were:

| Method | Class 0 CE | Class 1 CE |
|---|---:|---:|
| Supervised target reference | 0.012783 | 0.012924 |
| LaSCal + BBSE | 0.067359 | 0.013074 |
| LaSCal + RLLS | 0.067359 | 0.013074 |
| LaSCal + empirical-prior oracle | 0.073517 | 0.015135 |

The corresponding absolute errors were:

| Method | Class 0 | Class 1 | Macro |
|---|---:|---:|---:|
| BBSE | 0.054576 | 0.000150 | 0.027363 |
| RLLS | 0.054576 | 0.000150 | 0.027363 |
| Empirical-prior oracle | 0.060735 | 0.002210 | 0.031472 |

An important observation is that oracle-weight estimation does not necessarily produce a smaller realized calibration-error estimation error for every finite sample.

The estimator contains downstream sampling and binning variability in addition to weight-estimation error.

Therefore,

\[
E_{\mathrm{BBSE}}-E_{\mathrm{oracle}}
\]

must not be interpreted as an additive decomposition of error sources.

Cancellation effects are possible.

The canonical result also illustrates why the classwise vector is retained: the estimation error is strongly asymmetric between the two classes in this realization.

---

## 9. No-Shift Sanity Control

A 50/50 target-prior condition was used as a no-population-shift control.

For the canonical control run, BBSE estimated target priors of approximately

\[
[0.474,\;0.526]
\]

against empirical target proportions of approximately

\[
[0.492,\;0.508].
\]

The BBSE prior L2 error was approximately

\[
0.0261.
\]

The BBSE macro classwise CE estimation error was approximately

\[
0.017386,
\]

while the empirical-prior-oracle value was approximately

\[
0.017970.
\]

The nonzero finite-sample error under the no-shift population condition is expected: independently generated source and target samples do not have identical empirical distributions.

The pipeline therefore does not collapse or behave pathologically when the population label shift is removed.

Importantly, this experiment is a sanity control rather than evidence that finite-sample estimation error should vanish under equal population priors.

---

## 10. Shift-Severity Experiment

The target class-0 prior was varied across

\[
0.50,\;0.60,\;0.70,\;0.80
\]

with 10 random seeds per condition.

This gives 40 shift-severity runs in total.

All BBSE prior estimates were valid probability vectors:

| Target prior \(P_t(Y=0)\) | Valid BBSE runs |
|---:|---:|
| 0.50 | 10/10 |
| 0.60 | 10/10 |
| 0.70 | 10/10 |
| 0.80 | 10/10 |

The macro absolute classwise CE-estimation errors were:

| Target prior | BBSE mean ± SD | RLLS mean ± SD | Oracle mean ± SD |
|---:|---:|---:|---:|
| 0.50 | 0.019624 ± 0.005077 | 0.019624 ± 0.005077 | 0.018791 ± 0.005091 |
| 0.60 | 0.020618 ± 0.006553 | 0.020618 ± 0.006553 | 0.019138 ± 0.005173 |
| 0.70 | 0.026238 ± 0.012641 | 0.026238 ± 0.012641 | 0.021480 ± 0.007875 |
| 0.80 | 0.039203 ± 0.019244 | 0.039203 ± 0.019244 | 0.029225 ± 0.011480 |

BBSE target-prior L2 error was:

| Target prior | Mean ± SD |
|---:|---:|
| 0.50 | 0.035116 ± 0.024975 |
| 0.60 | 0.032910 ± 0.023573 |
| 0.70 | 0.038941 ± 0.023984 |
| 0.80 | 0.042615 ± 0.026205 |

The observed results indicate increasing finite-sample calibration-error estimation difficulty under the strongest tested label shift.

In particular, the mean BBSE macro CE-estimation error increased from approximately

\[
0.0196
\]

under the 50/50 condition to approximately

\[
0.0392
\]

under the 80/20 condition.

This is an observed finite-sample trend over the tested conditions.

These descriptive mean ± standard-deviation results should not be interpreted as formal statistical significance tests.

---

## 11. Sample-Size Sensitivity

The target prior was fixed at 70/30 while source sample size was varied:

\[
n_s\in\{500,1000,2000,4000,8000\}.
\]

Target sample size remained 2000.

Each condition used 10 random seeds.

The macro classwise CE-estimation errors were:

| Source size | BBSE CE error | RLLS CE error | Oracle CE error |
|---:|---:|---:|---:|
| 500 | 0.094502 ± 0.045520 | 0.094502 ± 0.045520 | 0.085097 ± 0.037868 |
| 1000 | 0.052756 ± 0.021213 | 0.052756 ± 0.021213 | 0.052517 ± 0.018183 |
| 2000 | 0.026238 ± 0.012641 | 0.026238 ± 0.012641 | 0.021480 ± 0.007875 |
| 4000 | 0.013832 ± 0.008157 | 0.013832 ± 0.008157 | 0.013948 ± 0.006928 |
| 8000 | 0.012397 ± 0.009616 | 0.012397 ± 0.009616 | 0.009468 ± 0.006434 |

The mean BBSE calibration-error estimation error decreased from approximately

\[
0.0945
\]

at \(n_s=500\) to

\[
0.0124
\]

at \(n_s=8000\).

This corresponds to an approximately 87% reduction in the observed mean error across the tested range.

The target-prior and importance-weight diagnostics also generally improved as source sample size increased:

| Source size | BBSE prior L2 error | BBSE weight L2 error |
|---:|---:|---:|
| 500 | 0.059233 ± 0.040955 | 0.118862 ± 0.082067 |
| 1000 | 0.047083 ± 0.047886 | 0.094227 ± 0.095819 |
| 2000 | 0.038941 ± 0.023984 | 0.077911 ± 0.047982 |
| 4000 | 0.023786 ± 0.015606 | 0.047579 ± 0.031216 |
| 8000 | 0.021654 ± 0.027085 | 0.043313 ± 0.054176 |

The trend is strong at the level of the means, but strict monotonicity is not assumed for individual seeds or every diagnostic.

This experiment establishes finite-sample sample-size sensitivity in the i.i.d. baseline. It will become particularly important when graph dependence is introduced because nominal node count and effective statistical information need not behave identically.

---

## 12. Temperature-Scaling Sanity Check

A controlled temperature experiment was used to verify that the calibration pipeline responds sensibly to deliberate changes in probabilistic confidence.

For probabilities \(p\), temperature scaling is applied through the equivalent softmax transformation of log probabilities.

The same temperature transformation was applied to source-validation and target probabilities before computing the corresponding LaSCal quantities.

Positive temperature scaling preserves the hard class prediction, so the hard-prediction BBSE estimate remains unchanged.

For the controlled seed-1 experiment:

| Temperature | LaSCal class 0 | LaSCal class 1 | Supervised class 0 | Supervised class 1 | Macro absolute estimation error |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.047725 | 0.026493 | 0.025269 | 0.025356 | 0.011796 |
| 1.0 | 0.038782 | 0.024424 | 0.014900 | 0.014983 | 0.016662 |
| 2.0 | 0.051121 | 0.040568 | 0.023750 | 0.023839 | 0.022050 |

The supervised calibration-error quantity was smallest at \(T=1\) and increased under both sharpening and softening.

The LaSCal estimates remained finite and of the same general scale.

The LaSCal estimation error itself is not required to change monotonically with temperature because it remains a finite-sample estimator.

The values

\[
0.011796,\quad0.016662,\quad0.022050
\]

should therefore not be interpreted as establishing a monotonic law.

This temperature experiment is a controlled sanity diagnostic, not a population-level statistical result.

---

## 13. ECE, NLL, and Brier Diagnostics

ECE, negative log-likelihood, and Brier score are included as secondary descriptive diagnostics.

They are not interchangeable with the primary classwise \(p=2\) LaSCal calibration-error quantity.

For the canonical 70/30 seed-42 run:

| Metric | Source test | Target |
|---|---:|---:|
| ECE | 0.0201 | 0.0141 |
| NLL | 0.3576 | 0.3595 |
| Brier | 0.2259 | 0.2275 |

The multiclass Brier score is defined in this implementation as

\[
BS
=
\frac{1}{N}
\sum_{i=1}^{N}
\sum_{k=1}^{K}
\left(
p_{ik}-\mathbf{1}[y_i=k]
\right)^2.
\]

The score is **not divided by the number of classes**.

Consequently, under this convention, a deterministic completely incorrect prediction in a binary classification problem has Brier score 2 rather than 1.

Brier score is a proper scoring rule measuring overall probabilistic prediction quality rather than a pure calibration-error measure.

For target probabilities under temperature scaling:

| Temperature | ECE | NLL | Brier |
|---:|---:|---:|---:|
| 0.5 | 0.0744 | 0.4229 | 0.2417 |
| 1.0 | 0.0141 | 0.3595 | 0.2275 |
| 2.0 | 0.1118 | 0.4152 | 0.2567 |

Both NLL and Brier are lowest at \(T=1\) among these three settings, consistent with degradation in probabilistic prediction quality under the two imposed temperature perturbations.

ECE also increases substantially under both perturbations in this particular run.

These quantities are descriptive classifier diagnostics. They are not used as substitutes for the primary LaSCal-compatible classwise calibration-error estimator.

---

## 14. Figures

Stage 4 produces seven principal diagnostic figures.

### Figure 1 — CE estimation error versus label-shift severity

`figures/stage4/stage4_shift_severity_ce_error.png`

Shows BBSE, RLLS, and empirical-prior-oracle macro absolute classwise calibration-error estimation error across target-prior conditions.

This is the main Stage 4 visualization for shift-severity sensitivity.

### Figure 2 — CE estimation error versus source sample size

`figures/stage4/stage4_sample_size_ce_error.png`

Shows the finite-sample improvement in calibration-error estimation as source sample size increases.

### Figure 3 — BBSE prior error versus shift severity

`figures/stage4/stage4_bbse_prior_error_vs_shift.png`

Shows target-prior L2 estimation error across label-shift severity.

This separates prior-estimation behavior from downstream calibration-error estimation.

### Figure 4 — Importance-weight error versus source sample size

`figures/stage4/stage4_weight_error_vs_sample_size.png`

Shows BBSE/RLLS importance-weight L2 error relative to the empirical-prior oracle across source sample sizes.

### Figure 5 — Temperature scaling: macro classwise CE

`figures/stage4/stage4_temperature_macro_ce.png`

Compares the label-free LaSCal estimate with the supervised target calibration-error reference under controlled temperature scaling.

### Figure 6 — Temperature scaling: classwise CE

`figures/stage4/stage4_temperature_classwise_ce.png`

Retains the class-specific behavior that would be hidden by macro aggregation.

### Figure 7 — BBSE minus oracle estimation error

`figures/stage4/stage4_bbse_minus_oracle_vs_shift.png`

Shows the paired difference between BBSE and empirical-prior-oracle macro absolute CE-estimation error across label-shift conditions.

This figure is diagnostic only.

It must not be interpreted as an additive decomposition of total estimation error.

For multi-seed figures, displayed uncertainty bars represent mean ± standard deviation rather than confidence intervals.

Raw seed-level observations are also shown where available.

The figures are intended to expose both central trends and finite-seed variability rather than displaying only aggregated means.

---

## 15. Automated Tests

At Stage 4 closure, the full repository test suite reports:

```text
72 passed in 1.46s
```

The tests cover, among other components:

- synthetic label-shift generation;
- source/target data handling;
- BBSE;
- RLLS;
- LaSCal;
- supervised classwise calibration error;
- calibration metrics;
- negative log-likelihood;
- multiclass Brier score;
- relevant validation and failure conditions.

The Brier implementation includes explicit tests for:

- perfect predictions;
- a hand-computed binary example;
- a hand-computed multiclass example;
- ordering of good versus poorer predictions;
- the deterministic worst-case binary score under the chosen convention;
- mismatched sample counts;
- invalid probability vectors;
- negative probabilities;
- invalid class labels;
- non-integer labels;
- empty inputs.

Passing tests establish internal software consistency for the tested cases.

They do not by themselves establish the scientific validity of later graph experiments.

---

## 16. Main Stage 4 Findings

Stage 4 establishes the following empirical baseline.

### 16.1 Implementation validity

The local RLLS and LaSCal implementations agree numerically with their respective reference implementations on controlled validation cases.

The LaSCal implementation also agrees with the official implementation when exercised on the actual synthetic Stage 4 pipeline under the tested configuration.

Together with the automated test suite, these checks substantially reduce the risk that later graph results are artifacts of a basic implementation error.

### 16.2 Label-shift estimation

BBSE produced valid target-prior vectors in all 40 runs of the shift-severity experiment.

Under the tested synthetic conditions, BBSE and RLLS-hard produced effectively identical results.

This is specific to these experiments and is not interpreted as theoretical equivalence.

### 16.3 Finite-sample behavior

Calibration-error estimation becomes substantially more reliable as source sample size increases.

Across the tested source-size range, mean BBSE macro CE-estimation error decreased by approximately 87%.

This provides a clear finite-sample reference trend before graph-induced dependence is introduced.

### 16.4 Shift severity

The strongest tested label-shift condition, 80/20, exhibited the largest mean calibration-error estimation error among the tested conditions.

This is an observed pattern rather than a formal significance claim.

### 16.5 No-shift behavior

Finite-sample calibration-error estimation error remains nonzero even when population source and target class priors are equal.

This demonstrates why nonzero estimation error alone cannot be interpreted as evidence of distribution shift or estimator failure.

### 16.6 Oracle diagnostic

Empirical-prior oracle weights do not eliminate downstream finite-sample calibration-error estimation variability.

Consequently, prior/weight estimation and downstream calibration-error estimation must be studied separately.

### 16.7 Probabilistic diagnostics

ECE, NLL, Brier score, and controlled temperature scaling behave sensibly as secondary probabilistic diagnostics.

They support implementation sanity without replacing the primary classwise LaSCal-compatible calibration quantity.

---

## 17. What Stage 4 Does Not Establish

Stage 4 does **not** establish that label-free calibration fails on graph data.

It also does not establish that graph dependence will necessarily increase estimation error.

No graph structure has yet been introduced.

Therefore Stage 4 cannot support conclusions about:

- graph dependence;
- graph homophily;
- message passing;
- GNN representations;
- effective graph sample size;
- graph-induced violation of classifier-level label-shift assumptions.

Those questions belong to the subsequent stages.

This distinction is essential because the research experiment must determine whether graph dependence harms, preserves, or conditionally changes estimator reliability.

The result must not be predetermined.

---

## 18. Limitations

Stage 4 intentionally remains limited in scope.

First, all observations are independent samples. No graph topology, message passing, homophily, or node dependence is present.

Second, the synthetic classifier and distributions are deliberately simple. This is useful for validation but is not intended to reproduce the full complexity of real graph datasets.

Third, the shift-severity and sample-size experiments use 10 seeds per condition. Mean ± standard deviation describes observed variability but is not a formal significance analysis.

Fourth, the empirical-prior oracle is a diagnostic reference rather than a deployable estimator.

Fifth, the difference between estimated-weight and oracle-weight CE errors is not an additive causal decomposition.

Sixth, the temperature-LaSCal experiment is a controlled single-seed sanity check and should not be interpreted as a general statistical conclusion.

Seventh, BBSE and RLLS overlap numerically in the current synthetic experiments. This does not imply that the two estimators will remain equivalent under harder or graph-dependent conditions.

Finally, ECE, NLL, Brier score, and the primary LaSCal classwise \(p=2\) quantity measure different properties and should not be compared numerically as if they shared a common scale.

---

## 19. Connection to the Research Question

The broader project studies:

> **Under label shift, how does graph-induced dependence affect the finite-sample reliability of importance-weighted label-free calibration-error estimation in GNNs, and how is this effect modulated by graph homophily?**

Stage 4 provides the necessary control condition for that question.

If graph-dependent experiments later exhibit increased error, the Stage 4 results provide the corresponding i.i.d. finite-sample reference.

If graph-dependent experiments show similar behavior to Stage 4, that is also scientifically meaningful evidence.

If the effect depends on homophily, coupling strength, sample size, or shift severity, the project may instead identify a conditional boundary.

Thus the Stage 4 baseline allows later graph results to be interpreted relative to an experimentally validated non-graph reference rather than in isolation.

---

## 20. Stage 4 → Stage 5 Boundary

Stage 4 asks:

> Does the implementation provide a trustworthy i.i.d. label-shift calibration baseline before graph dependence is introduced?

The evidence collected in this stage supports proceeding to controlled graph experiments.

Stage 5 introduces graph structure while attempting to preserve experimental control.

The principal new variables will include:

- graph-induced dependence;
- graph homophily;
- coupling/message-passing strength;
- label-shift severity.

The central comparison will be against the Stage 4 i.i.d. baseline.

Stage 5 must distinguish at least four possible mechanisms:

1. target-prior estimation error;
2. importance-weight estimation error;
3. downstream calibration-error estimation error;
4. possible violation of the classifier-level label-shift relationship.

Useful diagnostics will therefore include quantities such as

\[
E_q
=
\|\hat q_t-q_t\|_2,
\]

\[
E_w
=
\|\hat w-w\|_2,
\]

and

\[
E_{CE}
=
\left|
\widehat{CE}_t-CE_t
\right|.
\]

The classifier-level shift diagnostic will additionally examine whether the relevant source and target prediction/label relationships remain sufficiently stable once graph message passing is introduced.

No conclusion about graph-induced failure or robustness is assumed in advance.

A null result, robustness result, failure result, or conditional boundary result are all scientifically admissible outcomes.

---

## 21. Reproducibility

Core Stage 4 experiments are implemented as executable scripts under `scripts/`.

Key artifacts include:

- `scripts/run_stage4_bbse_baseline.py`
- `scripts/run_stage4_multiseed.py`
- `scripts/run_stage4_sample_size.py`
- `scripts/run_stage4_temperature_lascal.py`
- `scripts/compare_rlls_reference.py`
- `scripts/compare_supervised_ece_reference.py`
- `scripts/compare_lascal_generated_data_reference.py`
- `scripts/plot_stage4_results.py`

Aggregate experiment results are stored under:

`results/stage4/`

Principal figures are stored under:

`figures/stage4/`

The repository intentionally does not need to retain every generated per-seed JSON artifact when the experiment scripts and aggregate summaries provide the required reproducibility path.

The full automated test suite can be executed from the repository root with:

```bash
PYTHONPATH=. pytest -q
```

At the final Stage 4 test run:

```text
72 passed in 1.46s
```

The principal reference checks can be reproduced with:

```bash
PYTHONPATH=. python scripts/compare_rlls_reference.py
PYTHONPATH=. python scripts/compare_supervised_ece_reference.py
PYTHONPATH=. python scripts/compare_lascal_generated_data_reference.py
```

The canonical 70/30 baseline can be reproduced with:

```bash
PYTHONPATH=. python scripts/run_stage4_bbse_baseline.py \
  --target-prior-0 0.7 \
  --seed 42 \
  --n-source 2000 \
  --n-target 2000
```

Stage 4 figures can be regenerated with:

```bash
PYTHONPATH=. python scripts/plot_stage4_results.py
```

---

## 22. Research Integrity and Interpretation Rules

The following rules govern interpretation of Stage 4 and subsequent experiments.

1. Target labels are never used by label-free estimators.

2. Target labels may be used only for researcher-side evaluation and explicitly identified oracle diagnostics.

3. BBSE failures must be recorded rather than silently corrected.

4. RLLS hyperparameters must not be tuned using target labels.

5. The empirical-prior oracle must not be presented as a deployable method.

6. Estimated-versus-oracle error differences must not be interpreted as an additive decomposition unless such a decomposition is mathematically justified.

7. Standard ECE, NLL, and Brier score must not be treated as numerically equivalent to the primary classwise \(p=2\) calibration quantity.

8. Mean ± standard deviation must not be described as a confidence interval.

9. A controlled numerical match to a reference implementation must not be described as proof of universal equivalence.

10. Stage 5 results must not be selected or interpreted according to whether they support the initial hypothesis.

The experiment determines the result.

---

## 23. Stage 4 Completion Criteria

The Stage 4 scientific pipeline contains:

- [x] deterministic pure-label-shift generator;
- [x] non-graph probabilistic classifier;
- [x] BBSE implementation;
- [x] RLLS-hard implementation;
- [x] empirical-prior oracle diagnostic;
- [x] LaSCal-compatible classwise CE estimator;
- [x] supervised target classwise CE reference;
- [x] controlled LaSCal reference comparison;
- [x] generated-data LaSCal reference comparison;
- [x] RLLS reference comparison;
- [x] supervised CE reference comparison;
- [x] no-shift sanity control;
- [x] four-condition shift-severity experiment;
- [x] 10 seeds per shift condition;
- [x] five-condition source sample-size experiment;
- [x] 10 seeds per sample-size condition;
- [x] BBSE validity diagnostics;
- [x] prior-estimation error diagnostics;
- [x] importance-weight error diagnostics;
- [x] empirical-prior-oracle comparison;
- [x] controlled temperature-LaSCal sanity experiment;
- [x] ECE diagnostic;
- [x] NLL diagnostic;
- [x] Brier-score implementation;
- [x] Brier unit tests;
- [x] Brier integration into canonical baseline diagnostics;
- [x] seven Stage 4 scientific figures;
- [x] complete automated test suite;
- [x] 72 passing tests;
- [x] Stage 4 scientific results report.

The only remaining actions after this report are repository-level closure checks:

- rerun final reference-validation scripts;
- run `git diff --check`;
- inspect the final Git status;
- commit the intended Stage 4 artifacts;
- push the final Stage 4 commit.

---

## 24. Final Stage 4 Status

**Scientific implementation: COMPLETE.**

**Experimental analysis: COMPLETE.**

**Automated tests: COMPLETE — 72 passed.**

**Figures: COMPLETE — 7 principal Stage 4 figures.**

**Scientific report: COMPLETE.**

**Repository closure: pending final audit, commit, and push.**

Stage 4 has established the controlled i.i.d. reference condition needed for the central graph-dependent investigation.

No additional Stage 4 experiment should be introduced merely to increase the number of experiments or figures.

Additional experiments should be added only if the final audit exposes a methodological gap.

Once the final reproducibility checks pass and the repository is committed, Stage 4 can be formally closed and Stage 5 can begin.