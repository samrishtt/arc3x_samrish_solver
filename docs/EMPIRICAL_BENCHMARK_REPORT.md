# Empirical Benchmark Report: Self-Correcting World Models

> **Project**: Self-Correcting World Models Through Active Experimentation
> **Generated**: 2026-09-11 20:45:08
> **Seeds per experiment**: 25
> **Leakage status**: Agent receives ZERO target identity, rule knowledge, or mutation notification

---

## Executive Summary

### WHAT I EXPECTED
Disagreement-driven experiment selection (Strategy D) should discover hidden rules
faster than random exploration (Strategy A) and greedy exploration (Strategy B).

### WHAT ACTUALLY HAPPENED

## 1. Experiment 1: Strategy Comparison — Rule Discovery

| Strategy | Success Rate | Mean Steps (↓) | Median | Std | 95% CI | Confidence |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline_Random** | 36.0% | 40.8 | 50.0 | ±15.3 | ±6.0 | N/A |
| **A_Random** | 16.0% | 45.3 | 50.0 | ±11.2 | ±4.4 | 3.4% |
| **B_Greedy** | 0.0% | 50.0 | 50.0 | ±0.0 | ±0.0 | 3.4% |
| **C_Uncertainty** | 28.0% | 42.5 | 50.0 | ±12.0 | ±4.7 | 4.5% |
| **D_Disagreement** | 52.0% | 36.5 | 33.0 | ±14.8 | ±5.8 | 6.1% |
| **E_InfoGain** | 52.0% | 38.4 | 38.0 | ±13.8 | ±5.4 | 6.2% |

### Visual Comparison
```text
Baseline_Random     : [████████████████████░░░░░] 40.8
A_Random            : [███████████████████████░░] 45.3
B_Greedy            : [█████████████████████████] 50.0
C_Uncertainty       : [█████████████████████░░░░] 42.5
D_Disagreement      : [██████████████████░░░░░░░] 36.5
E_InfoGain          : [███████████████████░░░░░░] 38.4
```

**Sample efficiency gain (D vs Baseline)**: +10.4%

---

## 2. Experiment 4: Rule Mutation — Self-Correction Without Notification

The environment silently changed its hidden rule at step 20.
The agent received **zero notification** — no parameter updates, no hints.

| Metric | Value |
| :--- | :---: |
| Adaptation Success Rate | 64.0% |
| Mean Recovery Steps | 14.6 ± 11.9 |
| Median Recovery Steps | 9.0 |
| 95% CI | ±4.7 |
| Mean Prediction Errors | 29.3 |

---

## 3. Experiment 5: Structural Transfer

Colours permuted (red→purple, green→orange, blue→cyan) while causal structure preserved.

| Metric | Value |
| :--- | :---: |
| Transfer Success Rate | 52.0% |
| Mean Transfer Steps | 36.5 ± 14.8 |
| 95% CI | ±5.8 |

---

## 4. Experiment 6: Component Ablation

| Configuration | Success Rate | Mean Steps (↓) | Std | 95% CI | Impact vs Full |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Full System** | 52.0% | 36.5 | ±14.8 | ±5.8 | Baseline |
| **No Active Selection** | 16.0% | 45.3 | ±11.2 | ±4.4 | +24.1% |
| **No Uncertainty (Point Est)** | 4.0% | 48.0 | ±9.6 | ±3.8 | +31.5% |
| **No Mental Simulator** | 28.0% | 42.5 | ±12.0 | ±4.7 | +16.4% |
| **No Memory** | 52.0% | 36.5 | ±14.8 | ±5.8 | +0.0% |
| **No Self-Correction** | 4.0% | 48.0 | ±9.6 | ±3.8 | +31.5% |

### Ablation Visual
```text
Full System                   : [██████████████████░░░░░░░] 36.5
No Active Selection           : [███████████████████████░░] 45.3
No Uncertainty (Point Est)    : [████████████████████████░] 48.0
No Mental Simulator           : [█████████████████████░░░░] 42.5
No Memory                     : [██████████████████░░░░░░░] 36.5
No Self-Correction            : [████████████████████████░] 48.0
```

---

## 5. Honest Assessment

### WHETHER THE RESULT SUPPORTS OR WEAKENS THE HYPOTHESIS

The results are reported above without spin. Interpret the numbers directly.

### WHAT EXPERIMENT SHOULD COME NEXT

1. Scale hypothesis space (Experiment 2): Test with 2/4/8/16 entities
2. Misleading evidence (Experiment 3): Environments designed to produce early false positives
3. Cross-environment evaluation: Test on MiniGrid or ARC-AGI-3 tasks

---

*Report generated programmatically from raw per-seed data. All trajectory data preserved in `experiments/data/`.*