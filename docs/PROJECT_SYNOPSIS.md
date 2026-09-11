# PROJECT SYNOPSIS
## Self-Correcting World Models Through Active Experimentation

**Researcher:** Samrish  
**Category:** Computer Science & Artificial Intelligence / Cognitive Systems  
**Affiliation / Fair:** IRIS National Science Fair 2025–2026  
**Repository & Open-Source Code:** [https://github.com/samrishtt/arc3x_samrish_solver](https://github.com/samrishtt/arc3x_samrish_solver)

---

### 1. Problem Statement & Research Objective

Current artificial intelligence systems exhibit a fundamental cognitive limitation:
- **Large Language Models (LLMs):** Excel at pattern matching across static internet corpora, but are entirely passive. They cannot interact with physical environments to verify, test, or falsify their beliefs.
- **Model-Free Reinforcement Learning (RL):** Uses undirected exploration ($\epsilon$-greedy or random noise), demanding millions of trials and frequently failing in sparse-reward environments.

**Human scientific reasoning works in reverse:** when children or scientists face an unfamiliar system, they form competing explanations, mentally simulate what each predicts, and execute targeted experiments where those theories make conflicting predictions. 

**Objective:** To design, build, and empirically evaluate an autonomous cognitive architecture that maintains competing explanations of an unfamiliar world, mentally simulates candidate actions, and selects experiments where explanations conflict most—evaluating whether this active epistemic loop accelerates causal discovery, enables self-correction when rules mutate, and allows continuous knowledge accumulation across versions.

---

### 2. Scientific Hypothesis

> **Hypothesis:** When an agent maintains a normalized belief distribution over competing explanations of an unfamiliar world, actions for which those explanations make maximally conflicting predictions yield the highest expected epistemic value. Selecting experiments that maximize predictive disagreement will significantly reduce interaction sample complexity compared to undirected or greedy exploration, while enabling autonomous self-correction without oracle guidance.

---

### 3. System Architecture & Methodology

The architecture operates in a closed cognitive loop without human hints or task-specific pretraining:

```
                            THE AUTONOMOUS AGI LOOP
                                       │
     Experience ──▶ Model ──▶ Predict ──▶ Experiment ──▶ Error ──▶ Self-Correct
                                       │
     ┌────────────────────────────────────────────────────────────────────────┐
     │ 1. All-Entity Perception: Extracts object coordinates & relations      │
     │ 2. Hypothesis Space: Normalized belief distribution over causal rules │
     │ 3. Mental Simulator: Forward rollouts of candidate actions            │
     │ 4. Experiment Selector: Multi-entity Gini predictive disagreement      │
     │ 5. Diagnostic Engine: Bayesian unlearning from prediction errors       │
     │ 6. 4-Tier Memory Bank: Prevents context overflow & freezes submissions │
     └────────────────────────────────────────────────────────────────────────┘
```

#### A. Multi-Entity Predictive Disagreement
For candidate action $a$, the agent computes forward rollouts across all active hypotheses $\{h_i\}_{i=1}^M$. The predictive disagreement $D(a)$ is computed as the total Gini impurity over predicted successor states across all entities $e$:
$$D(a) = \sum_{e \in \mathcal{E}} \left[ 1 - \sum_{s \in \mathcal{S}} P(s_e \mid a)^2 \right]$$
The action maximizing $D(a)$ is executed as the most informative physical experiment.

#### B. Four-Tier Hierarchical Memory Architecture
To prevent context overflow and transcript dumping, memory is structured into four distinct lifespans:
1. **`run_memory` (One run):** Temporary step transitions and prediction errors; purged immediately after distillation.
2. **`game_memory` (Across versions):** Distilled causal rules, interaction contingencies, and prior weights persisting across $v_1 \to v_N$.
3. **`global_memory` (Entire project):** Domain-wide invariants (collision physics, condition type success frequencies).
4. **`submission_memory` (Frozen at submission):** Read-only immutable snapshot ensuring deterministic evaluation without test-time drift.

---

### 4. Key Experimental Results

Evaluated across **25 procedural randomized seeds** (randomized entity placements and topological configurations) to eliminate layout bias:

| Exploration Strategy | Success Rate | Mean Steps (↓) | Median Steps | 95% Confidence Interval |
| :--- | :---: | :---: | :---: | :---: |
| **Strategy D: Predictive Disagreement (Ours)** | **52.0%** | **36.5** | **33.0** | **±5.8** |
| **Strategy E: Expected Information Gain** | **52.0%** | **38.4** | **38.0** | **±5.4** |
| **Baseline: Reactive Memory Agent** | 36.0% | 40.8 | 50.0 | ±6.0 |
| **Strategy C: Uncertainty-Only** | 28.0% | 42.5 | 50.0 | ±4.7 |
| **Strategy A: Random Exploration** | 16.0% | 45.3 | 50.0 | ±4.4 |
| **Strategy B: Reactive Greedy** | 0.0% | 50.0 | 50.0 | ±0.0 |

#### Major Empirical Findings:
1. **Superiority of Disagreement Selection:** Predictive disagreement achieved a **52.0% discovery rate**, more than tripling random exploration (16.0%).
2. **The "Greedy Trap":** Reactive greedy heuristics suffered a complete collapse (**0.0% success**), trapped in local object oscillation.
3. **Necessity of Hypothesis Uncertainty (Ablation):** Collapsing belief to a single point estimate reduced success to **4.0%**, proving that maintaining competing hypotheses is mathematically mandatory for discovery.
4. **Autonomous Mid-Run Self-Correction:** When physical rules silently mutated mid-run, the agent detected 29.3 prediction discrepancies on average and achieved a **64.0% adaptation recovery rate** without external reset.
5. **Cross-Run Knowledge Transfer:** The four-tier memory successfully warm-started Version 2 with verified causal rules from Version 1, while purging raw run trajectories to prevent context window bloat.

---

### 5. Conclusion & Significance for AGI

This project demonstrates that **epistemic agency**—the autonomous capacity to formulate competing models, simulate counterfactuals, and deliberately conduct experiments where models disagree—is a foundational prerequisite for general intelligence. By replacing blind trial-and-error with disagreement-driven experimentation and disciplined hierarchical memory distillation, AI systems can autonomously discover how novel worlds operate, adapt when assumptions fail, and continually accumulate knowledge.
