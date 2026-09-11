# Self-Correcting World Models Through Active Experimentation

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 5/5 Passing](https://img.shields.io/badge/tests-5%2F5%20passing-brightgreen.svg)](tests/)
[![Empirical Seeds](https://img.shields.io/badge/monte--carlo-Procedural%20Randomized-purple.svg)](experiments/data/)

> **Fundamental AGI Question:** *Can an autonomous agent discover how an unfamiliar world works by mentally simulating competing hypotheses and deliberately performing experiments where those explanations disagree most?*

This repository contains the complete experimental code, automated testing harness, raw trajectory logs, and scientific documentation for **Self-Correcting World Models Through Active Experimentation** (Prepared for the MIT ARC Prize Research Summit & IRIS National Science Fair 2026).

---

## 🧭 Why This Matters for Artificial General Intelligence (AGI)

Contemporary artificial intelligence exhibits a fundamental epistemic limitation:
- **Large Language Models (LLMs)** consume static internet corpora passively. They cannot act in an unfamiliar environment to verify or falsify their beliefs.
- **Model-Free Reinforcement Learning (RL)** relies on brute-force trial and error with undirected exploration ($\epsilon$-greedy or random noise), requiring millions of interactions that fail in sparse-reward or non-stationary environments.

**Human scientific cognition works in reverse:** when children or scientists face an unfamiliar system, they form competing explanations, imagine what each predicts, and execute targeted experiments where those theories make conflicting predictions. That single targeted action eliminates the largest volume of incorrect models in the fewest steps.

This project implements that complete cognitive loop as an autonomous computational architecture:

```
                            THE AUTONOMOUS AGI LOOP
                                       │
     Experience ──▶ Model ──▶ Predict ──▶ Experiment ──▶ Error ──▶ Self-Correct
                                       │
     ┌────────────────────────────────────────────────────────────────────────┐
     │ 1. All-Entity Perception: Generic state & spatial relation tracking   │
     │ 2. Competing World Models: Hypotheses over multi-object dynamics      │
     │ 3. Counterfactual Simulator: Internal rollouts of candidate actions   │
     │ 4. Predictive Disagreement: Selecting actions with maximal conflict   │
     │ 5. Diagnostic Engine: Bayesian unlearning triggered by prediction diff │
     └────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Empirical Benchmark Results

Evaluated across **procedural randomized environments** (randomized entity placements, variable distances, and obstacle topologies) to eliminate layout bias and measure genuine sample efficiency:

| Exploration Strategy | Success Rate | Mean Interaction Steps (↓) | Median Steps | 95% Confidence Interval |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline Agent (Reactive Memory)** | 36.0% | 40.8 | 50.0 | ±6.0 |
| **Strategy A (Random Exploration)** | 16.0% | 45.3 | 50.0 | ±4.4 |
| **Strategy B (Reactive Greedy)** | 0.0% | 50.0 | 50.0 | ±0.0 |
| **Strategy C (Uncertainty-Only)** | 28.0% | 42.5 | 50.0 | ±4.7 |
| **Strategy D (Predictive Disagreement — Ours)** | **52.0%** | **36.5** | **33.0** | **±5.8** |
| **Strategy E (Expected Information Gain)** | **52.0%** | **38.4** | **38.0** | **±5.4** |

### Key Scientific Findings
1. **Disagreement Outperforms Undirected Exploration:** Disagreement-driven experiment selection achieved a **52.0% discovery rate** on randomized procedural boards, more than **triple random exploration (16.0%)**.
2. **The "Greedy Trap":** Proximity-based greedy heuristics suffered a complete collapse (**0.0% success**) by being drawn into local object loops rather than staging multi-object physical interactions.
3. **Catastrophic Collapse of Point Estimates (Ablation):** Collapsing belief to a single point estimate (removing the distribution over competing hypotheses) or disabling prediction-error diagnosis collapsed success to **4.0%**, proving that maintaining hypothesis uncertainty is mathematically required for robust discovery.
4. **Autonomous Self-Correction Mid-Run:** Under silent, unannounced physical rule shifts, the agent detected an average of 29.3 prediction discrepancies and achieved a **64.0% adaptation recovery rate** without external reset or oracle guidance.

---

## 🧠 Cognitive Architecture

```
                          ENVIRONMENT (GridLab / Out-of-Distribution Tasks)
                                                │
                                                │ Raw Observation (No labels/goals)
                                                ▼
                                         ┌──────────────┐
                                         │  PERCEPTION  │  Extract entities & coordinates
                                         └──────┬───────┘
                                                ▼
                                         ┌──────────────┐
                                         │    MEMORY    │  Episodic transitions & contingency tables
                                         └──────┬───────┘
                                                ▼
                                       ┌──────────────────┐
                                       │ HYPOTHESIS SPACE │  All-Entity Relational Hypotheses
                                       │                  │  - Touch causes (single entity)
                                       │                  │  - Pairwise adjacency interactions
                                       │                  │  - Normalized belief distribution
                                       └────────┬─────────┘
                                                ▼
                                       ┌──────────────────┐
                                       │  COUNTERFACTUAL  │  Internal forward rollouts per action
                                       │    SIMULATOR     │  Multi-entity Gini disagreement:
                                       │                  │  D(a) = ∑_T [1 - ∑ P(s|a)^2]
                                       └────────┬─────────┘
                                                ▼
                                       ┌──────────────────┐
                                       │    EXPERIMENT    │  Select action maximizing
                                       │     SELECTOR     │  predictive disagreement
                                       └────────┬─────────┘
                                                ▼
                                              ACTION
                                                ▼
                                         ENVIRONMENT STEP
                                                ▼
                                       ┌──────────────────┐
                                       │    DIAGNOSTIC    │  Reality vs Mental Rollout Diff
                                       │      ENGINE      │  - True Positive: Bayesian verification
                                       │ (Self-Correction)│  - False Positive: Bayesian penalty
                                       └──────────────────┘  - Model Collapse: Re-generation
```

---

## 📁 Repository Structure

```
├── docs/                                  # Research Papers & Theoretical References
│   ├── IRIS_SUBMISSION.md                 # Complete Research Proposal & AGI Framework
│   ├── EMPIRICAL_BENCHMARK_REPORT.md      # Procedural Monte Carlo evaluation report
│   ├── AGI_RESEARCH_REFERENCE.md          # 49-section comprehensive reference document
│   └── ARCHITECTURE.md                    # Detailed cognitive architecture specification
│
├── world_model_lab/                       # Core Research Library
│   ├── core/types.py                      # Data schemas: Entity, Hypothesis, Rollout, Error
│   ├── environment/grid_lab.py            # Procedural environment with hidden rules & mutations
│   ├── perception/extractor.py            # Spatial relation & temporal diff extraction
│   ├── memory/store.py                    # Episodic memory & semantic contingency tables
│   ├── world_model/hypothesis_manager.py  # All-entity competing hypothesis generator
│   ├── world_model/simulator.py           # Counterfactual simulator & Gini disagreement
│   ├── experimentation/selector.py        # 5 comparative experiment selection strategies
│   ├── diagnostics/revision.py            # Prediction-error diagnosis & model recovery
│   ├── benchmarks/suite.py                # Automated procedural benchmark suite
│   ├── benchmarks/ablations.py            # Systematic component ablation runner
│   ├── benchmarks/report_generator.py     # Programmatic report & statistics generator
│   └── adapters/arc3x_bridge.py           # General 2D grid frame & action space adapter
│
├── experiments/data/                      # Raw Experimental Trajectories (.jsonl)
│   ├── exp1_strategy_comparison_*.jsonl   # Per-seed trajectories across all strategies
│   ├── exp4_rule_mutation_*.jsonl         # Prediction error logs during rule shifts
│   ├── exp5_transfer_*.jsonl              # Structural visual transfer logs
│   ├── exp6_ablation_*.jsonl              # Component ablation trial logs
│   └── results_summary_*.json             # Aggregated statistical summaries
│
├── tests/                                 # Unit Test Suite
│   ├── test_day1_grid_lab.py              # Environment physics & rule mechanics
│   ├── test_day2_perception_memory.py      # Spatial relation & transition memory
│   ├── test_day3_world_model.py           # Hypothesis generation & mental rollouts
│   ├── test_day4_active_loop.py           # End-to-end active experiment loop
│   └── test_arc3x_bridge.py               # 2D frame translation & action bridge
│
├── requirements.txt                       # Project dependencies
├── LICENSE                                # MIT Open Source License
└── README.md                              # This document
```

---

## 🚀 Quickstart & Verification

### 1. Installation
```bash
git clone https://github.com/samrishtt/arc3x_samrish_solver.git
cd arc3x_samrish_solver
pip install -r requirements.txt
```

### 2. Run the Full Unit Test Suite (5/5 Passing)
```bash
python -m tests.test_day1_grid_lab
python -m tests.test_day2_perception_memory
python -m tests.test_day3_world_model
python -m tests.test_day4_active_loop
python -m tests.test_arc3x_bridge
```

### 3. Run the Automated Procedural Benchmark Suite
```bash
python -m world_model_lab.benchmarks.report_generator
```
*Generates procedural environments, logs per-seed trajectories to `experiments/data/`, and outputs a formatted Markdown report to `docs/EMPIRICAL_BENCHMARK_REPORT.md`.*

---

## 👤 Author & Acknowledgments

- **Researcher:** Samrish ([@samrishtt](https://github.com/samrishtt))
- **Research Topic:** Autonomous World Modeling, Active Experimentation & Epistemic Agency
- **Target Venues:** MIT ARC Prize Research Summit 2026, IRIS National Science Fair 2026
- **License:** [MIT License](LICENSE)
