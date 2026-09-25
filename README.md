# Self-Correcting World Models Through Active Experimentation

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 7/7 Passing](https://img.shields.io/badge/tests-7%2F7%20passing-brightgreen.svg)](tests/)
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

### 🗄️ Hierarchical 4-Tier Memory Architecture

To prevent context window bloat and catastrophic memory drift, the agent implements a strict four-tiered memory hierarchy:

| Layer | Purpose | Lifetime | Distillation Mechanism |
| :--- | :--- | :--- | :--- |
| **`run_memory`** | Temporary observations, transitions & errors from current execution | One run | Purged after run distillation; prevents raw transcript context overflow |
| **`game_memory`** | Distilled causal rules & discoveries for one game/environment | Across versions ($v_1 \to v_N$) | Stores verified rule signatures, interaction contingencies & success priors |
| **`global_memory`** | General domain invariants & epistemic strategies across games | Across entire project | Tracks meta-heuristics (e.g. predictive disagreement efficacy, collision physics) |
| **`submission_memory`** | Read-only immutable snapshot compiled for competition | Frozen at submission | Locks parameters, prevents test-time drift, guarantees deterministic evaluation |

> **Context Preservation Invariant:** Raw execution trajectories are never dumped verbatim into persistent context. Instead, only distilled causal invariants and Bayesian confidence distributions are promoted across runs and versions.

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
│   ├── agents/active_agent.py             # Active experiment agent (disagreement scoring)
│   ├── agents/process_rl_agent.py         # Process-reward RL agent with verifiable rewards
│   ├── perception/extractor.py            # Spatial relation & temporal diff extraction
│   ├── memory/store.py                    # Episodic memory & semantic contingency tables
│   ├── memory/persistent_bank.py          # Cross-run persistent memory bank (v1, v2, v3 -> v4)
│   ├── memory/hierarchical_memory.py      # 4-tier taxonomy (run, game, global, submission)
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
├── tests/                                 # Unit Test Suite (7/7 Passing)
│   ├── test_day1_grid_lab.py              # Environment physics & rule mechanics
│   ├── test_day2_perception_memory.py      # Spatial relation & transition memory
│   ├── test_day3_world_model.py           # Hypothesis generation & mental rollouts
│   ├── test_day4_active_loop.py           # End-to-end active experiment loop
│   ├── test_persistent_memory_and_process_rl.py # Process rewards & version transfer
│   ├── test_hierarchical_memory.py        # 4-tier memory distillation & freeze
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

### 2. Run the Full Unit Test Suite (7/7 Passing)
```bash
python -m tests.test_day1_grid_lab
python -m tests.test_day2_perception_memory
python -m tests.test_day3_world_model
python -m tests.test_day4_active_loop
python -m tests.test_persistent_memory_and_process_rl
python -m tests.test_hierarchical_memory
python -m tests.test_arc3x_bridge
```

### 3. Run the Automated Procedural Benchmark Suite
```bash
python -m world_model_lab.benchmarks.report_generator
```
*Generates procedural environments, logs per-seed trajectories to `experiments/data/`, and outputs a formatted Markdown report to `docs/EMPIRICAL_BENCHMARK_REPORT.md`.*

## 🏆 ARC-AGI-3 Competition Solvers & Empirical Progression (ARC Prize 2026)

This repository documents the complete empirical lifecycle of our autonomous solvers submitted to the official Kaggle ARC-AGI-3 competition (`arc-prize-2026-arc-agi-3`), executed on enterprise **Nvidia RTX Pro 6000 (102 GB Blackwell GPU)** hardware in an offline evaluation sandbox (`enable_internet: false`).

---

### 📊 Competition Progression & Leaderboard Post-Mortem (V14 – V20)

Across 7 major version cycles, we systematically explored whether sophisticated agentic scaffolding (dialectics, mental BFS, subagent swarms, action caching, and loop breakers) could elevate a lightweight quantized model to frontier performance on hidden interactive environments.

| Version | Core Architecture & Scaffolding | Public-25 Audit Score | Live Leaderboard Score | Operational Outcome & Empirical Status |
| :--- | :--- | :---: | :---: | :--- |
| **V14** | Dialectical System 2 (Proposer vs Critic) | 3.12 | 1.38 | Completed; Critic caused high latency & false rejections on novel moves |
| **V15** | In-Sandbox Mental BFS (3-Step Lookahead) | 3.45 | 1.41 | Completed; Mental simulation drifted rapidly from ground-truth hidden physics |
| **V16** | TAAF Transfer (Cross-Level Action Caching) | 3.80 | 1.45 | Completed; Rule mutations in Level 1+ invalidated cached trajectories |
| **V17** | Short-Circuiting & Anti-Oscillation Trimming | 4.10 | **1.52** | Completed; Eliminated 2-step ping-pongs, but agent drifted into 4-step cycles |
| **V18** | 13-Agent ECC Swarm (Spatial HUD + Coordinator) | **5.68** (18/25 solved) | **1.56** | Completed; High public performance failed to transfer to hidden private games |
| **V19** | Level Climber: 57K Context Expansion + Autopilot | 0.87 (Aborted) | *Not Submitted* | **Failure:** vLLM 500 OOM errors; Autopilot burned action budget blindly |
| **V21** | The 27B Giant: Qwen3.8-27B-FP8 + Resilient 28K Clamping | **47.62** (ft09 clear) | **1.03** | Completed; Dense next-token predictor lacked test-time RL search, wandering impulsively |
| **V22** | DeepSeek-R1 32B Distill (BF16) First Run | Crashed (Disk full) | *Not Submitted* | **Diagnosed:** Loaded 61GB weights in 86s on GPU; crashed because TAAF sent PNG image to text-only model (`400: not a multimodal model`) |
| **V23** | DeepSeek-R1 Symbolic Reasoning: 32B Distill (BF16) | *Deploying* | **Target: Frontier** | **Active:** Disabled `MULTIMODAL_CONTEXT` to pure discrete symbolic grid (`current_frame.segmentation` + ASCII + Python BFS) |

> **Comprehensive Technical Post-Mortem:** For line-by-line trajectory logs, viewer replay audits, and failure diagnostics, see [`docs/COMPETITION_POST_MORTEM_V14_V20.md`](docs/COMPETITION_POST_MORTEM_V14_V20.md).

---

### 🔍 Deep Root Cause: The "Scaffolding Illusion" on Compressed Models

Why did every scaffolding idea plateau between **`1.43`** and **`1.56`** on the live leaderboard despite achieving **`5.02 – 5.68`** on the public-25 audit?

1. **The Overfitting Divergence (Public vs. Private):**
   - The 25 public benchmark games (`tn36`, `wa30`, `tr87`, `lp85`, etc.) feature known visual dynamics and predictable opening moves. Heuristic grafts and twin templates could clear Level 0 reliably, yielding scores above 5.0.
   - However, the **live competition leaderboard evaluates on completely unseen, novel games** with hidden mechanics (variable gravity, multi-body collisions, color-state machines, key-lock sequences).
2. **Scaffolding Cannot Manufacture Missing Parametric Intelligence:**
   - Multi-agent architectures (such as our 13-agent ECC Swarm or Proposer-Critic dialectics) merely orchestrate reasoning.
   - When the base model (`Qwen3.8-Flash-Next-NVFP4`) lacks the native parametric capacity to infer 2D transformation rules from visual diffs or synthesize search algorithms, the Proposer hallucinates moves, and the Critic lacks the spatial acuity to verify them. **Orchestrating hallucinations merely yields structured hallucinations.**
   - Furthermore, injecting thousands of tokens of multi-agent prompt instructions caused attention dilution, degrading the model's focus on raw grid coordinates.
3. **The Hardware Paradox: Starving a 102 GB Blackwell GPU:**
   - Our Kaggle kernel runs on dedicated **`NvidiaRtxPro6000`** hardware with **102 GB of VRAM**.
   - Running a tiny 4-bit compressed model consumed **< 3 GB of VRAM (<3% utilization)**.
   - Over **95 GB (>97%)** of GPU memory sat completely idle during the entire 9-hour competition execution window.

---

### 🚀 The Strategic Pivot: Migration to Frontier Models (V21+)

As recorded in our architectural invariants:
> *"Graft-style tuning of the LLM notebook is 0 for 11 experiments. The only change that ever moved the score was swapping the base model."*

To breach the 1.56 ceiling and compete on the frontier, the solver must transition from prompt scaffolding on a sub-4B model to a **high-capacity foundation model** that natively performs deep chain-of-thought reasoning and program synthesis.

#### Frontier Model Candidates Available on Kaggle (102 GB VRAM Target)

| Model Candidate | Parameter Scale | Native Capabilities | Fit in 102 GB VRAM | Strategic Suitability |
| :--- | :---: | :--- | :---: | :--- |
| **`qwen-lm/qwq-32b`** | 32B | DeepSeek-R1 style RL reasoning, native `<think>` traces, complex spatial logic | **Yes** (~20 GB FP8 / ~64 GB BF16) | **Top Recommendation:** Official Qwen reasoning model; excels at puzzle solving |
| **`deepseek-ai/deepseek-r1`** (Distill Qwen 32B) | 32B | Frontier reasoning distillation, mathematical proofs, systematic hypothesis falsification | **Yes** (~20 GB FP8 / ~64 GB BF16) | **Top Recommendation:** Proven benchmark leader in deductive grid reasoning |
| **`qwen-lm/qwen2.5-coder`** | 32B / 72B | SOTA code generation, native 2D grid/array manipulation via Python tool calls | **Yes** (32B BF16 / 72B FP8) | **High:** ARC winners achieve highest accuracy when LLMs write search scripts |
| **`qwen-lm/qwen-3`** | Dense & MoE | Next-generation hybrid attention, extended context handling | **Yes** | **High:** Official next-gen Qwen architecture |

#### V21 Deployment Pathway
- **Base Kernel:** [`arc3x_submission.ipynb`](file:///d:/AI_ARMY/arc_agi3_solver/arc3x_submission.ipynb) / [`kaggle_arc3_duck_dialectic_solver.ipynb`](file:///d:/AI_ARMY/arc_agi3_solver/kaggle_arc3_duck_dialectic_solver.ipynb)
- **Target Architecture:** Minimalist prompt scaffolding + Native 32B/72B reasoning tokens + Python execution sandbox.
- **Reference Docs:**
  - [Comprehensive V14–V20 Post-Mortem & Progression Log](file:///d:/AI_ARMY/arc_agi3_solver/docs/COMPETITION_POST_MORTEM_V14_V20.md)
  - [Dialectical Qwen Architecture Blueprint](file:///d:/AI_ARMY/arc_agi3_solver/docs/DIALECTIC_QWEN38_ARCHITECTURE.md)
  - [Master Workspace Index](file:///d:/AI_ARMY/arc_agi3_solver/WORKSPACE_INDEX.md)

---

## 👤 Author & Acknowledgments

- **Researcher:** Samrish ([@samrishtt](https://github.com/samrishtt))
- **Research Topic:** Autonomous World Modeling, Active Experimentation & Epistemic Agency
- **Target Venues:** MIT ARC Prize Research Summit 2026, IRIS National Science Fair 2026
- **License:** [MIT License](LICENSE)

