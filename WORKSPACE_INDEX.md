# ARC-AGI-3 & World Model Research: Master Workspace Index

Welcome! This document provides an **instant-lookup catalog** with clickable links to every notebook, solver module, research paper, keynote guide, experiment, and tool in this repository.

---

## 1. Kaggle Competition & Submission (ARC Prize 2026)

| Resource | Path | Description |
| :--- | :--- | :--- |
| **Option B (Premier Neural Solver)** | [kaggle_arc3_duck_dialectic_solver.ipynb](file:///d:/AI_ARMY/arc_agi3_solver/kaggle_arc3_duck_dialectic_solver.ipynb) | High-scoring frontier neural solver: Qwen 3.8-Flash-Next-NVFP4 on Nvidia RTX Pro 6000 + Dialectical System 1/System 2 World Model + TAAF Transfer. |
| **Option A (Standalone Offline)** | [kaggle_arc3_submission.ipynb](file:///d:/AI_ARMY/arc_agi3_solver/kaggle_arc3_submission.ipynb) | 19-cell self-contained notebook bundling Dialectical Debate, Neural Student, and embedded 25-game plans. Outputs `submission.parquet` and `submission.csv`. |
| **Ultra-Slim Submission Notebook** | [kaggle_arc3_slim_submission.ipynb](file:///d:/AI_ARMY/arc_agi3_solver/kaggle_arc3_slim_submission.ipynb) | Only 4 cells (14 KB)! Loads skills directly from attached Kaggle dataset bundle. |
| **Kaggle Skills Dataset Zip** | [dist/arc3x_skills_bundle.zip](file:///d:/AI_ARMY/arc_agi3_solver/dist/arc3x_skills_bundle.zip) | 1-click uploadable Kaggle Dataset package containing `arc3x/`, `world_model_lab/`, and `plans.json`. |
| **Production Notebook Copy** | [arc3x_submission.ipynb](file:///d:/AI_ARMY/arc_agi3_solver/arc3x_submission.ipynb) | Identical production copy for backup and alternative kernel deployment. |
| **Notebooks Directory** | [notebooks/](file:///d:/AI_ARMY/arc_agi3_solver/notebooks/) | Folder containing all competition notebooks, including [arc3_apex_solver_v20.ipynb](file:///d:/AI_ARMY/arc_agi3_solver/notebooks/arc3_apex_solver_v20.ipynb) and [kaggle_arc3_duck_dialectic_solver.ipynb](file:///d:/AI_ARMY/arc_agi3_solver/notebooks/kaggle_arc3_duck_dialectic_solver.ipynb). |
| **Dialectical Qwen-3.8 Architecture** | [docs/DIALECTIC_QWEN38_ARCHITECTURE.md](file:///d:/AI_ARMY/arc_agi3_solver/docs/DIALECTIC_QWEN38_ARCHITECTURE.md) | Technical blueprint connecting Qwen-3.8-Flash-Next-NVFP4 with the Dialectical Multi-Agent System 2 reasoning loop. |
| **Online API Evaluator** | [run_arc3_online_eval.py](file:///d:/AI_ARMY/arc_agi3_solver/run_arc3_online_eval.py) | Standalone client that connects to the live ARC-AGI-3 API using an API key to benchmark and generate official scorecards. |
| **Pre-Computed Plans** | [arc3x/plans.json](file:///d:/AI_ARMY/arc_agi3_solver/arc3x/plans.json) | High-scoring, verified solution action sequences for all 25 game families. |

---

## 2. Research Papers, Keynotes & Science Fair Submissions

| Document | Path | Purpose |
| :--- | :--- | :--- |
| **Full Research Paper (15 Pages)** | [docs/FULL_RESEARCH_PAPER.md](file:///d:/AI_ARMY/arc_agi3_solver/docs/FULL_RESEARCH_PAPER.md) | Formal academic manuscript on Active Epistemic World Models and Neuro-Symbolic Debate for AGI. |
| **60-Minute Keynote Presentation** | [docs/KEYNOTE_PRESENTATION_45_60MIN.md](file:///d:/AI_ARMY/arc_agi3_solver/docs/KEYNOTE_PRESENTATION_45_60MIN.md) | Complete 6-Act keynote speech script with slide visuals, timing cues, Class 12 analogies, and MIT defense. |
| **IRIS 2025-26 Submission Package** | [docs/IRIS_SUBMISSION.md](file:///d:/AI_ARMY/arc_agi3_solver/docs/IRIS_SUBMISSION.md) | Official IRIS National Science Fair package (Deadline: Oct 3, 2026). |
| **IRIS Mandatory Portal Fields** | [docs/IRIS_MANDATORY_FIELDS.md](file:///d:/AI_ARMY/arc_agi3_solver/docs/IRIS_MANDATORY_FIELDS.md) | Copy-paste-ready text fields for the IRIS online application portal. |
| **Empirical Benchmark Report** | [docs/EMPIRICAL_BENCHMARK_REPORT.md](file:///d:/AI_ARMY/arc_agi3_solver/docs/EMPIRICAL_BENCHMARK_REPORT.md) | Detailed empirical results, statistical proof ($p < 0.001$), and ablation autopsy. |
| **90-Second Video Pitch Script** | [docs/PROJECT_VIDEO_SCRIPT_90S.md](file:///d:/AI_ARMY/arc_agi3_solver/docs/PROJECT_VIDEO_SCRIPT_90S.md) | Timed audio-visual script for competition/fair video submission. |
| **Project Synopsis** | [docs/PROJECT_SYNOPSIS.md](file:///d:/AI_ARMY/arc_agi3_solver/docs/PROJECT_SYNOPSIS.md) | Concise executive summary of the research methodology and findings. |
| **AGI Research Reference** | [docs/AGI_RESEARCH_REFERENCE.md](file:///d:/AI_ARMY/arc_agi3_solver/docs/AGI_RESEARCH_REFERENCE.md) | Literature review and theoretical grounding in cognitive science and world models. |

---

## 3. Engineering & Architecture Guides

| Guide | Path | Focus Area |
| :--- | :--- | :--- |
| **System Architecture** | [docs/ARCHITECTURE.md](file:///d:/AI_ARMY/arc_agi3_solver/docs/ARCHITECTURE.md) | Complete system block diagram, data structures, and pipeline interfaces. |
| **Execution Flow** | [docs/EXECUTION_FLOW.md](file:///d:/AI_ARMY/arc_agi3_solver/docs/EXECUTION_FLOW.md) | Step-by-step lifecycle from raw sensory grid to action arbitration and replay. |
| **Local Testing Guide** | [docs/guides/LOCAL_TESTING_GUIDE.md](file:///d:/AI_ARMY/arc_agi3_solver/docs/guides/LOCAL_TESTING_GUIDE.md) | How to run tests, twin simulations, and local benchmarks on Windows/Linux. |
| **Notebook Walkthrough** | [docs/guides/NOTEBOOK_WALKTHROUGH.md](file:///d:/AI_ARMY/arc_agi3_solver/docs/guides/NOTEBOOK_WALKTHROUGH.md) | Detailed walkthrough of each cell and execution mode in the Kaggle notebook. |
| **Detailed Analysis** | [docs/guides/DETAILED_ANALYSIS.md](file:///d:/AI_ARMY/arc_agi3_solver/docs/guides/DETAILED_ANALYSIS.md) | In-depth breakdown of game mechanics, failure modes, and plan compression. |
| **Official Competition Docs** | [docs/official_docs/](file:///d:/AI_ARMY/arc_agi3_solver/docs/official_docs/) | Official ARC Prize Foundation reference docs (Scoring, Actions, Scorecards, Swarms). |

---

## 4. Core Scientific Research System (`world_model_lab/`)

| Module | Path | Description |
| :--- | :--- | :--- |
| **Dialectical Debate Agent** | [`DebateAgent`](file:///d:/AI_ARMY/arc_agi3_solver/world_model_lab/agents/debate_agent.py) | Dual-agent debate (Proposer vs Adversarial Critic) with grounded simulator arbitration. |
| **Active World Model Agent** | [`ActiveWorldModelAgent`](file:///d:/AI_ARMY/arc_agi3_solver/world_model_lab/agents/active_agent.py) | Active inference agent driven by epistemic surprise and predictive disagreement. |
| **Process-RL Agent** | [`ProcessRLAgent`](file:///d:/AI_ARMY/arc_agi3_solver/world_model_lab/agents/process_rl_agent.py) | Agent utilizing dense step-level process supervision. |
| **Hypothesis Manager** | [`HypothesisManager`](file:///d:/AI_ARMY/arc_agi3_solver/world_model_lab/world_model/hypothesis_manager.py) | Generates, evaluates, and prunes symbolic rules of the environment. |
| **Counterfactual Simulator** | [`MentalSimulator`](file:///d:/AI_ARMY/arc_agi3_solver/world_model_lab/world_model/simulator.py) | Runs lookahead rollouts in the agent's internal mental model. |
| **Epistemic Action Selector** | [`ActionSelector`](file:///d:/AI_ARMY/arc_agi3_solver/world_model_lab/experimentation/selector.py) | Selects experiments to maximize knowledge gain ($\Delta H$). |
| **Hypothesis Revision** | [`RevisionEngine`](file:///d:/AI_ARMY/arc_agi3_solver/world_model_lab/diagnostics/revision.py) | Diagnoses prediction errors and falsifies incorrect rules. |
| **Perception Extractor** | [`PerceptionExtractor`](file:///d:/AI_ARMY/arc_agi3_solver/world_model_lab/perception/extractor.py) | Parses 64x64 grids into discrete entities, agents, and spatial bounds. |
| **Hierarchical Memory** | [`HierarchicalMemory`](file:///d:/AI_ARMY/arc_agi3_solver/world_model_lab/memory/hierarchical_memory.py) | Working, episodic, and conceptual long-term memory hierarchy. |
| **Core Types** | [`types.py`](file:///d:/AI_ARMY/arc_agi3_solver/world_model_lab/core/types.py) | Central dataclasses: `Action`, `Observation`, `Hypothesis`, `PredictionError`. |

---

## 5. ARC-AGI-3 Solver Engine (`arc3x/`)

| Module | Path | Description |
| :--- | :--- | :--- |
| **Twin Simulator** | [`twin.py`](file:///d:/AI_ARMY/arc_agi3_solver/arc3x/twin.py) | Fast in-process deepcopy game engine (~700–1,000 steps/sec on CPU for free). |
| **Dialectical Debate** | [`debate.py`](file:///d:/AI_ARMY/arc_agi3_solver/arc3x/debate.py) | Multi-agent debate controller (Proposer vs Critic) + Qwen-3.8-27B vLLM adapter. |
| **Student Policy** | [`student.py`](file:///d:/AI_ARMY/arc_agi3_solver/arc3x/student.py) | Neural network policy (MLP) trained on compressed winning trajectories. |
| **Go-Explore Search** | [`explore.py`](file:///d:/AI_ARMY/arc_agi3_solver/arc3x/explore.py) | Archive-based situation exploration and plan compression engine. |
| **Parallel Sweep** | [`sweep.py`](file:///d:/AI_ARMY/arc_agi3_solver/arc3x/sweep.py) | Multi-core parallel search across all 25 game families. |
| **Gateway Runner** | [`runner.py`](file:///d:/AI_ARMY/arc_agi3_solver/arc3x/runner.py) | Connects to competition gateway (`http://gateway:8001/`) and replays plans. |
| **A* Maze Solver** | [`maze_solver.py`](file:///d:/AI_ARMY/arc_agi3_solver/arc3x/maze_solver.py) | Topological pathfinding and corridor navigation (solves `wa30`, `tr87`). |
| **Click Solver** | [`click_solver.py`](file:///d:/AI_ARMY/arc_agi3_solver/arc3x/click_solver.py) | Connected-component click target reduction (solves `su15`, `tn36`). |
| **Sokoban Solver** | [`sokoban_solver.py`](file:///d:/AI_ARMY/arc_agi3_solver/arc3x/sokoban_solver.py) | Push-path BFS search and deadlock avoidance (solves `sc25`). |
| **Notebook Builder** | [`make_notebook.py`](file:///d:/AI_ARMY/arc_agi3_solver/arc3x/make_notebook.py) | Bundles all modules and plans into self-contained competition notebooks. |

---

## 6. Experiments, Sweeps & Benchmarks (`experiments/`)

| Resource | Path | Description |
| :--- | :--- | :--- |
| **Debate Ablation Study** | [experiments/debate_ablation_study.py](file:///d:/AI_ARMY/arc_agi3_solver/experiments/debate_ablation_study.py) | 25-seed empirical study comparing Debate vs Greedy vs Random (64% win rate). |
| **Ablation Results Data** | [experiments/data/debate_experiment_results.json](file:///d:/AI_ARMY/arc_agi3_solver/experiments/data/debate_experiment_results.json) | Full statistical output JSON with trap veto counts and confidence intervals. |
| **Game Family Sweeps** | [experiments/sweeps/](file:///d:/AI_ARMY/arc_agi3_solver/experiments/sweeps/) | Detailed logs and JSON outputs for 25-family search iterations (`sweep25_v2..v4`). |
| **Official Scorecards** | [experiments/scorecards/](file:///d:/AI_ARMY/arc_agi3_solver/experiments/scorecards/) | Stored official ARC-AGI-3 online scorecard results (`11d09afd-2263-...`). |
| **Experiment Runners** | [experiments/scripts/](file:///d:/AI_ARMY/arc_agi3_solver/experiments/scripts/) | Scripts for running cognitive brain evaluations and game-specific tests. |

---

## 7. Tests & Verification (`tests/`)

- **Test Suite Directory**: [tests/](file:///d:/AI_ARMY/arc_agi3_solver/tests/)
- **Run all 31 tests**:
  ```powershell
  python -m unittest discover tests/
  ```
- **Tests included**:
  - `test_debate_agent.py`: Multi-agent debate and hazard veto logic.
  - `test_day3_world_model.py`: Hypothesis generation and mental simulator.
  - `test_day4_active_loop.py`: Active inference and epistemic exploration.
  - `test_arc3x_bridge.py`: Twin bridge and action execution.
  - `test_hierarchical_memory.py`: Multi-tier memory retention and recall.
