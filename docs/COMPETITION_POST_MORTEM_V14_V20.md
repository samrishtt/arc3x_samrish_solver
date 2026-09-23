# ARC-AGI-3 Competition Post-Mortem & Progression Log (V14 – V20)

> **Competition Target:** ARC Prize 2026 (`arc-prize-2026-arc-agi-3`)  
> **Hardware Environment:** Kaggle Dedicated Blackwell GPU (`NvidiaRtxPro6000`, 102 GB VRAM)  
> **Evaluation Mode:** Offline Environment (`enable_internet: false`, `enable_gpu: true`)  
> **Repository:** [samrishtt/arc3x_samrish_solver](https://github.com/samrishtt/arc3x_samrish_solver)  
> **Lead Researcher:** Samrish ([@samrishtt](https://github.com/samrishtt))

---

## Executive Summary: The Empirical Reality

Between Version 14 and Version 20, our research program explored exhaustive prompt scaffolding, multi-agent swarms, dialectical critics, in-sandbox BFS lookahead, test-time action adaptation (TAAF), context window expansions, and heuristic loop-breakers. 

While these scaffolding techniques produced spectacular local benchmark results—reaching a **5.68** audit score in V18 and **5.02** in V20 across the 25 public games (solving levels in 21/25 games)—they encountered a **hard empirical plateau on the live hidden leaderboard**, hovering persistently between **`1.43`** and **`1.56`**.

| Version | Core Architecture & Scaffolding | Public-25 Audit Score | Live Leaderboard Score | Operational Outcome & Status |
| :--- | :--- | :---: | :---: | :--- |
| **V14** | Dialectical System 2 (Proposer vs Critic) | 3.12 | 1.38 | Completed; Critic caused high latency & false rejections |
| **V15** | In-Sandbox Mental BFS (3-Step Lookahead) | 3.45 | 1.41 | Completed; Mental simulation drifted from real grid physics |
| **V16** | TAAF Transfer (Cross-Level Action Caching) | 3.80 | 1.45 | Completed; Rule mutations in Level 1+ invalidated cached actions |
| **V17** | Short-Circuiting & Anti-Oscillation Trimming | 4.10 | **1.52** | Completed; Eliminated 2-step ping-pongs, but model looped in 4-step cycles |
| **V18** | 13-Agent ECC Swarm (Coordinate Mapper + Spatial HUD) | **5.68** (18/25 solved) | **1.56** | Completed; Massive divergence between public games & unseen private test set |
| **V19** | Level Climber: 57K Context Expansion + Autopilot | 0.87 (Aborted) | *Not Submitted* | **Failure:** vLLM 500 OOM errors; Autopilot burned action budget blindly |
| **V20** | Astra-Apex: Clamped 28K Context, Disabled Autopilot | **5.02** (21/25 solved) | **1.43** | Completed cleanly; Confirmed hard mathematical ceiling of compressed model |

---

## Part 1: Deep Root Cause Analysis — Why Every Prompt Scaffolding Idea Flopped

### 1. The "Scaffolding Illusion" on Sub-4B / Compressed Models
The primary foundation model deployed in V14–V20 was `keithtyser/qwen3-8-flash-next-nvfp4` (RadixArk NVFP4 quantization). 

Across 11 distinct graft experiments, prompt scaffolding yielded virtually zero transfer to unseen games. As documented in our internal architecture logs:
> *"Graft-style tuning of the LLM notebook is 0 for 11 experiments. The only change that ever moved the score was swapping the base model."*

**Why did this occur?**
- **Orchestrating Hallucinations:** Multi-agent architectures (like the 13-agent ECC Swarm or Proposer-Critic dialectics) assume that dividing reasoning into specialized sub-tasks improves accuracy. However, when the underlying model lacks parametric spatial reasoning, the Proposer hallucinates moves, and the Critic (running on the identical model) lacks the capacity to verify the physics. Orchestrating hallucinations merely yields structured hallucinations.
- **Cognitive Overload via Prompt Bloat:** The 13-agent swarm injected thousands of tokens of instructions, personas, coordinate guidelines, and role boundaries into the system prompt. For a compressed model, this massive prompt diluted the attention heads, degrading its ability to attend to the raw ASCII grid and transition diffs.

### 2. The Divergence: Public-25 Games vs. Hidden Private Test Games
A critical discovery from V18 and V20 is the stark gap between the public audit score and the live competition score:
- **Public Audit Score:** **5.02 – 5.68** (Clearing 18 to 21 out of 25 games)
- **Live Leaderboard Score:** **1.43 – 1.56**

**Root Cause:**
- The 25 public games (`tn36`, `wa30`, `tr87`, `lp85`, etc.) feature known visual archetypes, deterministic opening moves, and predictable interaction surfaces. Heuristics, hardcoded twins, and prompt cues can navigate Level 0 of these games with high frequency.
- The **live hidden test set** consists of completely novel, out-of-distribution games with unannounced mechanics: multi-body collision physics, gravity vectors, color transformations, key-lock sequences, and Sokoban-style push mechanics.
- A model with low parametric capacity cannot infer novel causal transition rules from visual observations on the fly. It can only reproduce familiar statistical associations.

### 3. The Hardware Paradox: Starving a 102 GB Supercomputer
The runtime environment is configured with `machine_shape: "NvidiaRtxPro6000"`, providing **102 GB of ultra-high-bandwidth Blackwell VRAM**.

- Running `Qwen3.8-Flash-Next-NVFP4` consumed **< 3 GB of VRAM**.
- **Over 95 GB (>95%) of GPU memory sat completely idle during the entire 9-hour competition execution window.**
- Rather than leveraging the immense hardware capacity to run frontier-class 32B or 72B reasoning models, we attempted to compensate for a 3.8B model's cognitive deficits through complex CPU-side prompt scaffolding.

---

## Part 2: Detailed Retrospective of Every Tested Idea

### Idea 1: Dialectical System 2 (Proposer vs Critic) — V14
- **Mechanism:** Decoupled action generation into a two-pass architecture. Agent A proposed an action hypothesis based on grid segmentation; Agent B critiqued the proposal for boundary collisions, danger zones, and loop invariants.
- **Observed Behavior:** The Critic suffered from high false-positive rejection rates. Valid exploratory moves were rejected because the Critic could not distinguish between intentional experimentation and invalid actions.
- **Leaderboard Impact:** Scored 1.38.

### Idea 2: In-Sandbox Mental BFS Simulation — V15
- **Mechanism:** Before emitting an action, the agent was instructed to perform a 3-to-5 step forward mental rollout in its scratchpad, simulating how the grid would transform.
- **Observed Behavior:** Hallucinatory Drift. In ARC-AGI-3, game physics are hidden. Because the model did not possess the ground-truth transition function, its simulated mental grid drifted completely from physical reality by step 2. Actions planned against hallucinated grids led to wasted steps.
- **Leaderboard Impact:** Scored 1.41.

### Idea 3: TAAF Transfer & Action Sequence Caching — V16
- **Mechanism:** Cached successful Level 0 action trajectories and transferred them as initial priors to Level 1 and Level 2.
- **Observed Behavior:** Rule Inversion Failure. ARC-AGI-3 environments deliberately introduce rule mutations at higher levels (e.g., reversing movement axes, adding moving hazards, or changing goal colors). Replaying Level 0 trajectories resulted in immediate deaths.
- **Leaderboard Impact:** Scored 1.45.

### Idea 4: Short-Circuiting & Loop Breaking — V17
- **Mechanism:** Intercepted repeated alternating action pairs (`UP` $\leftrightarrow$ `DOWN`, `LEFT` $\leftrightarrow$ `RIGHT`, or repeated clicks on identical coordinates) and forced a deterministic orthogonal perturbation.
- **Observed Behavior:** Successfully eliminated 2-step ping-pong loops. However, the model quickly adapted by falling into 3-step or 4-step circular loops (`UP` $\to$ `RIGHT` $\to$ `DOWN` $\to$ `LEFT`), bypassing the 2-step deduplication filter.
- **Leaderboard Impact:** Scored 1.52.

### Idea 5: 13-Agent Swarm (ECC Harness) — V18
- **Mechanism:** Integrated the full 13-agent collaborative system: Coordinate Mapper, Object Segmenter, Rule Verifier, Death Spiral Purge, Graveyard Memory, and Spatial HUD Filter.
- **Observed Behavior:** Reached **5.68** on public-25 games, but collapsed to **1.56** on the live leaderboard. The subagent communication overhead consumed significant runtime, and the model failed to resolve conflicting advice from different personas on novel puzzles.
- **Leaderboard Impact:** Scored 1.56.

### Idea 6: Level Climber with 57K Context Expansion & Autopilot — V19
- **Mechanism:** Expanded `LOCAL_ANALYZER_CONTEXT_WINDOW` to 57,344 tokens to retain complete multi-level game histories, coupled with `arc3x.autopilot` for autonomous heuristic recovery.
- **Observed Behavior:** Catastrophic failure. The 57K context triggered memory fragmentation and vLLM HTTP 500 internal server errors. When the LLM timed out, the autopilot engaged and burned through the entire game action budget with unguided random moves.
- **Audit Impact:** Score collapsed to 0.87.

### Idea 7: Astra-Apex (Clamped 28K Context, Disabled Autopilot, R1/R3 Recovery) — V20
- **Mechanism:** Eliminated vLLM 500 crashes by clamping context strictly to 28,000 tokens. Disabled blind autopilot (`ARC3X_PILOT=0`). Retained R1 (Death Spiral Purge) and R3 (Knowledge Carrier Handoff).
- **Observed Behavior:** Stable, zero-crash execution. Cleared 32 levels across 21/25 public games with an average score of **5.02**. However, on the live leaderboard, it scored **1.43**.
- **Leaderboard Impact:** Scored 1.43. Demonstrated that the 28K clean pipeline had fully maximized the reasoning ceiling of the base model.

---

## Part 3: The Strategic Paradigm Shift — Model Migration (V21+)

The empirical findings from V14 through V20 deliver an inescapable conclusion: **No amount of prompt engineering or multi-agent orchestration can compensate for insufficient base model capacity.**

To challenge the top tier of the ARC-AGI-3 leaderboard, the solver must transition from prompt scaffolding on a compressed model to a **high-capacity foundation model** that natively executes deep reasoning and program synthesis.

### Candidate Foundation Models Available on Kaggle

Given our **102 GB Blackwell GPU (`NvidiaRtxPro6000`)** and offline competition constraints (`enable_internet: false`), the following frontier models are available:

| Model Candidate | Parameter Scale | Native Capabilities | Fit in 102 GB VRAM | Strategic Suitability |
| :--- | :---: | :--- | :---: | :--- |
| **`qwen-lm/qwq-32b`** | 32B | DeepSeek-R1 style RL reasoning, native `<think>` traces, complex spatial logic | **Yes** (~20 GB FP8 / ~64 GB BF16) | **Top Choice:** State-of-the-art open reasoning model; excels at puzzle solving |
| **`deepseek-ai/deepseek-r1`** (Distill Qwen 32B) | 32B | Frontier reasoning distillation, mathematical proofs, systematic hypothesis falsification | **Yes** (~20 GB FP8 / ~64 GB BF16) | **Top Choice:** Proven track record on ARC-style deductive reasoning |
| **`qwen-lm/qwen2.5-coder`** | 32B / 72B | SOTA code generation, native 2D grid/array manipulation via Python tool calls | **Yes** (32B BF16 / 72B FP8) | **High:** ARC winners achieve highest accuracy when LLMs write search scripts |
| **`qwen-lm/qwen-3`** | Dense & MoE | Next-generation hybrid attention, extended context handling | **Yes** | **High:** Official next-gen Qwen architecture |

### The V21 Architectural Blueprint

1. **Model Swap in `kernel-metadata.json`:**
   - Replace `keithtyser/qwen3-8-flash-next-nvfp4` with `qwen-lm/qwq-32b` or `qwen-lm/qwen2.5-coder`.
2. **Simplified, High-Bandwidth Prompting:**
   - Strip away the 13-agent persona bloat.
   - Present the model with clean, unambiguous state representations: connected component masks, coordinate diffs, and action space.
3. **Native Thinking Traces (`<think>`):**
   - Allow the 32B reasoning model to utilize its native reinforcement-learned chain of thought to falsify hypotheses before emitting tool actions.
4. **Code-Centric Tool Execution:**
   - Equip the model to write Python algorithms (BFS, DFS, flood fill, pattern matching) directly inside its ephemeral execution sandbox.
