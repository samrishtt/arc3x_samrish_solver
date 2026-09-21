# ARC-AGI-3 Solver: Version 19 ("The Level-Climber")

## 1. Executive Summary
Version 19 shatters the mathematical Level 0 ceiling (3.52) by deploying a unified **Dual-Engine (System 1 Reflexes + System 2 Cognition) with a Cross-Level Memory Bridge**.

Historically, single-level solvers plateau below 3.52 because Level 0 contributes only a tiny fraction of the total game weight, and the competition harness wipes memory between levels. Version 19 directly eliminates both bottlenecks.

---

## 2. Mathematical Law of the ARC-AGI-3 Scorer
The competition score across the 110 private games is defined as:
$$\text{level } i \text{ score} = \min\left(115, \left(\frac{\text{baseline\_actions}_i}{\text{actions\_spent}_i}\right)^2 \times 100\right) \quad (\text{if cleared, else } 0)$$
$$\text{game score} = \min\left( \frac{\sum (\text{score}_i \times i)}{\sum i}, \frac{\sum \text{cleared } i}{\sum \text{all } i} \times 100 \right)$$

### The Depth Multiplier
| Levels Cleared | Mean Achievable Score | Notes |
| :--- | :---: | :--- |
| **Level 0 only** | **3.52** | Hard mathematical ceiling for any solver that never clears Level 1. |
| **Levels 0 + 1** | **10.57** | $3\times$ increase at baseline speed. |
| **Levels 0 + 1 + 2** | **21.14** | $6\times$ increase. |

**Key Axiom**: An action spent on a level you clear costs score quadratically. An action spent on a level you fail costs zero score.

---

## 3. Four Core Architectural Pillars

### Pillar 1: System 1 Reflexes (`arc3x.autopilot`)
* Sub-millisecond (0.001s) NumPy path execution.
* BFS flood-fill and Monotone Ratchet objective tracking.
* Bypasses slow LLM inference for verified movement and obstacle navigation, saving thousands of seconds of wall clock across the 110 games.

### Pillar 2: System 2 Cognition (13-Agent ECC Dialectical Swarm)
* Deployed on Qwen 3.8 NVFP4 (RadixArk ModelOpt FP4) on dedicated enterprise GPU (`NvidiaRtxPro6000`).
* Engaged only when high-level reasoning, novel mechanics, or hypothesis elimination are required.

### Pillar 3: Memory Bridge (`taaf_grafts.recovery` R3 Knowledge Carrier)
* **Problem**: The competition gateway wipes chat history and state upon level completion (the "Vendor Level Wipe"), creating amnesia on Level 1.
* **Solution**: `R3` distills avatar color, movement deltas, and discovered rules into `cross_level_notes` (the sole persistent dictionary untouched by the level wipe).
* **Result**: Level 1 and Level 2 inherit mechanics learned on Level 0, enabling instant clearance.

### Pillar 4: Resilience & Anti-Stall Guards
* **R1 Death-Spiral Cleaner**: Purges toxic failure history upon repeated death (`PROBE_MAX_ACTIONS = 0` to avoid action-wasting probe tax).
* **Active Cycle Breaker**: Detects 2-cycle ping-pong oscillation and static obstacle collisions, injecting corrective shock prompts.
* **Graceful Buzzer Protection**: Enforces 60-second end-of-game buffer, preventing HTTP timeout exceptions.

---

## 4. Repository Structure
* `notebooks/arc3x-sam-solver-v19.ipynb`: Complete competition solver notebook.
* `arc3x/`: System 1 reflex engine (`autopilot.py`, `dream.py`, `percept.py`, `progress.py`).
* `taaf_grafts/`: Composite graft stack (`composite.py`, `recovery.py`, `retry_guard.py`).
* `ecc/`: 13-Agent dialectical swarm and cognitive harness.
