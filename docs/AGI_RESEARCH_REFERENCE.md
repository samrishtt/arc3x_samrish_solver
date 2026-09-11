# AGI Research Direction: Reference Document

> **Core Research Focus**: Self-Correcting World Models Through Active Experimentation  
> **Target Uses**: Quick pitch / explainer, IRIS Science Fair Proposal, and MIT / ARC Prize research presentation.

---

## 1. The Big Picture

### What are we trying to understand?

We are interested in a fundamental question:

> **How could an artificial system learn how an unfamiliar world works, reason about that world, experiment to improve its understanding, create solutions, recognize when its understanding is wrong, and continually become better at solving new problems?**

That is the broader AGI problem.

We are **not claiming to build complete AGI in five days**.

Instead, we investigate one potentially important mechanism behind general intelligence:

> **Self-correcting world models learned through active experimentation.**

The core setup is an agent entering an unknown environment, constructing an internal world model, maintaining competing explanations, simulating actions, actively experimenting, comparing predictions with reality, and revising its model upon prediction error.

---

## 2. What Does AGI Mean?

**AGI = Artificial General Intelligence.**

There is no single universally accepted engineering specification for AGI, but the central idea is **generality**.

A narrow AI might be:

```text
Chess AI          Image Classifier         Coding Model
    ↓                    ↓                      ↓
Excellent at chess   Excellent at vision     Excellent at code
```

Each may be extremely capable within its domain, but each is optimized around particular predefined capabilities.

A general intelligence should be able to take on **many different kinds of unfamiliar problems**, learn what is needed, reason about them, adapt, and act.

Conceptually:

```text
                       GENERAL INTELLIGENCE
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
       LEARN                  THINK                 ACT
         │                     │                     │
    adapt to new          reason about          execute plans
      problems               problems
```

---

## 3. What Capabilities Might AGI Contain?

AGI isn't one monolithic ability; it is an ensemble of interacting capabilities:

* **Perception**: Understanding what is happening in the environment (pixels $\to$ objects $\to$ relations $\to$ scene understanding). For a software agent, structured observations.
* **Memory**: Retaining useful episodic and semantic information about previous experience so it doesn't restart from zero.
* **Learning**: Changing behavior or internal knowledge representations based on experience.
* **Reasoning**: Deriving consequences from known information (deduction, induction, causal reasoning, hypothesis formation, explanation, counterfactual rollouts, planning).
* **Exploration**: Actively investigating when uncertain: *"What happens if I try this?"*
* **Planning**: Considering multiple possible future actions before committing.
* **Adaptation**: Detecting when the world changes and assumptions no longer hold.
* **Generalization**: Transferring abstract structural rules across differing surface appearances (e.g., entity interaction causing state transition).

---

## 4. Does AGI Include Creating Things?

**Yes. Absolutely.** Creation is a vital capability of general intelligence:

* Writing programs
* Designing machines & experiments
* Inventing algorithms
* Formulating scientific hypotheses
* Synthesizing tools

> **However: Being able to create something does not by itself make an AI an AGI.** (e.g., an image generator creates images, but lacks reasoning, hypothesis testing, and world modeling).

The critical loop is:
$$\text{Understand} \longrightarrow \text{Reason} \longrightarrow \text{Design} \longrightarrow \text{Create} \longrightarrow \text{Test} \longrightarrow \text{Learn} \longrightarrow \text{Improve}$$

---

## 5. Creation as Part of the Learning Loop

Creation can be embedded directly into the discovery process:
```text
Observe Anomaly → Generate Hypotheses → Design Experiment → Create Experiment → Execute → Analyze → Revise
```
Creation is not a detached side module; it is the generator in the **scientific discovery loop**.

---

## 6. The Central Idea of Our Research

Narrowing from AGI as a cosmic concept to a concrete engineering mechanism:

# **Self-Correcting World Models Through Active Experimentation**

> **Can an AI agent become better at learning unfamiliar environments if it maintains an internal world model, keeps competing hypotheses, actively chooses informative experiments, and revises its model when predictions fail?**

---

## 7. What is a World Model?

A **world model** is an internal representation that helps the agent predict what will happen next. It represents:
* Objects and entities
* States and properties
* Relationships and interactions
* Transition rules
* Epistemic uncertainty

---

## 8. Memory vs. World Model

This distinction is crucial:

* **Memory** records experiences (*What happened*):
  * "Pressed red $\to$ nothing."
  * "Pressed red near green $\to$ blue moved."
* **World Model** explains those experiences (*Why it happened / Rules*):
  * "Hypothesis: Red + Green interaction triggers Blue state transition."

$$\text{Memory} = \text{Data / Observations} \quad\Longleftrightarrow\quad \text{World Model} = \text{Theory / Explanation}$$

---

## 9. The Agent Needs Uncertainty

When an event occurs, multiple hypotheses are consistent with the data:
* $H_1$: Red causes Blue movement ($P=0.50$)
* $H_2$: Green causes Blue movement ($P=0.20$)
* $H_3$: Red + Green interaction causes Blue movement ($P=0.30$)

The agent must hold probability distributions / beliefs over hypotheses rather than prematurely collapsing to dogma.

---

## 10. Why Multiple Hypotheses Matter

When $H_1$ and $H_2$ compete, random guessing is inefficient.  
**Optimal Strategy**: Formulate an experiment where $H_1$ and $H_2$ make **contradictory predictions** (e.g., touch Green while keeping Red away). The outcome decisively rules out or reinforces a hypothesis.

---

## 11. Active Experimentation

* **Passive Agent**: Observe $\to$ Act $\to$ Observe $\to$ Act (reactive trial-and-error).
* **Active-Learning Agent**: Observe $\to$ Quantify Uncertainty $\to$ Generate Competing Hypotheses $\to$ Select Maximally Informative Experiment $\to$ Act $\to$ Observe $\to$ Update.

The objective function shifts from *greedy reward maximization* to *expected information gain* (uncertainty reduction).

---

## 12. Mental Simulation (Counterfactual Rollouts)

The agent builds an internal simulation to "play in its mind":
* Before executing an action in reality, it projects forward:
  $$S_0 \xrightarrow{A} S_1, \quad S_0 \xrightarrow{B} S_2, \quad S_0 \xrightarrow{C} S_3$$
* It reasons across potential futures without incurring the physical/step cost of reality.

---

## 13. Prediction

The world model makes explicit counterfactual predictions:
$$\hat{Y}_{t+1} = f(S_t, A_t; \mathcal{M})$$

---

## 14. Prediction Error

When reality deviates from prediction:
$$e_t = |Y_{t+1} - \hat{Y}_{t+1}| > 0$$
The prediction error is not just a loss number; it is a **diagnostic signal**: *"Which specific assumption in my world model caused this expectation to fail?"*

---

## 15. Self-Correction

$$\text{Model} \to \text{Prediction} \to \text{Action} \to \text{Observation} \to \text{Error} \to \text{Diagnose Failure} \to \text{Revise Hypothesis} \to \text{New Model}$$

This closed-loop self-correction is the operational engine of autonomous learning.

---

## 16. Complete Research Architecture

```text
                  ENVIRONMENT
                       │
                       │ observation
                       ▼
                ┌─────────────┐
                │ PERCEPTION  │
                └──────┬──────┘
                       ▼
                ┌─────────────┐
                │   MEMORY    │
                └──────┬──────┘
                       ▼
              ┌─────────────────┐
              │   WORLD MODEL   │
              │                 │
              │ objects         │
              │ states          │
              │ relationships   │
              │ rules           │
              │ hypotheses      │
              │ uncertainty     │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │ INTERNAL        │
              │ SIMULATOR       │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │ EXPERIMENT      │
              │ SELECTOR        │
              └────────┬────────┘
                       ▼
                    ACTION
                       ▼
                 ENVIRONMENT
                       ▼
                    RESULT
                       ▼
              PREDICTION VS REALITY
                       ▼
               PREDICTION ERROR
                       ▼
                MODEL REVISION
                       │
                       └────────────→ (Repeat)
```

---

## 17. The Agent's Composition

The agent is built from modular components rather than training a massive foundation model from scratch:
$$\text{Agent} = \text{LLM Reasoning Engine} + \text{Memory Store} + \text{World Model / Hypothesis Manager} + \text{Internal Simulator} + \text{Experiment Selector} + \text{Env Interface}$$

---

## 18. The Environment

The environment provides:
1. **Observation space**: What the agent senses.
2. **Action space**: What actions are valid.
3. **State transition function**: $P(S_{t+1} \mid S_t, A_t)$ (initially hidden).
4. **Task/Goal signals**: Success criteria.

---

## 19 & 20. The Controlled Laboratory (MiniGrid vs. 3D Worlds)

We use controlled grid environments (e.g., MiniGrid / custom ARC-style grids) rather than heavy 3D simulators because the research question is:
> *"Can the agent discover hidden rules and revise its model sample-efficiently?"*

Small controlled environments allow:
* Exact ground-truth rule verification
* Rapid reproducibility and iteration
* Direct measurement of epistemic uncertainty
* Clean ablations without visual rendering confounding variables

---

## 21 & 22. Controlled Hidden-Rule Paradigm: Walkthrough

* **Setup**: Objects $R$ (Red), $G$ (Green), $B$ (Blue).
* **Hidden Rule**: Moving $R$ adjacent to $G$ flips $B$'s state.
* **Step 1**: Agent identifies ignorance. Proposes $H_1, H_2, H_3$.
* **Step 2 (Diagnostic Experiment 1)**: Move $R$ alone. Prediction: If $H_1$, $B$ moves. Observation: $B$ remains static. Update: $P(H_1) \downarrow$.
* **Step 3 (Diagnostic Experiment 2)**: Bring $R$ and $G$ together. Prediction: If $H_3$, $B$ moves. Observation: $B$ changes state. Update: $P(H_3) \approx 1.0$.

---

## 23 & 24. Experiment Selection & Creation

The agent transitions from asking *"What action brings immediate reward?"* to:
> **"What action will teach me the most and resolve maximal uncertainty?"**

When existing actions fail to distinguish hypotheses, the agent can design multi-step experimental interventions.

---

## 25. Operationalizing "Learning How to Learn"

"Learning to learn" is measured as **epistemic efficiency**:
* **Baseline**: 100 random or greedy interactions to discover a rule.
* **Active Agent**: Discovers the rule in 15 targeted experiments by targeting hypothesis divergence.

---

## 26. The Core Baseline Comparison

* **Baseline**: $\text{Observation} \to \text{LLM} \to \text{Memory} \to \text{Action}$ (Standard ReAct / Memory-augmented agent).
* **Our System**: $\text{Observation} \to \text{Memory} \to \text{World Model} \to \text{Competing Hypotheses} \to \text{Simulation} \to \text{Experiment Selection} \to \text{Action} \to \text{Error Diagnosis} \to \text{Revision}$.

---

## 27. Primary Evaluation Metrics

1. **Task Success Rate**: Percentage of hidden-rule environments solved.
2. **Sample Efficiency**: Number of environment steps required to identify true rules.
3. **Prediction Accuracy**: Match rate between simulated counterfactuals and true environmental transitions.
4. **Adaptation Latency**: Steps required to adapt when a rule is secretly inverted mid-run.
5. **Structural Transfer**: Transfer efficiency when surface properties (colors/shapes) change but relational dynamics remain invariant.
6. **Model Recovery Rate**: Speed of repairing an intentionally corrupted prior world model.

---

## 28 & 29. Stress Tests: Rule Shifts and Transfer

* **Rule Mutation Test**: $A \to B$ suddenly changes to $A \to C$. Measure steps to detect failure, suppress old beliefs, and isolate new dynamics.
* **Abstract Transfer Test**: Test whether learned interaction rules $(Entity_1 \star Entity_2 \to \Delta Entity_3)$ transfer across different visual token bindings.

---

## 30. Ablation Matrix

Systematically evaluate contributions by ablating:
* $-\text{Memory}$: Test impact of history retention.
* $-\text{World Model}$: Direct action generation without state-transition explanation.
* $-\text{Hypothesis Uncertainty}$: Force single point-estimate hypothesis.
* $-\text{Mental Simulator}$: Remove counterfactual rollout verification.
* $-\text{Active Selection}$: Replace information-gain action choice with random / greedy choice.

---

## 31. Hypothetical Research Target

| Metric | Standard Baseline | Active World-Model Agent |
| :--- | :---: | :---: |
| **Rule Discovery Steps** | 41 | **19** |
| **Transfer Accuracy** | 24% | **61%** |
| **Rule-Change Recovery Steps** | 38 | **14** |
| **Next-State Prediction Acc.** | 63% | **89%** |

---

## 32 & 33. Novelty Claim and Academic Rigor

* **Do NOT claim**: "We invented world models" or "We solved AGI."
* **Precise Novelty**:  
  > *A formal mechanism that uses prediction disagreement among competing world-model hypotheses as an explicit information signal for action selection, coupled with causal error diagnosis for model revision.*

---

## 34 & 35. The AGI Hierarchy & Creation

```text
                         AGI
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
      Learn             Reason             Act
        │                 │                 │
        └──────────┬──────┴──────┬──────────┘
                   │             │
             World modelling   Planning
                   │
          Active experimentation
                   │
        Prediction + self-correction
                   │
             OUR RESEARCH
```

---

## 36. The 30-Second "Scientist Agent" Analogy

> *"Imagine an AI dropped into an unfamiliar world without a rulebook. Instead of thrashing around randomly, it maintains several competing theories of how the world works, mentally simulates what each theory predicts, performs deliberate experiments that pit those theories against each other, observes what happens, and repairs its theory when predictions fail. Over time, it gets faster at cracking new worlds and adapting when rules change."*

---

## 37 & 38. Five-Day Prototyping Roadmap

* **Day 1**: Environment setup (Gym/MiniGrid/ARC-like grids), API harness, logging, and repo foundation.
* **Day 2**: Perception pipeline, state observation parser, and episodic/semantic memory store.
* **Day 3**: World Model representation, hypothesis generator, and internal simulator.
* **Day 4**: Prediction error diagnostic engine, model revision loop, and active experiment selector.
* **Day 5**: Baseline vs. Full Agent benchmarking, rule-shift experiments, ablations, and visualization generation.

---

## 39 & 40. Paper & Proposal Guidelines

* **Tone**: Empirical, rigorous, disciplined, hypothesis-driven.
* **Avoid**: Hyperbolic AGI claims, ungrounded philosophical assertions.
* **Emphasize**: Measurable sample efficiency, adaptation under distribution shifts, and ablation verification.

---

## 41. Key Requirements for the IRIS Science Fair Proposal

* Clear, testable hypothesis with a defined independent variable (agent architecture with active hypothesis testing vs. standard baseline) and dependent variables (sample efficiency, adaptation steps).
* Controlled testbed isolating epistemic exploration from visual complexity.
* Quantitative data presentation (box plots, step curves, ablation tables).

---

## 42. Key Requirements for ARC Prize / MIT Presentation

Framing tailored to ARC Prize 2026 Paper Prize evaluation criteria:
* **Universality & Theory**: Grounding in Bayesian experimental design, active inference, and causal model revision.
* **Accuracy & Completeness**: Concrete algorithmic formulation of hypothesis generation, scoring, and rollouts.
* **Novelty**: Disagreement-driven exploration signal for discrete symbolic/relational environments.

---

## 43. 20-Minute Technical Presentation Outline

1. **0–2 min (Problem)**: Fragility of reactive models when assumptions fail ($Prediction \neq Reality$).
2. **2–4 min (Hypothesis)**: Active discrimination among competing world models reduces sample complexity in unfamiliar environments.
3. **4–7 min (Architecture)**: Walkthrough of the Perception $\to$ World Model $\to$ Simulator $\to$ Experiment Selector $\to$ Error Revision loop.
4. **7–9 min (Literature & Gap)**: Positioning relative to Ha & Schmidhuber world models, active inference, and ReAct agents.
5. **9–14 min (Experiments)**: Three suites: Rule Discovery, Rule Mutation, and Structural Transfer.
6. **14–17 min (Results & Ablations)**: Quantitative performance graphs and ablation breakdown.
7. **17–19 min (Mechanistic Analysis)**: Why model disagreement outperforms random exploration.
8. **19–20 min (Conclusion)**: Rigorous summary of findings and future research directions.

---

## 44. The Fundamental Scientific Question

> **Can an AI turn experience into a progressively better internal model of the world, determine when that model is inadequate, decide what information it needs next, acquire that information through experimentation, and use it to create better predictions and better actions?**

---

## 45. Interlocking Cognitive Loop

```text
                AGI
                 │
 ┌───────────────┼────────────────┐
 ↓               ↓                ↓
LEARN           REASON           CREATE
 ↓               ↓                ↓
adapt          infer            design
remember       plan             build
generalize     simulate         test
                 │                │
                 └──────┬─────────┘
                        ↓
                   EXPERIMENT
                        ↓
                     OBSERVE
                        ↓
                   SELF-CORRECT
                        ↓
                    GENERALIZE
```

---

## 46. The Socratic Progression of the Agent

1. *"What do I believe?"* (Current Model)
2. *"Why do I believe it?"* (Evidence / Memory)
3. *"What would happen if I am right?"* (Simulation)
4. *"What would happen if I am wrong?"* (Counterfactual Analysis)
5. *"What experiment discriminates between these two?"* (Experiment Selection)
6. *"What did reality reveal?"* (Observation & Error)
7. *"How must my internal model adapt?"* (Self-Correction)

---

## 47. One-Minute Elevator Pitch

> *"I am investigating a mechanism for sample-efficient general intelligence: self-correcting world models through active experimentation. When placed in an environment with hidden rules, instead of relying on brute-force trial-and-error, the agent maintains an internal world model with multiple competing hypotheses. It mentally simulates possible actions, selects the experiment that maximally discriminates between those hypotheses, and executes it. When predictions fail, it diagnoses which model assumption broke down and revises its beliefs. We test whether this mechanism accelerates learning efficiency, enables rapid adaptation to rule changes, and improves transfer to novel tasks."*

---

## 48. The One-Sentence Core

> **Build an agent that enters an unfamiliar world, forms competing explanations of how it works, simulates possible futures, actively performs experiments, compares predictions with reality, repairs its internal model when wrong, and tests whether this makes it better at learning and generalizing to new worlds.**

---

## 49. The Master Mental Hierarchy

```text
                 AGI
                  │
        ┌─────────┼─────────┐
        │         │         │
      Learn     Reason     Create
        │         │         │
        └────┬────┴────┬────┘
             │         │
          Explore    Plan
             │         │
             └───┬─────┘
                 ↓
            WORLD MODEL
                 ↓
             PREDICT
                 ↓
             EXPERIMENT
                 ↓
             OBSERVE
                 ↓
              ERROR
                 ↓
           SELF-CORRECT
                 ↓
             GENERALIZE
                 ↓
                ACT
```
