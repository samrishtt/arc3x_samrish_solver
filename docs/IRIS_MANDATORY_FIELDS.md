# IRIS National Science Fair 2025–2026: Mandatory Form Fields

This document contains the exact text formatted for direct copy-paste into the IRIS National Science Fair application portal. Every field is verified against the official word count constraints.

---

### Field 1: Title
**Self-Correcting World Models Through Active Experimentation: Resolving Predictive Disagreement to Discover Causal Dynamics in Unfamiliar Environments**

---

### Field 2: Introduction & Objective (100–150 words)
> **Word Count:** 132 words (Constraint: 100–150 words)

Contemporary artificial intelligence systems excel at pattern recognition within familiar datasets, but fundamentally struggle with out-of-distribution reasoning in unfamiliar environments. Large language models passively ingest static text, while reinforcement learning agents rely on inefficient, undirected trial-and-error exploration requiring millions of environment steps. In contrast, human scientific reasoning forms competing mental explanations, simulates candidate actions internally, and performs targeted experiments specifically where those theories make conflicting predictions.

The primary objective of this research is to investigate whether an autonomous agent equipped with competing world-model hypotheses and counterfactual mental simulation can discover hidden physical dynamics faster than standard exploration heuristics. Specifically, this study evaluates whether choosing actions that maximize predictive disagreement reduces interaction sample complexity, enables autonomous recovery when environmental rules silently shift, and supports cross-version knowledge transfer without human supervision or oracle hints.

---

### Field 3: Innovation (50–100 words)
> **Word Count:** 82 words (Constraint: 50–100 words)

Rather than updating a monolithic black-box network via scalar reward, this work introduces a multi-hypothesis world model paired with counterfactual simulation. The core innovation is operationalizing epistemic curiosity as multi-entity predictive disagreement (Gini impurity over simulated outcomes): the agent autonomously performs the exact physical experiment where its competing theories disagree most. Furthermore, a four-tier hierarchical memory architecture isolates temporary run observations from distilled cross-version game rules and frozen submission snapshots, preventing catastrophic context overflow and test-time policy drift without privileged domain hints.

---

### Field 4: Methodology (150–250 words)
> **Word Count:** 215 words (Constraint: 150–250 words)

The system architecture integrates five modular, leak-free components:

1. **Perception:** Extracts spatial coordinates, object states, and geometric relations across all entities without oracle goal labels.
2. **Hypothesis Generation:** Initializes a normalized Dirichlet-multinomial belief distribution over candidate relational mechanisms (touch causes, pairwise object adjacencies, and state transitions).
3. **Counterfactual Simulator:** Computes internal forward rollouts for candidate actions under each competing hypothesis. It calculates action disagreement using multi-entity Gini impurity over predicted successor states:
   $$D(a) = \sum_{e} \left[1 - \sum_{s} P(s_e \mid a)^2\right]$$
4. **Active Experiment Selector:** Chooses the action maximizing epistemic disagreement while balancing verifiable process rewards ($R_{\text{proc}} = R_{\text{outcome}} + 2 N_{\text{verified}} - N_{\text{falsified}} - 0.01$).
5. **Diagnostic Engine:** Compares real step observations against simulated rollouts. Observed effects trigger Bayesian verification (boosting valid hypotheses), whereas discrepancies penalize false explanations and trigger autonomous model revision when all hypotheses fail.

The empirical benchmark evaluates five comparative exploration strategies across 25 procedural seeds in randomized grid topologies with hidden causal mechanics. Non-stationary adaptability is evaluated by inducing silent mid-run rule mutations at step 15. Finally, a four-tier hierarchical memory (run_memory, game_memory, global_memory, submission_memory) stores distilled, structured rules rather than raw step transcripts, enabling cross-run knowledge transfer ($v_1 \to v_4$) and frozen, deterministic competition deployment.

---

### Field 5: Results & Conclusions (100–150 words)
> **Word Count:** 133 words (Constraint: 100–150 words)

Across 25 procedural randomized seeds, predictive disagreement achieved a 52.0% discovery success rate (mean 36.5 ± 5.8 steps), outperforming undirected random exploration (16.0%, 45.3 ± 4.4 steps) and reactive greedy exploration (0.0%, 50.0 steps), which suffered total failure from local object trapping. Ablation experiments proved that collapsing beliefs into a single point estimate or removing prediction-error diagnosis reduced discovery to 4.0%, demonstrating that maintaining hypothesis uncertainty is mathematically required for discovery.

Under silent physical rule shifts, the agent detected 29.3 prediction discrepancies on average, achieving a 64.0% autonomous recovery rate without external reset. The four-tier memory successfully transferred verified rules across versions while preventing context overflow. In conclusion, active experimentation guided by predictive disagreement provides a principled, sample-efficient foundation for autonomous causal discovery in unfamiliar environments.

---

### Field 6: Acknowledgement & References (50–100 words)
> **Word Count:** 86 words (Constraint: 50–100 words)

I acknowledge my mentors and teachers for guidance in scientific methodology, and the open-source community for foundational computational tools.

Key References:
1. Ha, D., & Schmidhuber, J. (2018). Recurrent World Models Facilitate Policy Evolution. *NeurIPS*.
2. Settles, B. (2009). Active Learning Literature Survey. University of Wisconsin–Madison.
3. Friston, K. (2010). The Free-Energy Principle: A Unified Brain Theory? *Nature Reviews Neuroscience*.
4. Chollet, F. (2019). On the Measure of Intelligence. *arXiv:1911.01547*.
5. Lindley, D. V. (1956). On a Measure of the Information Provided by an Experiment. *Ann. Math. Statist.*

---

### Field 7: Abstract (250 words max)
> **Word Count:** 236 words (Constraint: ≤ 250 words)

Contemporary artificial intelligence systems excel within static training distributions but struggle with out-of-distribution reasoning in unfamiliar environments. Model-free reinforcement learning relies on inefficient, undirected exploration, while large language models cannot physically test their beliefs. In contrast, human scientific cognition formulates competing hypotheses, mentally simulates consequences, and deliberately conducts experiments where explanations conflict most. This study investigates whether an autonomous agent utilizing predictive disagreement between competing world models can achieve sample-efficient causal learning without human supervision or privileged hints.

We developed an autonomous research framework comprising: an all-entity perception module; a hypothesis manager maintaining normalized belief distributions over relational mechanics; a counterfactual mental simulator calculating multi-entity Gini disagreement; a diagnostic self-correction engine; and a four-tier hierarchical memory architecture isolating temporary run observations from distilled cross-version knowledge. 

Across 25 procedural randomized environments, predictive disagreement achieved a 52.0% causal discovery rate (36.5 ± 5.8 steps), more than tripling random exploration (16.0%, 45.3 ± 4.4 steps) and avoiding the catastrophic failure of greedy proximity heuristics (0.0%). Ablating hypothesis distributions to single point estimates collapsed success to 4.0%, proving that maintaining epistemic uncertainty is mathematically essential for discovery. Under unannounced physical rule mutations, the agent detected an average of 29.3 prediction errors and achieved a 64.0% autonomous adaptation rate. 

These results provide rigorous empirical evidence that active experimentation driven by internal predictive disagreement enables rapid causal discovery and robust self-correction, offering a foundational architecture for autonomous artificial general intelligence.
