# Literature Review — Goeckner et al. (2024)

**Author:** Pelin Konak  
**Project:** Communication-Resilient Task Allocation for UAV–UGV Teams in Weed Management  
**Date:** November 24, 2026

## 1. Publication Information

- Authors: Anthony Goeckner, Yueyuan Sui, Nicolas Martinet, Xinliang Li, Qi Zhu
- Title: Graph Neural Network-based Multi-agent Reinforcement Learning for Resilient Distributed Coordination of Multi-Robot Systems
- Year: 2024
- Method: MAGEC
- Reference: arXiv:2403.13093
- DOI: https://doi.org/10.48550/arXiv.2403.13093
- Source: https://arxiv.org/html/2403.13093

## 2. Research Objective

The study introduces a learning-based coordination framework intended to maintain multi-robot performance despite incomplete environmental information, unreliable communication and robot failures.

## 3. Proposed Method

MAGEC combines:

- Graph Neural Networks (GNNs)
- Multi-Agent Proximal Policy Optimization (MAPPO)
- Centralized Training and Decentralized Execution (CTDE)
- Actor-critic learning
- Action masking

The environment is represented as a graph. A GraphSAGE-inspired actor processes node and edge information to select navigation actions. A centralized critic evaluates global system conditions during training.

## 4. Experimental Environment

The authors demonstrate their approach through multi-robot patrolling in a ROS 2 simulation environment.

Robots repeatedly visit graph nodes, with performance evaluated through node idleness.

The evaluation considers communication disturbances, agent attrition and generalization.

## 5. Main Findings

MAGEC surpassed existing coordination approaches in several disturbed scenarios while remaining competitive in normal operating conditions.

The results motivate further investigation of decentralized learning policies in unreliable robotic environments.

## 6. Relevance to Our Project

### 6.1. Communication Resilience

Our project similarly investigates coordination under unreliable communication.

However, our planned experiments focus on independent and temporally correlated message losses using Bernoulli and Gilbert-Elliott channel models.

### 6.2. Centralized Training and Decentralized Execution

Our planned MAPPO implementation adopts CTDE.

During training, the critic will have access to global simulation information. During execution, drones and ground robots will act using their respective local observations and successfully received messages.

### 6.3. Observation Design

Communication failures must affect the information available to decentralized agents.

Our environment design should therefore distinguish:

- Locally observed information.
- Successfully received messages.
- Missing or outdated information.
- Global information available exclusively during training.

### 6.4. Action Masking

Our heterogeneous environment requires role-specific action restrictions.

For example, the spraying action must be unavailable to drones and available to ground robots according to the environment rules.

### 6.5. Robustness Evaluation

The proposed agricultural experiments should compare the performance of trained policies across different communication conditions.

Possible evaluation questions include:

- Does increasing message loss reduce weed treatment effectiveness?
- Does temporally correlated loss cause greater degradation than independent loss?
- Can decentralized policies maintain useful behaviour during communication interruptions?
- Does MAPPO outperform classical task allocation methods under specific disturbance conditions?

## 7. Comparison with Our Project

| Dimension | MAGEC | Our proposed project |
|---|---|---|
| Application | Multi-robot patrolling | Agricultural weed management |
| Environment | Graph-based navigation | WeedMap-derived 2D grids |
| Learning architecture | GNN-enhanced MAPPO | EPyMARL-based MAPPO |
| Disturbance focus | Robot attrition and communication disturbances | Independent and burst message losses, detection errors |
| Agent responsibilities | Patrolling | Role-specific detection and treatment |
| Performance objective | Node revisit efficiency | Weed treatment effectiveness and operational efficiency |

## 8. Design Recommendations for Our Implementation

Based on the reviewed approach, the following design decisions are recommended for our project:

- Preserve strict separation between local observations and global state.
- Prevent communication failures from being bypassed through observation leakage.
- Apply role-specific action masks.
- Evaluate communication robustness across reproducible random seeds.
- Report performance degradation relative to a no-loss reference condition.
- Compare MAPPO against classical task allocation baselines.

A GNN-based policy could also be considered as a future extension. It is not required for the current project implementation.

## 9. Related Work Paragraph

Goeckner et al. (2024) combine graph-based representations with MAPPO to improve distributed robotic coordination under disturbances. Their patrolling experiments motivate the use of decentralized policies in environments with unreliable communication. Our proposed research investigates a related resilience problem in heterogeneous agricultural robots, with explicit communication-channel models and weed treatment objectives.

## 10. Conclusion

MAGEC provides methodological motivation for communication-resilient multi-agent learning.

Its results support investigating decentralized MAPPO policies in our agricultural simulation, but they do not establish that our proposed approach will achieve similar performance.

The effectiveness of our implementation must be demonstrated through independent agricultural experiments.

## 11. Review Limitations

This review uses the publicly available arXiv preprint. The numerical results have not been independently reproduced.

The proposed agricultural design recommendations are our own project-specific interpretations and should not be attributed directly to the authors.