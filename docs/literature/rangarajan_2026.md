# Literature Review — Rangarajan et al. (2026)

**Author:** Pelin Konak  
**Project:** Communication-Resilient Task Allocation for UAV–UGV Teams in Weed Management  
**Date:** November 24, 2026  
**Review status:** Abstract-based preliminary review

## 1. Publication Information

- Authors: Sricharan Rangarajan, Harshad Anandan, Dhinesh A M, Mogana Priya Chinnasamy
- Journal: Smart Agricultural Technology
- Volume: 13
- Article: 101842
- Year: 2026
- DOI: https://doi.org/10.1016/j.atech.2026.101842

## 2. Research Objective

The study evaluates reinforcement learning approaches for cooperative crop inspection under partial observability in a simplified agricultural environment.

## 3. Experimental Setup

| Parameter | Description |
|---|---|
| Framework | PettingZoo / Gymnasium |
| Environment | Single row containing 32 plants |
| Agents | Two homogeneous agents |
| Training | 5,000 episodes |
| Objective | Binary plant disease detection |
| Actions | Discrete movement and inspection |

The evaluated algorithms are PPO, DQN, MADDPG and QMIX.

## 4. Reward Function

| Event | Reward |
|---|---:|
| Correct detection | +5 |
| False positive | -0.1 |
| Repeated detection | -0.2 |
| Forward movement | +0.1 |
| Backward movement | -0.01 |

## 5. Main Findings

PPO achieved the strongest reported performance, with an episodic reward of approximately 0.7 and 5.27 correct detections per episode. False and repeated detections were close to zero.

These findings support investigating policy optimization methods in agricultural robotics.

## 6. Five Potential Research Differences

### 6.1. Heterogeneous Multi-Agent Architecture

Rangarajan et al. evaluate homogeneous agents.

Our project proposes heterogeneous UAV–UGV teams in which drones perform detection and ground robots execute treatment tasks.

This introduces role-specific observations, actions and coordination requirements.

### 6.2. Detection and Physical Intervention

The reviewed experimental task focuses on identifying diseased plants.

Our project extends the agricultural workflow to weed detection, task allocation and simulated targeted spraying.

This requires coordination between sensing and treatment agents.

### 6.3. Real Agricultural Dataset Integration

The reviewed study uses a simplified plant-row environment.

Our project plans to generate two-dimensional agricultural environments from real WeedMap annotations, using at least four 64×64 field plots.

This enables evaluation across different spatial weed distributions.

### 6.4. Communication Failure Modelling

Our project explicitly investigates communication failures as an experimental variable through:

- Bernoulli independent message loss.
- Gilbert-Elliott temporally correlated message loss.
- Matched average loss rates across communication models.

The primary objective is to investigate how different communication failure patterns affect task allocation and treatment performance.

The reviewed abstract does not specify equivalent controlled communication-loss experiments. This distinction requires full-text verification.

### 6.5. Learning-Based and Classical Task Allocation

The reviewed study compares four reinforcement learning algorithms.

Our project proposes a broader comparison between learning-based coordination and classical task allocation strategies:

- Fixed Route
- Sequential Auction
- Consensus-Based Bundle Algorithm (CBBA)
- Multi-Agent Proximal Policy Optimization (MAPPO)
- Single Robot reference baseline

This comparison will evaluate operational performance under different communication conditions and detection error rates.

## 7. Relevance to Our Project

The reviewed research provides a relevant foundation for investigating reinforcement learning in agricultural environments.

Its use of PettingZoo, partial observability and multi-objective rewards is aligned with several design choices in our project.

However, the reported performance of PPO does not establish that MAPPO will outperform classical task allocation methods in heterogeneous UAV–UGV environments.

This remains an open experimental question.

## 8. Proposed Research Question

How does MAPPO perform relative to classical task allocation methods in heterogeneous UAV–UGV weed management teams operating under independent and temporally correlated communication failures?

## 9. Expected Research Contributions

- Heterogeneous UAV–UGV coordination.
- WeedMap-based grid environments.
- Explicit independent and burst communication-loss models.
- Comparison between MAPPO and classical task allocation methods.
- Evaluation under communication failures and detection uncertainty.

These are proposed research contributions rather than experimentally validated findings.

## 10. Related Work Paragraph

Rangarajan et al. (2026) investigate reinforcement learning methods for cooperative crop inspection in a simplified row-cultivation environment. Their reported results favour PPO over the alternative evaluated methods. Building on this research direction, our study proposes heterogeneous aerial-ground teams, communication failure modelling and targeted weed treatment. The planned evaluation compares learning-based coordination with classical task allocation strategies using WeedMap-derived agricultural environments.

## 11. Review Limitations

This preliminary review is based on the accessible publication abstract and bibliographic records.

Detailed network architectures, experimental protocols, communication assumptions and statistical analyses require full-text verification.

The proposed differences should not be interpreted as confirmed claims of research novelty until the complete paper and related literature have been examined.