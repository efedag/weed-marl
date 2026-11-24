# MAPPO Environment Integration Requirements

**Author:** Pelin Konak  
**Date:** 24 November 2026  
**Status:** Initial Technical Specification

## 1. Environment Configuration

| Parameter | Current | Proposed |
|---|---|---|
| Number of agents | 2 | Configurable |
| Grid size | 8×8 | 16×16 initial training |
| Number of actions | 5 | 6 |
| Observation size | 2 | To be finalized |
| State size | Not defined | Fixed-size global state |
| Episode limit | 50 | Configurable |

## 2. Action Space

| ID | Action | Drone | Ground Robot |
|---|---|---|---|
| 0 | STAY | Yes | Yes |
| 1 | UP | Yes | Yes |
| 2 | DOWN | Yes | Yes |
| 3 | LEFT | Yes | Yes |
| 4 | RIGHT | Yes | Yes |
| 5 | SPRAY | No | Yes |

Drone action mask: `[1, 1, 1, 1, 1, 0]`

Ground robot action mask: `[1, 1, 1, 1, 1, 1]`

These are initial static masks. Dynamic restrictions, such as battery limitations, may require additional masking logic.

## 3. Observation Space

### Drone

- Current row and column
- Remaining battery level
- Locally detected weed information
- Detection confidence
- Locally available communication information

### Ground Robot

- Current row and column
- Remaining battery level
- Received target coordinates
- Locally available weed information
- Received message status

Observations must have fixed dimensions throughout an episode. Different observation sizes can be handled through zero padding.

Missing information should be represented explicitly using validity indicators.

Agents must not access information from messages that were lost during transmission.

## 4. Global State

MAPPO follows Centralized Training with Decentralized Execution (CTDE).

The proposed centralized critic state may include:

- Agent positions
- Agent battery levels
- Ground-truth weed distribution
- Sprayed cell information
- Communication channel state
- Current episode progress

Global information must not be directly exposed to decentralized agent policies.

The default GymmaWrapper concatenates agent observations. A separate global state implementation is required to expose additional centralized information.

## 5. Reward Function

Initial cooperative reward:

`R = 10 × W - 5 × U - 0.1`

Where:

- W: Number of successfully sprayed weeds
- U: Number of unnecessary sprays
- -0.1: Step penalty

Reward coefficients are configurable and subject to experimental tuning.

The team reward must be counted once per environment step to avoid duplicate rewards during aggregation.

## 6. Environment Interface

Required methods:

- reset()
- step()
- get_obs()
- get_obs_agent()
- get_obs_size()
- get_state()
- get_state_size()
- get_avail_actions()
- get_avail_agent_actions()
- get_total_actions()
- get_env_info()
- close()

The environment must preserve consistent agent ordering and correctly distinguish termination from time-limit truncation.

## 7. Communication Constraints

The environment must support:

- Bernoulli communication loss
- Gilbert-Elliott communication loss
- Successful and failed message transmission
- Configurable communication parameters

Communication loss must affect the information available to decentralized policies.

## 8. Initial Integration Milestone

- Connect the PettingZoo environment to EPyMARL.
- Validate observation and action dimensions.
- Validate action masks and reward aggregation.
- Verify episode termination and truncation.
- Run initial MAPPO training on a 16×16 grid.

## 9. Pending Decisions

- Final observation vector dimensions
- Global state representation and dimensions
- Detection range
- Battery consumption model
- Message representation
- Episode termination conditions

These decisions will be finalized collaboratively with the simulation developer.