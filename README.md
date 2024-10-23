# Reinforcement Learning Algorithms: MENACE and Multi-Armed Bandit

This repository contains the implementation of three reinforcement learning algorithms: MENACE (Matchbox Educable Noughts and Crosses Engine), Binary Bandit, and Non-Stationary Bandit. Each algorithm demonstrates different approaches to solving decision-making problems in uncertain environments.

## Table of Contents
- [MENACE](#menace)
- [Binary Bandit](#binary-bandit)
- [Non-Stationary Bandit](#non-stationary-bandit)
- [How to Run](#how-to-run)
- [References](#references)

## MENACE
MENACE is a reinforcement learning model that plays the game of Noughts and Crosses (Tic-Tac-Toe) by learning from its wins, losses, and draws. It uses matchboxes and beads as a physical simulation of a reinforcement learning agent.

![menance](https://github.com/user-attachments/assets/2216986e-63a8-4831-8587-852d0b638fcd)

### Algorithm
The MENACE algorithm is based on trial and error:
- Initially, the matchboxes (states) contain an equal number of beads (actions).
- After each game, the bead counts are updated depending on whether MENACE won, lost, or drew the game.
- Over time, MENACE improves by selecting the actions that led to winning states.

## Binary Bandit
The Binary Bandit problem is a simplified version of the multi-armed bandit, where there are two possible actions, each with a different probability of providing a reward. The goal is to find the action that maximizes the reward.

![binary bandit](https://github.com/user-attachments/assets/81c6983f-de63-4fae-9ce8-44c50e6ed91f)

### Algorithm
The Binary Bandit algorithm involves:
- Exploring both actions to estimate their reward probabilities.
- Exploiting the action that has the highest estimated probability of success.
- Balancing exploration and exploitation to maximize total rewards.

## Non-Stationary Bandit
The Non-Stationary Bandit problem is a variation of the traditional multi-armed bandit where the reward probabilities of the actions change over time. This introduces the challenge of adapting to the changing environment.

![non stationary](https://github.com/user-attachments/assets/bd649f8e-445c-440b-ac65-2f0b46f8a954)

### Algorithm
To handle non-stationarity, the following strategies can be used:
- Use a decaying learning rate to give more weight to recent actions.
- Implement strategies like "Upper Confidence Bound" (UCB) or "ε-greedy" with a dynamic ε to adapt to the changing rewards.
- Continuously re-evaluate the estimated reward probabilities as more data is gathered.

## How to Run
1. Clone this repository:
    ```bash
    git clone https://github.com/nitinkoberoii/AI_Lab7_Nexus.git
    cd AI_Lab7_Nexus
    ```
2. Run the MENACE implementation:
    ```bash
    python menace-implementation.py
    ```
3. Run the Binary Bandit implementation:
    ```bash
    python binary-bandit.py
    ```
4. Run the Non-Stationary Bandit implementation:
    ```bash
    python non-stationary-bandit.py
    ```

## References
[(https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)]

