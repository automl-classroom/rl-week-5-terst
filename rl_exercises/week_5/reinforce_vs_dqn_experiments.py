"""
Experiment script to compare REINFORCE and DQN on CartPole-v1.
- Varies trajectory length, network architecture, and learning rate for REINFORCE.
- Runs DQN with matching architectures and learning rates.
- Runs each config with multiple seeds.
- Saves per-episode returns for RLiable analysis.
"""

import os

import gymnasium as gym
import numpy as np
from rl_exercises.week_4.dqn import DQNAgent
from rl_exercises.week_5 import REINFORCEAgent

RESULTS_DIR = "rl_exercises/week_5/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

ENV_ID = "CartPole-v1"
SEEDS = [0, 1, 2, 3, 4]
EPISODES = 300
EVAL_INTERVAL = 10

# Experiment grid
TRAJ_LENGTHS = [25, 50]  # max episode steps
HIDDEN_SIZES = [64, 128]
LRS = [1e-2, 1e-3]


def run_reinforce(traj_len, hidden_size, lr, seed):
    env = gym.make(ENV_ID, max_episode_steps=traj_len)
    agent = REINFORCEAgent(env, lr=lr, hidden_size=hidden_size, seed=seed)
    returns = []
    for ep in range(EPISODES):
        state, _ = env.reset(seed=seed)
        done = False
        batch = []
        while not done:
            action, info = agent.predict_action(state)
            next_state, reward, term, trunc, _ = env.step(action)
            batch.append((state, action, reward, next_state, term or trunc, info))
            state = next_state
            done = term or trunc
        agent.update_agent(batch)
        total_return = sum([r for _, _, r, *_ in batch])
        returns.append(total_return)
    return returns


def run_dqn(traj_len, hidden_size, lr, seed):
    env = gym.make(ENV_ID, max_episode_steps=traj_len)
    agent = DQNAgent(env, lr=lr, seed=seed)
    returns = []
    for ep in range(EPISODES):
        state, _ = env.reset(seed=seed)
        done = False
        total_return = 0
        while not done:
            action = agent.predict_action(state)
            next_state, reward, term, trunc, _ = env.step(action)
            agent.buffer.add(state, action, reward, next_state, term or trunc, {})
            if len(agent.buffer) >= agent.batch_size:
                batch = agent.buffer.sample(agent.batch_size)
                agent.update_agent(batch)
            state = next_state
            total_return += reward
            done = term or trunc
        returns.append(total_return)
    return returns


def main():
    for traj_len in TRAJ_LENGTHS:
        for hidden_size in HIDDEN_SIZES:
            for lr in LRS:
                for seed in SEEDS:
                    # REINFORCE
                    r_returns = run_reinforce(traj_len, hidden_size, lr, seed)
                    fname = f"{RESULTS_DIR}/reinforce_traj{traj_len}_h{hidden_size}_lr{lr}_seed{seed}.npy"
                    np.save(fname, np.array(r_returns))
                    # DQN
                    d_returns = run_dqn(traj_len, hidden_size, lr, seed)
                    fname = f"{RESULTS_DIR}/dqn_traj{traj_len}_h{hidden_size}_lr{lr}_seed{seed}.npy"
                    np.save(fname, np.array(d_returns))


if __name__ == "__main__":
    main()
