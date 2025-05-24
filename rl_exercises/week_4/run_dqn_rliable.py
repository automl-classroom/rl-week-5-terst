import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from rl_exercises.week_4.dqn import DQNAgent, set_seed
from rliable import library as rly
from rliable import metrics

ENV = "CartPole-v1"
SEEDS = [0, 1, 2, 3, 4]
NUM_FRAMES = 10_000

all_rewards = []
for seed in SEEDS:
    env = gym.make(ENV)
    set_seed(env, seed)
    agent = DQNAgent(env, seed=seed)
    state, _ = env.reset()
    ep_reward = 0
    rewards = []
    for _ in range(NUM_FRAMES):
        action = agent.predict_action(state)
        s2, r, done, trunc, _ = env.step(action)
        agent.buffer.add(state, action, r, s2, done or trunc, {})
        state = s2
        ep_reward += r
        if len(agent.buffer) >= agent.batch_size:
            batch = agent.buffer.sample(agent.batch_size)
            _ = agent.update_agent(batch)
        if done or trunc:
            state, _ = env.reset()
            rewards.append(ep_reward)
            ep_reward = 0
    all_rewards.append(rewards)

max_len = max(len(r) for r in all_rewards)
rewards_arr = np.vstack(
    [np.pad(r, (0, max_len - len(r)), mode="edge") for r in all_rewards]
)

frames = np.arange(1, max_len + 1)
iqm_mean = np.zeros(max_len)
iqm_lo = np.zeros(max_len)
iqm_hi = np.zeros(max_len)

for i in range(max_len):
    per_seed = rewards_arr[:, i]
    pe, ie = rly.get_interval_estimates(
        {"DQN": per_seed[:, None]},
        metrics.aggregate_iqm,
        reps=200,
    )
    iqm_mean[i] = pe["DQN"]
    low, high = ie["DQN"]
    iqm_lo[i], iqm_hi[i] = float(low), float(high)

final_rewards = rewards_arr[:, -1]
points, lowers, uppers = [], [], []
names = ["mean", "median", "iqm", "optimality_gap"]
funcs = [
    metrics.aggregate_mean,
    metrics.aggregate_median,
    metrics.aggregate_iqm,
    metrics.aggregate_optimality_gap,
]

for name, fn in zip(names, funcs):
    pe, ie = rly.get_interval_estimates(
        {"DQN": final_rewards[:, None]},
        fn,
        reps=200,
    )
    val = float(pe["DQN"])
    low, high = ie["DQN"]
    points.append(val)
    lowers.append(float(low))
    uppers.append(float(high))

points = np.array(points)
lowers = np.array(lowers)
uppers = np.array(uppers)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(frames, iqm_mean, label="IQM")
ax1.fill_between(frames, iqm_lo, iqm_hi, alpha=0.3)
ax1.set_xlabel("Episode")
ax1.set_ylabel("Reward")
ax1.set_title("IQM Training Curve (5 seeds)")
ax1.legend()

x = np.arange(len(names))

yerr = np.vstack([points - lowers, uppers - points])
ax2.bar(x, points, yerr=yerr, capsize=5)
ax2.set_xticks(x)
ax2.set_xticklabels(names)
ax2.set_ylabel("Reward")
ax2.set_title("Final-Episode Metrics (95% CI)")

plt.tight_layout()
plt.savefig("rl_exercises/week_4/plots/dqn_rliable_results.png")
plt.show()
