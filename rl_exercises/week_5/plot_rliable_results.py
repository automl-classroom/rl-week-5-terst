"""
Plotting script using RLiable to compare REINFORCE and DQN results.
- Aggregates results from experiment script.
- Plots IQM, median, mean, and optimality gap for each config.
- Saves plots to 'rl_exercises/week_5/plots/'.
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import rliable.metrics as metrics

RESULTS_DIR = "rl_exercises/week_5/results"
PLOTS_DIR = "rl_exercises/week_5/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load all results
files = glob.glob(f"{RESULTS_DIR}/*.npy")

# Group by agent and config
results = {}
for f in files:
    fname = os.path.basename(f)
    agent = "reinforce" if fname.startswith("reinforce") else "dqn"
    key = fname.replace(f"{agent}_", "").replace(".npy", "")
    if (agent, key) not in results:
        results[(agent, key)] = []
    results[(agent, key)].append(np.load(f))

# For each config, plot IQM, median, mean, optimality gap
for (agent, key), runs in results.items():
    runs = np.stack(runs)  # shape: (seeds, episodes)
    # Compute metrics
    scores = runs[:, -50:]  # last 50 episodes
    metrics_dict = {
        "mean": np.mean(scores),
        "median": metrics.aggregate_median(scores),
        "iqm": metrics.aggregate_iqm(scores),
        "optimality_gap": metrics.aggregate_optimality_gap(scores, 500.0),
    }
    # Plot training curves
    plt.figure(figsize=(8, 4))
    for i, run in enumerate(runs):
        plt.plot(
            np.arange(len(run)),
            run,
            alpha=0.3,
            label=f"Seed {i + 1}" if i == 0 else None,
            color="C0" if agent == "reinforce" else "C1",
        )
    plt.plot(
        np.arange(runs.shape[1]),
        runs.mean(axis=0),
        label="Mean",
        color="k",
        linewidth=2,
    )
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(f"{agent.upper()} {key}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{agent}_{key}_curve.png")
    plt.close()
    # Print metrics
    print(f"{agent.upper()} {key}: {metrics_dict}")

# Aggregate all REINFORCE and all DQN runs (across all configs)
all_reinforce = []
all_dqn = []
for (agent, key), runs in results.items():
    runs = np.stack(runs)
    if agent == "reinforce":
        all_reinforce.append(runs)
    else:
        all_dqn.append(runs)

# Concatenate across all configs and seeds
all_reinforce = np.concatenate(all_reinforce, axis=0) if all_reinforce else np.array([])
all_dqn = np.concatenate(all_dqn, axis=0) if all_dqn else np.array([])

# Plot mean learning curve for each algorithm
plt.figure(figsize=(10, 5))
if all_reinforce.size > 0:
    plt.plot(
        np.arange(all_reinforce.shape[1]),
        all_reinforce.mean(axis=0),
        label="REINFORCE",
        color="C0",
        linewidth=2,
    )
if all_dqn.size > 0:
    plt.plot(
        np.arange(all_dqn.shape[1]),
        all_dqn.mean(axis=0),
        label="DQN",
        color="C1",
        linewidth=2,
    )
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("REINFORCE vs DQN: Mean Learning Curve (All Configs)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/reinforce_vs_dqn_mean_curve.png")
plt.close()

# Print overall robust metrics for each algorithm
if all_reinforce.size > 0:
    scores = all_reinforce[:, -50:]
    metrics_dict = {
        "mean": np.mean(scores),
        "median": metrics.aggregate_median(scores),
        "iqm": metrics.aggregate_iqm(scores),
        "optimality_gap": metrics.aggregate_optimality_gap(scores, 500.0),
    }
    print(f"REINFORCE (all configs): {metrics_dict}")
if all_dqn.size > 0:
    scores = all_dqn[:, -50:]
    metrics_dict = {
        "mean": np.mean(scores),
        "median": metrics.aggregate_median(scores),
        "iqm": metrics.aggregate_iqm(scores),
        "optimality_gap": metrics.aggregate_optimality_gap(scores, 500.0),
    }
    print(f"DQN (all configs): {metrics_dict}")
