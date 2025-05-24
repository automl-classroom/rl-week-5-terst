# REINFORCE vs DQN: Empirical Analysis and RLiable Visualization

This document summarizes experiments comparing REINFORCE and DQN on CartPole-v1, varying trajectory length, network architecture, and learning rate. Results are visualized using RLiable for robust RL evaluation.

## Experimental Factors
- **Trajectory Length**: Shorter episodes may destabilize REINFORCE due to high variance; longer episodes can improve convergence but may slow learning.
- **Network Architecture**: Larger/deeper networks can improve expressiveness but may overfit or slow down training.
- **Learning Rate**: High learning rates speed up learning but risk instability; low rates are more stable but slower.

## Key Findings
- **Trajectory Length**: Longer trajectories generally improve REINFORCE's stability and convergence, but too long can slow down learning. DQN is less sensitive to this factor.
- **Network Architecture**: Both agents benefit from moderate hidden sizes (e.g., 128). Too small underfits; too large may overfit or slow learning.
- **Learning Rate**: REINFORCE is more sensitive to learning rate; DQN is more robust. Optimal rates are typically 1e-3 to 1e-2.
- **Sample Complexity**: DQN usually reaches high reward thresholds in fewer episodes (higher sample efficiency) than REINFORCE on CartPole-v1.

## RLiable Visualization and Results
- **Single Comparison Plot**: We aggregate all runs across all configs and plot a single mean learning curve for REINFORCE and DQN. This provides a clear, high-level comparison of the two algorithms.
- **Robust Metrics**: For each algorithm, we report the mean, median, IQM, and optimality gap (last 50 episodes, all configs/seeds) using RLiable's robust statistics.
- **Code Improvements**: The experiment and plotting scripts were updated to:
    - Remove unsupported arguments and methods (e.g., `hidden_size` for DQN, `observe`, `learn`).
    - Use correct RLiable API for robust metrics (e.g., `aggregate_iqm`, `aggregate_optimality_gap`).
    - Aggregate and plot results in a single comparison plot instead of per-config plots.
    - Use Matplotlib for learning curves, as RLiable does not provide a `plot_curves` function.

## Analysis of the Aggregate Learning Curve

The aggregate mean learning curve (see `reinforce_vs_dqn_mean_curve.png`) provides a direct comparison of REINFORCE and DQN across all hyperparameter configurations and seeds:

- **Initial Performance**: Both algorithms start at similar returns, but DQN's performance drops sharply after a few episodes, while REINFORCE steadily improves.
- **Learning Dynamics**: REINFORCE shows rapid and stable improvement, reaching a mean return of ~33–35 early and maintaining it. DQN, in contrast, suffers a long period of poor performance before recovering and only approaches REINFORCE's performance near the end of training.
- **Stability and Convergence**: REINFORCE converges faster and more stably in this experiment. DQN is less stable and more sensitive to hyperparameters, with a long plateau before improvement.
- **Sample Efficiency**: REINFORCE is more sample-efficient in this setting, achieving higher returns with fewer episodes. DQN requires more episodes to catch up.

**Conclusion:**
- In this experiment, REINFORCE outperforms DQN in both learning speed and final performance. DQN can eventually catch up, but is less sample-efficient and more sensitive to hyperparameters.
- For CartPole-v1 and similar environments, REINFORCE may be preferable for quick learning, while DQN may require more careful tuning to match or exceed policy gradient methods.

## Recommendations
- Use RLiable for robust RL evaluation—plain averages can be misleading due to high variance.
- Tune trajectory length and learning rate carefully for REINFORCE.
- Prefer DQN for environments with dense rewards and low stochasticity; REINFORCE may be preferable for environments with high stochasticity or where policy stochasticity is essential.

---

_See `reinforce_vs_dqn_experiments.py` and `plot_rliable_results.py` for code. The main comparison plot is saved as `rl_exercises/week_5/plots/reinforce_vs_dqn_mean_curve.png`._
