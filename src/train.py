import numpy as np
import matplotlib.pyplot as plt

def train(agent, env, episodes, update_target_every, moving_avg_window=100, verbose=True):
    """
    Train the agent and visualize training progress.
    """
    rewards = []
    path_lengths = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.buffer.add(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        path_lengths.append(steps)

        if episode % update_target_every == 0:
            agent.update_params()

        # Decay epsilon
        agent.eps = max(agent.eps * agent.eps_decay, agent.eps_min)

        if verbose and episode % 100 == 0:
            moving_avg_reward = np.mean(rewards[-moving_avg_window:]) if len(rewards) >= moving_avg_window else np.mean(rewards)
            moving_avg_holding = np.mean(path_lengths[-moving_avg_window:]) if len(path_lengths) >= moving_avg_window else np.mean(path_lengths)
            print(
                f"Episode {episode}/{episodes}, "
                f"Total Reward: {total_reward:.2f}, "
                f"Moving Avg Reward: {moving_avg_reward:.2f}, "
                f"Moving Avg Holding: {moving_avg_holding:.2f}, "
                f"Epsilon: {agent.eps:.4f}"
            )

    if len(rewards) >= moving_avg_window:
        moving_avg_rewards = np.convolve(rewards, np.ones(moving_avg_window) / moving_avg_window, mode='valid')
        moving_avg_holding = np.convolve(path_lengths, np.ones(moving_avg_window) / moving_avg_window, mode='valid')
    else:
        moving_avg_rewards = rewards
        moving_avg_holding = path_lengths

    # Plot Total Rewards
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label="Total Reward per Episode", alpha=0.4)
    if len(rewards) >= moving_avg_window:
        plt.plot(
            range(moving_avg_window - 1, len(rewards)),
            moving_avg_rewards,
            label=f"Moving Avg Reward (window={moving_avg_window})",
            color='red'
        )
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress: Rewards")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Holding Times
    plt.figure(figsize=(12, 6))
    plt.plot(path_lengths, label="Holding Time per Episode", alpha=0.4)
    if len(path_lengths) >= moving_avg_window:
        plt.plot(
            range(moving_avg_window - 1, len(path_lengths)),
            moving_avg_holding,
            label=f"Moving Avg Holding Time (window={moving_avg_window})",
            color='blue'
        )
    plt.xlabel("Episode")
    plt.ylabel("Holding Time (Steps)")
    plt.title("Training Progress: Holding Times")
    plt.legend()
    plt.grid(True)
    plt.show()

    return rewards, path_lengths