import argparse, os, math
import matplotlib.pyplot as plt
import torch
import json
import numpy as np

from driving_env import DrivingEnv
from dqn_agent import DQNAgent

def train(args):
    """Train a DQN agent on the driving environment."""
    print("=" * 60)
    print("Starting Training")
    print("=" * 60)
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Episodes: {args.episodes}")
    print(f"Road type: {'Curved' if args.curved else 'Straight'}")
    print(f"Lane width: {'Narrow' if args.narrow else 'Normal'}")
    print("=" * 60)
    
    env = DrivingEnv(render=False, curved=args.curved, narrow=args.narrow)
    state_dim = env.state_size
    action_dim = env.action_space
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = DQNAgent(state_dim, action_dim, device=device, lr=args.lr, gamma=args.gamma,
                     batch_size=args.batch_size, buffer_size=args.buffer_size,
                     target_update=args.target_update)
    
    eps_start = 1.0
    eps_end = 0.05
    eps_decay = args.eps_decay
    
    # Training statistics
    rewards_history = []
    episode_lengths = []
    success_history = []  # Track if episode completed successfully
    best_avg = -1e9
    best_episode = 0
    
    for ep in range(1, args.episodes + 1):
        state = env.reset()
        total_reward = 0.0
        done = False
        steps = 0
        while not done and steps < args.max_steps:
            eps = eps_end + (eps_start - eps_end) * math.exp(-1. * agent.steps_done / eps_decay)
            action = agent.select_action(state, eps=eps)
            next_state, reward, done, _ = env.step(action)
            agent.store(state, action, reward, next_state, done)
            _ = agent.optimize()
            state = next_state
            total_reward += reward
            steps += 1
        
        # Record statistics
        rewards_history.append(total_reward)
        episode_lengths.append(steps)
        success_history.append(1 if not done else 0)  # Success if reached max steps
        
        if ep % args.log_interval == 0:
            avg_reward = sum(rewards_history[-args.log_interval:]) / args.log_interval
            avg_length = sum(episode_lengths[-args.log_interval:]) / args.log_interval
            success_rate = sum(success_history[-args.log_interval:]) / args.log_interval * 100
            max_reward = max(rewards_history[-args.log_interval:])
            
            print(f"Episode {ep:4d} | "
                  f"Avg Reward: {avg_reward:7.2f} | "
                  f"Max Reward: {max_reward:7.2f} | "
                  f"Avg Length: {avg_length:6.1f} | "
                  f"Success Rate: {success_rate:5.1f}% | "
                  f"Epsilon: {eps:.3f}")
            
            if avg_reward > best_avg:
                best_avg = avg_reward
                best_episode = ep
                os.makedirs('results', exist_ok=True)
                torch.save(agent.policy_net.state_dict(), 'results/model.pth')
    
    # Final statistics
    final_stats = {
        'total_episodes': args.episodes,
        'final_avg_reward': np.mean(rewards_history[-50:]) if len(rewards_history) >= 50 else np.mean(rewards_history),
        'best_avg_reward': best_avg,
        'best_episode': best_episode,
        'final_success_rate': np.mean(success_history[-50:]) * 100 if len(success_history) >= 50 else np.mean(success_history) * 100,
        'overall_success_rate': np.mean(success_history) * 100,
        'avg_episode_length': np.mean(episode_lengths),
        'max_reward': max(rewards_history),
        'min_reward': min(rewards_history),
        'road_type': 'curved' if args.curved else 'straight',
        'lane_width': 'narrow' if args.narrow else 'normal'
    }
    
    # Save statistics
    os.makedirs('results', exist_ok=True)
    with open('results/training_stats.json', 'w') as f:
        json.dump(final_stats, f, indent=2)
    
    # Enhanced plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Rewards over time
    axes[0, 0].plot(rewards_history, alpha=0.3, color='blue', label='Episode Reward')
    # Moving average
    window = min(50, len(rewards_history) // 10)
    if window > 1:
        moving_avg = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(rewards_history)), moving_avg, 
                       color='red', linewidth=2, label=f'Moving Avg ({window})')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Training Rewards Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Episode lengths
    axes[0, 1].plot(episode_lengths, alpha=0.3, color='green')
    if window > 1:
        moving_avg_len = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(range(window-1, len(episode_lengths)), moving_avg_len, 
                       color='darkgreen', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Episode Length')
    axes[0, 1].set_title('Episode Lengths Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Success rate over time
    success_window = min(50, len(success_history) // 10)
    if success_window > 1:
        success_avg = np.convolve(success_history, np.ones(success_window)/success_window, mode='valid')
        axes[1, 0].plot(range(success_window-1, len(success_history)), 
                       [s * 100 for s in success_avg], color='purple', linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Success Rate (%)')
    axes[1, 0].set_title('Success Rate Over Time')
    axes[1, 0].set_ylim([0, 105])
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Reward distribution
    axes[1, 1].hist(rewards_history, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(np.mean(rewards_history), color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {np.mean(rewards_history):.2f}')
    axes[1, 1].set_xlabel('Total Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Reward Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/training_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also save simple rewards plot for compatibility
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_history, alpha=0.3, color='blue')
    if window > 1:
        plt.plot(range(window-1, len(rewards_history)), moving_avg, 
                color='red', linewidth=2, label=f'Moving Average ({window})')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/rewards_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best average reward: {best_avg:.2f} (Episode {best_episode})")
    print(f"Final average reward (last 50): {final_stats['final_avg_reward']:.2f}")
    print(f"Overall success rate: {final_stats['overall_success_rate']:.1f}%")
    print(f"Final success rate (last 50): {final_stats['final_success_rate']:.1f}%")
    print(f"Average episode length: {final_stats['avg_episode_length']:.1f} steps")
    print(f"Max reward achieved: {final_stats['max_reward']:.2f}")
    print("=" * 60)
    print("Results saved to:")
    print("  - results/model.pth (best model)")
    print("  - results/training_analysis.png (comprehensive plots)")
    print("  - results/rewards_plot.png (simple rewards plot)")
    print("  - results/training_stats.json (statistics)")
    print("=" * 60)        
    if args.curved:
        start_x = 50.0
        target_x = env.width - 60.0
        lane_distance = max(0.0, target_x - start_x)
        last_duration = episode_lengths[-1] if episode_lengths else 0
        last_reward = rewards_history[-1] if rewards_history else 0.0
        reached = bool(success_history[-1]) if success_history else False
        outcome_label = 'reward' if reached else 'penalty'
        status = 'reached destination' if reached else 'destination missed'
        print(f"[Curved road outcome] Distance ~ {lane_distance:.1f}px | Episodes: {args.episodes} | Duration: {last_duration} steps | {status} with {outcome_label}: {last_reward:.2f}")

def evaluate(args):
    """Evaluate a trained DQN agent on the driving environment."""
    print("=" * 60)
    print("Evaluation Mode")
    print("=" * 60)
    
    env = DrivingEnv(render=True, curved=args.curved, narrow=args.narrow)
    state_dim = env.state_size
    action_dim = env.action_space
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = DQNAgent(state_dim, action_dim, device=device)
    
    if not args.model_path or not os.path.exists(args.model_path):
        raise FileNotFoundError(f'Model path not found: {args.model_path}')
    
    print(f"Loading model from: {args.model_path}")
    agent.policy_net.load_state_dict(torch.load(args.model_path, map_location=device))
    agent.policy_net.eval()
    
    print(f"Road type: {'Curved' if args.curved else 'Straight'}")
    print(f"Lane width: {'Narrow' if args.narrow else 'Normal'}")
    print("=" * 60)
    print("Starting evaluation... (Close window or press ESC to exit)")
    print("=" * 60)
    
    # Run multiple episodes for statistics
    num_episodes = args.eval_episodes
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        min_distance_from_center = float('inf')
        max_distance_from_center = 0.0
        
        while not done and steps < args.max_steps:
            action = agent.select_action(state, eps=0.0)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            steps += 1
            
            # Track distance from center (from state)
            if len(state) > 4:
                dist_center = abs(state[4]) * env.lane_half
                min_distance_from_center = min(min_distance_from_center, dist_center)
                max_distance_from_center = max(max_distance_from_center, dist_center)
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        if not done:  # Successfully completed episode
            success_count += 1
        
        if num_episodes > 1:
            print(f"Episode {ep+1}/{num_episodes}: Reward={total_reward:.2f}, "
                  f"Length={steps}, Success={'Yes' if not done else 'No'}")
    
    # Print evaluation summary
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Episodes run: {num_episodes}")
    print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Best reward: {max(episode_rewards):.2f}")
    print(f"Worst reward: {min(episode_rewards):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Success rate: {success_count}/{num_episodes} ({success_count/num_episodes*100:.1f}%)")
    print("=" * 60)
    
    # Save evaluation results
    eval_stats = {
        'num_episodes': num_episodes,
        'avg_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'best_reward': float(max(episode_rewards)),
        'worst_reward': float(min(episode_rewards)),
        'avg_length': float(np.mean(episode_lengths)),
        'std_length': float(np.std(episode_lengths)),
        'success_rate': float(success_count / num_episodes * 100),
        'road_type': 'curved' if args.curved else 'straight',
        'lane_width': 'narrow' if args.narrow else 'normal'
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/evaluation_stats.json', 'w') as f:
        json.dump(eval_stats, f, indent=2)
    
    print("Evaluation statistics saved to: results/evaluation_stats.json")
    print("=" * 60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--model-path', type=str, default='results/model.pth')
    parser.add_argument('--episodes', type=int, default=400)
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--buffer-size', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--target-update', type=int, default=1000)
    parser.add_argument('--eps-decay', type=float, default=3000)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--eval-episodes', type=int, default=1, 
                       help="Number of episodes to run during evaluation")

    # Environment flags
    parser.add_argument('--curved', action='store_true', 
                       help="Use curved (sine-wave) road instead of straight lane")
    parser.add_argument('--narrow', action='store_true', 
                       help="Make the lane narrower for harder difficulty")

    args = parser.parse_args()
    
    if args.train:
        train(args)
    elif args.eval:
        evaluate(args)
    else:
        print('Run with --train to train the agent or --eval to evaluate a saved model.')
