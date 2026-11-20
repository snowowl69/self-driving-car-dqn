"""
Deep Q-Network (DQN) Agent Implementation

This module implements a DQN agent with the following features:
- Experience replay for stable learning
- Target network for stable Q-value estimation
- Epsilon-greedy action selection
- Periodic target network updates
"""

import random
import torch
import torch.nn.functional as F
from model import DQNNetwork


class DQNAgent:
    """
    DQN Agent for reinforcement learning.
    
    Uses a policy network to estimate Q-values and a target network for
    stable learning. Implements experience replay to break correlation
    between consecutive samples.
    
    Args:
        state_dim: Dimension of the state space
        action_dim: Number of possible actions
        device: Device to run on ('cpu' or 'cuda')
        lr: Learning rate for optimizer (default: 1e-3)
        gamma: Discount factor for future rewards (default: 0.99)
        batch_size: Batch size for training (default: 64)
        buffer_size: Size of experience replay buffer (default: 50000)
        target_update: Steps between target network updates (default: 1000)
    """
    
    def __init__(self, state_dim, action_dim, device='cpu', lr=1e-3, gamma=0.99, 
                 batch_size=64, buffer_size=50000, target_update=1000):
        self.device = torch.device(device)
        self.action_dim = action_dim
        self.gamma = gamma  # Discount factor
        self.batch_size = batch_size
        self.target_update = target_update  # Frequency of target network updates
        
        # Policy network: used for action selection and training
        self.policy_net = DQNNetwork(state_dim, action_dim).to(self.device)
        
        # Target network: used for stable Q-value estimation
        self.target_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network always in eval mode
        
        # Optimizer for policy network
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Experience replay buffer
        from replay_buffer import ReplayBuffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Track total steps for epsilon decay
        self.steps_done = 0
    
    def select_action(self, state, eps=0.05):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state (numpy array)
            eps: Epsilon value for exploration (probability of random action)
        
        Returns:
            Selected action (integer)
        """
        # Exploration: random action
        if random.random() < eps:
            return random.randrange(self.action_dim)
        
        # Exploitation: select best action according to policy network
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(s)
            return int(q_values.argmax().item())
    
    def store(self, s, a, r, ns, done):
        """
        Store a transition in the experience replay buffer.
        
        Args:
            s: Current state
            a: Action taken
            r: Reward received
            ns: Next state
            done: Whether episode terminated
        """
        self.replay_buffer.add(s, a, r, ns, done)
    
    def optimize(self):
        """
        Perform one step of Q-learning optimization.
        
        Samples a batch from replay buffer and updates the policy network
        using the Bellman equation. Updates target network periodically.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        # Need enough samples in buffer before training
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample a batch of experiences
        import numpy as np
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        # Compute Q-values for taken actions
        q_values = self.policy_net(states).gather(1, actions)
        
        # Compute target Q-values using target network
        # Use max Q-value of next state (off-policy)
        next_q_values = self.target_net(next_states).max(1)[0].detach().unsqueeze(1)
        
        # Bellman equation: Q(s,a) = r + γ * max Q(s',a') * (1 - done)
        expected_q_values = rewards + (1.0 - dones) * self.gamma * next_q_values
        
        # Compute loss and update policy network
        loss = F.mse_loss(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
