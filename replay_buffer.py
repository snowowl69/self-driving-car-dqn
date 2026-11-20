"""
Experience Replay Buffer for DQN

This module implements an experience replay buffer that stores agent transitions
and allows random sampling for training. This breaks the correlation between
consecutive samples and improves learning stability.
"""

import random
from collections import deque
import numpy as np


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.
    
    Stores transitions (state, action, reward, next_state, done) in a
    circular buffer. When capacity is reached, oldest transitions are
    automatically removed (FIFO).
    
    Args:
        capacity: Maximum number of transitions to store
    
    Example:
        >>> buffer = ReplayBuffer(capacity=10000)
        >>> buffer.add(state, action, reward, next_state, done)
        >>> states, actions, rewards, next_states, dones = buffer.sample(32)
    """
    
    def __init__(self, capacity):
        self.capacity = capacity
        # Use deque with maxlen for automatic FIFO behavior
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state after action
            done: Whether episode terminated
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample a random batch of transitions from the buffer.
        
        Args:
            batch_size: Number of transitions to sample
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
            All returned as numpy arrays suitable for PyTorch tensors
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Cannot sample {batch_size} transitions from buffer of size {len(self.buffer)}")
        
        # Randomly sample batch_size transitions
        batch = random.sample(self.buffer, batch_size)
        
        # Unzip the batch into separate arrays
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to numpy arrays
        return (
            np.vstack(states),              # Stack states into (batch_size, state_dim)
            np.array(actions),              # (batch_size,)
            np.array(rewards, dtype=np.float32),  # (batch_size,)
            np.vstack(next_states),         # (batch_size, state_dim)
            np.array(dones, dtype=np.uint8) # (batch_size,)
        )
    
    def __len__(self):
        """Return the current number of transitions in the buffer."""
        return len(self.buffer)
