"""
Neural Network Architecture for DQN

This module defines the Q-network architecture used by the DQN agent.
The network takes state observations as input and outputs Q-values for each action.
"""

import torch.nn as nn


class DQNNetwork(nn.Module):
    """
    Deep Q-Network (DQN) architecture.
    
    A feedforward neural network that maps states to Q-values for each action.
    Uses fully connected layers with ReLU activations.
    
    Args:
        input_dim: Dimension of input state vector
        output_dim: Number of actions (output Q-values)
        hidden: List of hidden layer sizes (default: [128, 128])
    
    Example:
        >>> net = DQNNetwork(state_dim=10, action_dim=4)
        >>> q_values = net(state_tensor)  # Shape: (batch_size, 4)
    """
    
    def __init__(self, input_dim, output_dim, hidden=[128, 128]):
        super().__init__()
        layers = []
        last = input_dim
        
        # Build hidden layers
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())  # ReLU activation for non-linearity
            last = h
        
        # Output layer: Q-values for each action (no activation)
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input state tensor of shape (batch_size, input_dim)
        
        Returns:
            Q-values for each action, shape (batch_size, output_dim)
        """
        return self.net(x)
