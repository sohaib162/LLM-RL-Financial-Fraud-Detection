import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FraudDetectionEnv(gym.Env):
    """
    A custom Gym environment for Fraud Detection using embeddings.

    State: Embedding of a transaction.
    Action: 0 (Declare Not Fraud), 1 (Declare Fraud).
    Reward: Based on correctly/incorrectly classifying fraud vs non-fraud.
    """
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray, reward_config: dict):
        super().__init__()

        # Ensure data consistency
        assert embeddings.shape[0] == labels.shape[0], "Embeddings and labels must have the same number of instances."
        assert embeddings.shape[1] == 768, f"Embeddings must be 768-dimensional, but got {embeddings.shape[1]}"

        self.embeddings = embeddings.astype(np.float32)
        self.labels = labels.astype(np.int64)

        self.num_instances = self.embeddings.shape[0]
        self.reward_config = reward_config

        # Define action and observation space
        # Action Space: Discrete(2) -> 0 for Not Fraud, 1 for Fraud
        self.action_space = spaces.Discrete(2)

        # Observation Space: Box(low, high, shape, dtype) -> 768-dim vector
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(768,), dtype=np.float32)

        # Internal state
        self._current_index = 0
        self._order = np.arange(self.num_instances)
        np.random.shuffle(self._order) # Shuffle the order of instances initially


    def step(self, action: int):
        # Check if episode is done
        if self._current_index >= self.num_instances:
            print("Warning: step() called when episode is already done.")
            return self.observation_space.sample() * 0, 0, True, False, {} # Return dummy values

        # Get current instance data based on shuffled order
        actual_index = self._order[self._current_index]
        current_embedding = self.embeddings[actual_index]
        true_label = self.labels[actual_index]

        # Determine reward
        reward = 0
        if action == 1 and true_label == 1:
            reward = self.reward_config.get('TP', 0)
        elif action == 1 and true_label == 0:
            reward = self.reward_config.get('FP', 0)
        elif action == 0 and true_label == 1:
            reward = self.reward_config.get('FN', 0)
        elif action == 0 and true_label == 0:
            reward = self.reward_config.get('TN', 0)

        # Move to the next instance
        self._current_index += 1

        # Check if the episode is finished
        done = self._current_index >= self.num_instances
        truncated = False

        # Get the next observation
        next_observation = np.zeros_like(current_embedding, dtype=np.float32) # Default for done state
        if not done:
             next_observation = self.embeddings[self._order[self._current_index]]

        info = {
            'true_label': true_label,
            'predicted_action': action,
            'instance_uid': actual_index,
            'is_done': done
        }

        return next_observation, reward, done, truncated, info


    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Handles seeding

        # Reset index and shuffle order for a new episode
        self._current_index = 0
        self._order = np.arange(self.num_instances)
        self.np_random.shuffle(self._order) # Use the environment's random number generator

        # Get the first observation of the new episode
        initial_observation = self.embeddings[self._order[self._current_index]]

        info = {'instance_uid': self._order[self._current_index]}

        return initial_observation, info

    def close(self):
        # Optional: Implement cleanup
        pass