import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from game import UNKNOWN
from learning.config import DQNConfig
from learning.model import DQNNetwork
from learning.replay import ReplayMemory


class DQNAgent:
    def __init__(self, config=None):
        self.config = config or DQNConfig()
        self.device = torch.device(self.config.device)
        self.policy_net = DQNNetwork(
            height=self.config.height,
            width=self.config.width,
        ).to(self.device)
        self.target_net = DQNNetwork(
            height=self.config.height,
            width=self.config.width,
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=self.config.learning_rate,
        )
        self.memory = ReplayMemory(self.config.replay_capacity)
        self.steps_done = 0

    @property
    def action_size(self):
        return self.config.height * self.config.width

    def select_action(self, state, valid_actions, training=True):
        if training and random.random() < self.epsilon:
            return self._sample_valid_action(valid_actions)

        return self.select_best_action(state, valid_actions)

    def select_best_action(self, state, valid_actions):
        was_training = self.policy_net.training
        self.policy_net.eval()
        with torch.no_grad():
            state_tensor = self.encode_states([state], self.device)
            q_values = self.policy_net(state_tensor).squeeze(0)
            valid_mask = torch.as_tensor(
                valid_actions,
                dtype=torch.bool,
                device=self.device,
            )
            q_values = q_values.masked_fill(~valid_mask, -1e9)
            action = int(torch.argmax(q_values).item())
        if was_training:
            self.policy_net.train()
        return action

    def select_best_move(self, state, valid_actions=None):
        if valid_actions is None:
            valid_actions = self.valid_actions_from_state(state)
        return self.action_to_coords(self.select_best_action(state, valid_actions))

    def action_to_coords(self, action):
        action = int(action)
        y = action // self.config.width
        x = action % self.config.width
        return x, y

    def coords_to_action(self, x, y):
        return int(y) * self.config.width + int(x)

    def remember(self, transition):
        self.memory.push(transition)

    def learn(self):
        min_size = max(self.config.min_replay_size, self.config.batch_size)
        if len(self.memory) < min_size:
            return None

        batch = self.memory.sample(self.config.batch_size)
        states = self.encode_states(
            [transition.state for transition in batch],
            self.device,
        )
        next_states = self.encode_states(
            [transition.next_state for transition in batch],
            self.device,
        )
        actions = torch.as_tensor(
            [transition.action for transition in batch],
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(1)
        rewards = torch.as_tensor(
            [transition.reward for transition in batch],
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.as_tensor(
            [transition.done for transition in batch],
            dtype=torch.bool,
            device=self.device,
        )
        next_valid_actions = torch.as_tensor(
            np.stack([transition.next_valid_actions for transition in batch]),
            dtype=torch.bool,
            device=self.device,
        )

        q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            next_q_values = next_q_values.masked_fill(~next_valid_actions, -1e9)
            next_q_values = next_q_values.max(dim=1).values
            next_q_values = torch.where(
                dones,
                torch.zeros_like(next_q_values),
                next_q_values,
            )
            expected_q_values = rewards + (self.config.gamma * next_q_values)

        loss = F.smooth_l1_loss(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.config.target_update_every == 0:
            self.update_target_network()

        return float(loss.item())

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                'config': self.config.__dict__,
                'policy_net': self.policy_net.state_dict(),
                'target_net': self.target_net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'steps_done': self.steps_done,
            },
            path,
        )

    @classmethod
    def load(cls, path, device=None):
        checkpoint = torch.load(path, map_location=device or 'cpu')
        config = DQNConfig(**checkpoint['config'])
        if device is not None:
            config.device = device

        agent = cls(config)
        agent.policy_net.load_state_dict(checkpoint['policy_net'])
        agent.target_net.load_state_dict(checkpoint['target_net'])
        agent.optimizer.load_state_dict(checkpoint['optimizer'])
        agent.steps_done = checkpoint.get('steps_done', 0)
        agent._move_optimizer_state_to_device()
        return agent

    def update_training_config(
        self,
        learning_rate=None,
        gamma=None,
        batch_size=None,
        min_replay_size=None,
        max_steps_per_episode=None,
    ):
        if learning_rate is not None:
            self.config.learning_rate = learning_rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate
        if gamma is not None:
            self.config.gamma = gamma
        if batch_size is not None:
            self.config.batch_size = batch_size
        if min_replay_size is not None:
            self.config.min_replay_size = min_replay_size
        if max_steps_per_episode is not None:
            self.config.max_steps_per_episode = max_steps_per_episode

    def _move_optimizer_state_to_device(self):
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)

    @property
    def epsilon(self):
        progress = min(1.0, self.steps_done / self.config.epsilon_decay_steps)
        return (
            self.config.epsilon_start
            + progress * (self.config.epsilon_end - self.config.epsilon_start)
        )

    def _sample_valid_action(self, valid_actions):
        valid_indices = np.flatnonzero(valid_actions)
        if len(valid_indices) == 0:
            raise ValueError('No valid actions are available.')
        return int(random.choice(valid_indices))

    @staticmethod
    def encode_states(states, device):
        states_array = np.asarray(states, dtype=np.int64)
        if states_array.ndim == 2:
            states_array = states_array[np.newaxis, ...]

        states_tensor = torch.as_tensor(
            states_array,
            dtype=torch.long,
            device=device,
        )
        batch_size, height, width = states_tensor.shape
        encoded = torch.zeros(
            (batch_size, 10, height, width),
            dtype=torch.float32,
            device=device,
        )
        encoded[:, 0] = states_tensor == UNKNOWN

        for value in range(9):
            encoded[:, value + 1] = states_tensor == value

        return encoded

    @staticmethod
    def valid_actions_from_state(state):
        return (np.asarray(state).reshape(-1) == UNKNOWN).copy()
