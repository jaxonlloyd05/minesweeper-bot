from dataclasses import dataclass


@dataclass
class DQNConfig:
    height: int = 16
    width: int = 30
    mines: int = 99
    learning_rate: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 64
    replay_capacity: int = 50000
    min_replay_size: int = 1000
    target_update_every: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50000
    max_steps_per_episode: int = 1000
    device: str = 'cpu'
