import csv
import time
from pathlib import Path

import numpy as np


class TrainingDiagnostics:
    fieldnames = [
        'episode',
        'reward',
        'loss',
        'won',
        'hit_mine',
        'steps',
        'safe_moves',
        'safe_click_rate',
        'avg_safe_click_rate',
        'win_rate',
        'mine_hit_rate',
        'avg_reward',
        'avg_loss',
        'avg_steps',
        'epsilon',
        'replay_size',
        'elapsed_seconds',
    ]

    def __init__(self, csv_path=None, log_every=10, window=10):
        self.csv_path = Path(csv_path) if csv_path else None
        self.log_every = log_every
        self.window = window
        self.started_at = time.time()

        if self.csv_path:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with self.csv_path.open('w', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
                writer.writeheader()

    def record(self, episode, summary, history, replay_size):
        row = self._build_row(episode, summary, history, replay_size)

        if self.csv_path:
            with self.csv_path.open('a', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
                writer.writerow(row)

        if self.log_every and episode % self.log_every == 0:
            print(self.format_row(row))

        return row

    def _build_row(self, episode, summary, history, replay_size):
        recent = history[-self.window:] if self.window else history
        return {
            'episode': episode,
            'reward': summary['reward'],
            'loss': summary['loss'],
            'won': int(summary['won']),
            'hit_mine': int(summary['hit_mine']),
            'steps': summary['steps'],
            'safe_moves': summary['safe_moves'],
            'safe_click_rate': summary['safe_click_rate'],
            'avg_safe_click_rate': self._mean(recent, 'safe_click_rate'),
            'win_rate': self._mean(recent, 'won'),
            'mine_hit_rate': self._mean(recent, 'hit_mine'),
            'avg_reward': self._mean(recent, 'reward'),
            'avg_loss': self._mean(recent, 'loss'),
            'avg_steps': self._mean(recent, 'steps'),
            'epsilon': summary['epsilon'],
            'replay_size': replay_size,
            'elapsed_seconds': time.time() - self.started_at,
        }

    def format_row(self, row):
        return (
            f"episode={row['episode']} "
            f"avg_reward={row['avg_reward']:.2f} "
            f"avg_loss={row['avg_loss']:.4f} "
            f"win_rate={row['win_rate']:.2f} "
            f"safe_click_rate={row['avg_safe_click_rate']:.2f} "
            f"mine_hit_rate={row['mine_hit_rate']:.2f} "
            f"avg_steps={row['avg_steps']:.1f} "
            f"epsilon={row['epsilon']:.3f} "
            f"replay={row['replay_size']} "
            f"elapsed={row['elapsed_seconds']:.1f}s"
        )

    @staticmethod
    def _mean(items, key):
        if not items:
            return 0.0
        return float(np.mean([item[key] for item in items]))
