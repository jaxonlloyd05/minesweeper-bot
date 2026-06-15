import numpy as np

from src.game import MinesweeperEnv
from src.learning.agent import DQNAgent
from src.learning.config import DQNConfig
from src.learning.replay import Transition


class DQNTrainer:
    def __init__(self, config=None, env=None, agent=None):
        self.config = config or DQNConfig()
        self.env = env or MinesweeperEnv(
            height=self.config.height,
            width=self.config.width,
            mines=self.config.mines,
        )
        self.agent = agent or DQNAgent(self.config)
        self.history = []

    def train(self, episodes, checkpoint_path=None, checkpoint_every=None, log_every=10):
        for episode in range(1, episodes + 1):
            summary = self.train_episode()
            self.history.append(summary)

            if log_every and episode % log_every == 0:
                avg_reward = self._average_recent('reward', log_every)
                avg_loss = self._average_recent('loss', log_every)
                win_rate = self._average_recent('won', log_every)
                print(
                    f'episode={episode} '
                    f'avg_reward={avg_reward:.2f} '
                    f'avg_loss={avg_loss:.4f} '
                    f'win_rate={win_rate:.2f} '
                    f'epsilon={self.agent.epsilon:.3f}'
                )

            should_checkpoint = (
                checkpoint_path
                and checkpoint_every
                and episode % checkpoint_every == 0
            )
            if should_checkpoint:
                self.agent.save(checkpoint_path)

        if checkpoint_path:
            self.agent.save(checkpoint_path)

        return self.history

    def train_episode(self):
        state = self.env.reset()
        total_reward = 0.0
        losses = []
        info = {}

        for step in range(1, self.config.max_steps_per_episode + 1):
            valid_actions = self.env.valid_action_mask()
            action = self.agent.select_action(state, valid_actions, training=True)
            next_state, reward, done, info = self.env.step(action)
            next_valid_actions = self.env.valid_action_mask()

            self.agent.remember(
                Transition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    next_valid_actions=next_valid_actions,
                )
            )

            loss = self.agent.learn()
            if loss is not None:
                losses.append(loss)

            state = next_state
            total_reward += reward

            if done:
                break

        return {
            'reward': total_reward,
            'steps': step,
            'won': bool(info.get('won', False)),
            'hit_mine': bool(info.get('hit_mine', False)),
            'loss': float(np.mean(losses)) if losses else 0.0,
            'epsilon': self.agent.epsilon,
        }

    def _average_recent(self, key, window):
        recent = self.history[-window:]
        if not recent:
            return 0.0
        return float(np.mean([item[key] for item in recent]))


def train_dqn(episodes=1000, checkpoint_path='minesweeper_dqn.pt', config=None):
    trainer = DQNTrainer(config=config)
    return trainer.train(
        episodes=episodes,
        checkpoint_path=checkpoint_path,
        checkpoint_every=max(1, episodes // 10),
    )
