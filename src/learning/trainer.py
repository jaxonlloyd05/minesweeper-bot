import numpy as np

from game import MinesweeperEnv
from learning.agent import DQNAgent
from learning.config import DQNConfig
from learning.diagnostics import TrainingDiagnostics
from learning.replay import Transition


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

    def train(
        self,
        episodes,
        checkpoint_path=None,
        checkpoint_every=None,
        log_every=10,
        metrics_path=None,
    ):
        diagnostics = TrainingDiagnostics(
            csv_path=metrics_path,
            log_every=log_every,
            window=log_every or 10,
        )

        for episode in range(1, episodes + 1):
            summary = self.train_episode()
            self.history.append(summary)
            diagnostics.record(
                episode=episode,
                summary=summary,
                history=self.history,
                replay_size=len(self.agent.memory),
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
            'safe_moves': max(0, step - int(info.get('hit_mine', False))),
            'safe_click_rate': (
                max(0, step - int(info.get('hit_mine', False))) / step
                if step
                else 0.0
            ),
        }

def train_dqn(episodes=1000, checkpoint_path='minesweeper_dqn.pt', config=None):
    trainer = DQNTrainer(config=config)
    return trainer.train(
        episodes=episodes,
        checkpoint_path=checkpoint_path,
        checkpoint_every=max(1, episodes // 10),
    )
