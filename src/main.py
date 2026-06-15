import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from game import UNKNOWN
from learning import DQNAgent, DQNConfig, DQNTrainer
from web import WebHandler


WEIGHTS_DIR = Path('weights')
RUNS_DIR = Path('runs')


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not hasattr(args, 'handler'):
        parser.print_help()
        return 1

    return args.handler(args)


def build_parser():
    parser = argparse.ArgumentParser(
        description='Train or run a DQN Minesweeper agent.'
    )
    subparsers = parser.add_subparsers(dest='command')

    train_parser = subparsers.add_parser(
        'train',
        help='Train a DQN agent on the local Minesweeper simulator.',
    )
    add_train_args(train_parser)
    train_parser.set_defaults(handler=run_train)

    play_parser = subparsers.add_parser(
        'play-web',
        help='Load saved weights and play the online game through Playwright.',
    )
    add_play_web_args(play_parser)
    play_parser.set_defaults(handler=run_play_web)

    return parser


def add_train_args(parser):
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--height', type=int, default=16)
    parser.add_argument('--width', type=int, default=30)
    parser.add_argument('--mines', type=int, default=99)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--checkpoint-every', type=int, default=None)
    parser.add_argument('--weights-path', type=Path, default=None)
    parser.add_argument('--resume-weights-path', type=Path, default=None)
    parser.add_argument('--metrics-path', type=Path, default=None)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--min-replay-size', type=int, default=1000)
    parser.add_argument('--max-steps-per-episode', type=int, default=1000)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99)


def add_play_web_args(parser):
    parser.add_argument('--weights-path', type=Path, default=None)
    parser.add_argument('--max-moves', type=int, default=200)
    parser.add_argument('--delay', type=float, default=0.5)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--headless', action='store_true')


def run_train(args):
    WEIGHTS_DIR.mkdir(exist_ok=True)
    RUNS_DIR.mkdir(exist_ok=True)

    if args.resume_weights_path:
        if not args.resume_weights_path.exists():
            print(f'Resume weights file does not exist: {args.resume_weights_path}')
            return 1

        agent = DQNAgent.load(args.resume_weights_path, device=args.device)
        agent.update_training_config(
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            batch_size=args.batch_size,
            min_replay_size=args.min_replay_size,
            max_steps_per_episode=args.max_steps_per_episode,
        )
        config = agent.config
        trainer = DQNTrainer(config=config, agent=agent)
        print(f'Resuming from weights: {args.resume_weights_path}')
        warn_if_board_args_differ_from_checkpoint(args, config)
    else:
        config = DQNConfig(
            height=args.height,
            width=args.width,
            mines=args.mines,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            batch_size=args.batch_size,
            min_replay_size=args.min_replay_size,
            max_steps_per_episode=args.max_steps_per_episode,
            device=args.device,
        )
        trainer = DQNTrainer(config=config)

    metrics_path = args.metrics_path or default_metrics_path()
    checkpoint_path = args.weights_path or args.resume_weights_path
    checkpoint_every = args.checkpoint_every if checkpoint_path else None

    print(
        f'Training for {args.episodes} episodes on '
        f'{config.height}x{config.width} with {config.mines} mines.'
    )
    print(f'Device: {config.device}')
    print(f'Metrics CSV: {metrics_path}')
    if checkpoint_path and checkpoint_every:
        print(f'Checkpointing every {checkpoint_every} episodes to {checkpoint_path}')

    trainer.train(
        episodes=args.episodes,
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
        log_every=args.log_every,
        metrics_path=metrics_path,
    )

    if checkpoint_path:
        print(f'Saved weights: {checkpoint_path}')
        return 0

    save_path = default_weights_path()
    answer = input(f'Save final weights to {save_path}? [Y/n] ').strip().lower()
    if answer in ('', 'y', 'yes'):
        trainer.agent.save(save_path)
        print(f'Saved weights: {save_path}')
    else:
        print('Weights were not saved.')

    return 0


def run_play_web(args):
    weights_path = args.weights_path or choose_weights_file()
    if not weights_path:
        print('No weights file selected.')
        return 1

    if not weights_path.exists():
        print(f'Weights file does not exist: {weights_path}')
        return 1

    agent = DQNAgent.load(weights_path, device=args.device)
    web = WebHandler(headless=args.headless)
    grid = web.create_grid(agent.config.height, agent.config.width)

    print(f'Loaded weights: {weights_path}')
    print('Opening Minesweeper online...')

    try:
        web.open()

        for move_number in range(1, args.max_moves + 1):
            grid = web.update_grid_from_open_tiles(grid)
            state = web.game_state()
            if state != 'playing':
                print(f'Stopping before move {move_number}: game state is {state}.')
                return 0

            valid_actions = agent.valid_actions_from_state(grid)
            if not np.any(valid_actions):
                print('No valid unopened tiles remain.')
                return 0

            x, y = agent.select_best_move(grid, valid_actions)
            known_tiles = int(np.count_nonzero(grid != UNKNOWN))
            print(
                f'move={move_number} '
                f'x={x} y={y} '
                f'known_tiles={known_tiles} '
                f'valid_moves={int(np.count_nonzero(valid_actions))}'
            )
            web.click_tile(x, y)
            web.wait_after_move(args.delay)

            grid = web.update_grid_from_open_tiles(grid)
            state = web.game_state()
            if state != 'playing':
                print(f'Game finished after move {move_number}: {state}.')
                return 0

        print(f'Stopped after reaching max moves: {args.max_moves}.')
        return 0
    finally:
        web.close()


def choose_weights_file():
    WEIGHTS_DIR.mkdir(exist_ok=True)
    candidates = sorted(WEIGHTS_DIR.glob('*.pt'))
    if not candidates:
        return None

    print('Available weights files:')
    for index, path in enumerate(candidates, start=1):
        print(f'{index}. {path}')

    while True:
        choice = input('Choose weights file number: ').strip()
        try:
            index = int(choice)
        except ValueError:
            print('Please enter a number.')
            continue

        if 1 <= index <= len(candidates):
            return candidates[index - 1]

        print(f'Please choose a number between 1 and {len(candidates)}.')


def default_metrics_path():
    RUNS_DIR.mkdir(exist_ok=True)
    return RUNS_DIR / f'train_{timestamp()}.csv'


def default_weights_path():
    WEIGHTS_DIR.mkdir(exist_ok=True)
    return WEIGHTS_DIR / f'minesweeper_dqn_{timestamp()}.pt'


def timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def warn_if_board_args_differ_from_checkpoint(args, config):
    requested = (args.height, args.width, args.mines)
    loaded = (config.height, config.width, config.mines)
    if requested == loaded:
        return

    print(
        'Using board shape from checkpoint '
        f'({config.height}x{config.width}, {config.mines} mines); '
        'height/width/mines CLI values are ignored when resuming.'
    )


if __name__ == '__main__':
    raise SystemExit(main())
