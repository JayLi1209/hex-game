from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from hex_agents import (
    MODEL_PATH,
    AlphaZeroMCTS,
    HexPolicyValueNet,
    HexRules,
    encode_board,
)


TrainingExample = Tuple[torch.Tensor, torch.Tensor, float]


def choose_move_from_policy(policy: dict, temperature: float) -> int:
    moves = list(policy.keys())
    probabilities = torch.tensor([policy[move] for move in moves], dtype=torch.float32)

    if temperature <= 0:
        return max(policy, key=policy.get)

    probabilities = probabilities.pow(1.0 / temperature)
    probabilities = probabilities / probabilities.sum()
    selected = torch.multinomial(probabilities, 1).item()
    return moves[selected]


def play_self_play_game(
    model: HexPolicyValueNet,
    board_size: int,
    simulations: int,
    device: torch.device,
    temperature_moves: int,
) -> List[TrainingExample]:
    cells = [[None for _ in range(board_size)] for _ in range(board_size)]
    current_player = "blue"
    history = []

    for turn in range(board_size * board_size):
        mcts = AlphaZeroMCTS(model, simulations=simulations, device=device)
        policy = mcts.run(cells, current_player)
        policy_target = torch.zeros(board_size * board_size, dtype=torch.float32)
        for move, probability in policy.items():
            policy_target[move] = probability

        history.append((encode_board(cells, current_player), policy_target, current_player))

        temperature = 1.0 if turn < temperature_moves else 0.0
        move = choose_move_from_policy(policy, temperature)
        cells = HexRules.apply_move(cells, move, current_player)

        if HexRules.has_winner(cells, current_player):
            winner = current_player
            return [
                (state, target, 1.0 if player == winner else -1.0)
                for state, target, player in history
            ]

        current_player = HexRules.other_player(current_player)

    return [(state, target, 0.0) for state, target, _player in history]


def train_epoch(
    model: HexPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    examples: List[TrainingExample],
    batch_size: int,
    device: torch.device,
) -> float:
    random.shuffle(examples)
    states = torch.stack([example[0] for example in examples])
    policies = torch.stack([example[1] for example in examples])
    values = torch.tensor([example[2] for example in examples], dtype=torch.float32)

    loader = DataLoader(TensorDataset(states, policies, values), batch_size=batch_size, shuffle=True)
    total_loss = 0.0
    total_items = 0

    model.train()
    for state_batch, policy_batch, value_batch in loader:
        state_batch = state_batch.to(device)
        policy_batch = policy_batch.to(device)
        value_batch = value_batch.to(device)

        optimizer.zero_grad()
        logits, predicted_value = model(state_batch)
        value_loss = F.mse_loss(predicted_value, value_batch)
        policy_loss = -(policy_batch * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        loss = value_loss + policy_loss
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * state_batch.size(0)
        total_items += state_batch.size(0)

    return total_loss / max(total_items, 1)


def save_checkpoint(
    model: HexPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    output_path: Path,
    board_size: int,
    iteration: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "board_size": board_size,
            "iteration": iteration,
        },
        str(output_path),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Hex AlphaZero policy-value network.")
    parser.add_argument("--board-size", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--games-per-iteration", type=int, default=8)
    parser.add_argument("--simulations", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--replay-size", type=int, default=5000)
    parser.add_argument("--temperature-moves", type=int, default=8)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HexPolicyValueNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-4)
    replay_buffer: List[TrainingExample] = []

    if args.resume and args.output.exists():
        checkpoint = torch.load(str(args.output), map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])

    for iteration in range(1, args.iterations + 1):
        model.eval()
        new_examples = []
        for game_index in range(1, args.games_per_iteration + 1):
            examples = play_self_play_game(
                model=model,
                board_size=args.board_size,
                simulations=args.simulations,
                device=device,
                temperature_moves=args.temperature_moves,
            )
            new_examples.extend(examples)
            print(
                "iteration {}/{} self-play game {}/{} produced {} positions".format(
                    iteration,
                    args.iterations,
                    game_index,
                    args.games_per_iteration,
                    len(examples),
                ),
                flush=True,
            )

        replay_buffer.extend(new_examples)
        if len(replay_buffer) > args.replay_size:
            replay_buffer = replay_buffer[-args.replay_size :]

        losses = [
            train_epoch(model, optimizer, replay_buffer, args.batch_size, device)
            for _ in range(args.epochs)
        ]
        save_checkpoint(model, optimizer, args.output, args.board_size, iteration)
        print(
            "iteration {}/{} saved {} examples to {} loss {:.4f}".format(
                iteration,
                args.iterations,
                len(replay_buffer),
                args.output,
                losses[-1],
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
