from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


Move = Dict[str, int]
BoardState = Dict[str, object]
CellGrid = List[List[Optional[str]]]
StateKey = Tuple[Tuple[Tuple[Optional[str], ...], ...], str]

PLAYERS = ("blue", "red")
MODEL_PATH = Path(__file__).resolve().parent / "models" / "alphazero_hex.pt"


@dataclass(frozen=True)
class AgentDescriptor:
    agent_id: str
    label: str


class HexRules:
    directions = (
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
    )

    @staticmethod
    def other_player(player: str) -> str:
        return "red" if player == "blue" else "blue"

    @staticmethod
    def available_moves(cells: CellGrid) -> List[int]:
        size = len(cells)
        return [
            row * size + col
            for row in range(size)
            for col in range(size)
            if cells[row][col] is None
        ]

    @staticmethod
    def apply_move(cells: CellGrid, move: int, player: str) -> CellGrid:
        size = len(cells)
        row, col = divmod(move, size)
        next_cells = [list(board_row) for board_row in cells]
        next_cells[row][col] = player
        return next_cells

    @classmethod
    def has_winner(cls, cells: CellGrid, player: str) -> bool:
        size = len(cells)
        frontier: List[Tuple[int, int]] = []
        visited = set()

        if player == "blue":
            for row in range(size):
                if cells[row][0] == player:
                    frontier.append((row, 0))
                    visited.add((row, 0))
        else:
            for col in range(size):
                if cells[0][col] == player:
                    frontier.append((0, col))
                    visited.add((0, col))

        while frontier:
            row, col = frontier.pop()
            if (player == "blue" and col == size - 1) or (
                player == "red" and row == size - 1
            ):
                return True

            for row_offset, col_offset in cls.directions:
                next_row = row + row_offset
                next_col = col + col_offset
                key = (next_row, next_col)
                if (
                    0 <= next_row < size
                    and 0 <= next_col < size
                    and key not in visited
                    and cells[next_row][next_col] == player
                ):
                    visited.add(key)
                    frontier.append(key)

        return False


def board_cells(board: BoardState) -> CellGrid:
    return [list(row) for row in board["cells"]]  # type: ignore[index]


def encode_board(cells: CellGrid, current_player: str) -> torch.Tensor:
    size = len(cells)
    opponent = HexRules.other_player(current_player)
    encoded = torch.zeros((3, size, size), dtype=torch.float32)

    for row in range(size):
        for col in range(size):
            if cells[row][col] == current_player:
                encoded[0, row, col] = 1.0
            elif cells[row][col] == opponent:
                encoded[1, row, col] = 1.0

    if current_player == "red":
        encoded[2].fill_(1.0)

    return encoded


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class HexPolicyValueNet(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 4) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.residual = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        self.policy_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.residual(self.stem(x))
        policy_logits = self.policy_head(x).flatten(start_dim=1)
        value = self.value_head(x).squeeze(1)
        return policy_logits, value


class AlphaZeroMCTS:
    def __init__(
        self,
        model: HexPolicyValueNet,
        simulations: int = 80,
        c_puct: float = 1.5,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.simulations = simulations
        self.c_puct = c_puct
        self.device = device or torch.device("cpu")
        self.priors: Dict[StateKey, Dict[int, float]] = {}
        self.visit_counts: Dict[Tuple[object, int], int] = {}
        self.value_sums: Dict[Tuple[object, int], float] = {}

    @staticmethod
    def state_key(cells: CellGrid, player: str) -> StateKey:
        return tuple(tuple(row) for row in cells), player

    def run(self, cells: CellGrid, current_player: str) -> Dict[int, float]:
        legal_moves = HexRules.available_moves(cells)
        if not legal_moves:
            return {}

        for _ in range(self.simulations):
            self._search(cells, current_player)

        root_key = self.state_key(cells, current_player)
        counts = {move: self.visit_counts.get((root_key, move), 0) for move in legal_moves}
        total = sum(counts.values())
        if total == 0:
            return {move: 1.0 / len(legal_moves) for move in legal_moves}
        return {move: count / total for move, count in counts.items()}

    def _search(self, cells: CellGrid, current_player: str) -> float:
        previous_player = HexRules.other_player(current_player)
        if HexRules.has_winner(cells, previous_player):
            return -1.0

        legal_moves = HexRules.available_moves(cells)
        if not legal_moves:
            return 0.0

        key = self.state_key(cells, current_player)
        if key not in self.priors:
            value = self._expand(key, cells, current_player, legal_moves)
            return value

        move = self._select_move(key, legal_moves)
        next_cells = HexRules.apply_move(cells, move, current_player)
        value = -self._search(next_cells, HexRules.other_player(current_player))

        edge = (key, move)
        self.visit_counts[edge] = self.visit_counts.get(edge, 0) + 1
        self.value_sums[edge] = self.value_sums.get(edge, 0.0) + value
        return value

    def _expand(
        self,
        key: StateKey,
        cells: CellGrid,
        current_player: str,
        legal_moves: List[int],
    ) -> float:
        with torch.no_grad():
            batch = encode_board(cells, current_player).unsqueeze(0).to(self.device)
            logits, value = self.model(batch)
            logits = logits.squeeze(0).cpu()

        masked_logits = torch.full_like(logits, -1.0e9)
        masked_logits[legal_moves] = logits[legal_moves]
        probabilities = F.softmax(masked_logits, dim=0)
        self.priors[key] = {move: float(probabilities[move]) for move in legal_moves}
        return float(value.item())

    def _select_move(
        self,
        key: StateKey,
        legal_moves: List[int],
    ) -> int:
        total_visits = 1 + sum(self.visit_counts.get((key, move), 0) for move in legal_moves)
        best_score = -float("inf")
        best_move = legal_moves[0]

        for move in legal_moves:
            edge = (key, move)
            visits = self.visit_counts.get(edge, 0)
            q_value = 0.0 if visits == 0 else self.value_sums.get(edge, 0.0) / visits
            prior = self.priors[key].get(move, 0.0)
            exploration = self.c_puct * prior * math.sqrt(total_visits) / (1 + visits)
            score = q_value + exploration
            if score > best_score:
                best_score = score
                best_move = move

        return best_move


class BaseHexAgent:
    descriptor: AgentDescriptor

    def choose_move(
        self,
        board: BoardState,
        current_player: str,
        opponent: str,
    ) -> Optional[Move]:
        raise NotImplementedError

    @staticmethod
    def available_moves(board: BoardState) -> List[Move]:
        moves: List[Move] = []
        cells = board["cells"]

        for row_index, row in enumerate(cells):  # type: ignore[assignment]
            for col_index, cell in enumerate(row):
                if cell is None:
                    moves.append({"row": row_index, "col": col_index})

        return moves


class RandomAgent(BaseHexAgent):
    descriptor = AgentDescriptor(agent_id="random", label="Random")

    def choose_move(
        self,
        board: BoardState,
        current_player: str,
        opponent: str,
    ) -> Optional[Move]:
        del current_player
        del opponent
        moves = self.available_moves(board)
        return random.choice(moves) if moves else None


class AlphaZeroAgent(BaseHexAgent):
    descriptor = AgentDescriptor(agent_id="alphazero", label="AlphaZero")

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HexPolicyValueNet().to(self.device)
        self.is_trained = MODEL_PATH.exists()
        if self.is_trained:
            checkpoint = torch.load(str(MODEL_PATH), map_location=self.device)
            state_dict = checkpoint.get("model_state", checkpoint)
            self.model.load_state_dict(state_dict)
        self.model.eval()

    def choose_move(
        self,
        board: BoardState,
        current_player: str,
        opponent: str,
    ) -> Optional[Move]:
        del opponent
        cells = board_cells(board)
        legal_moves = HexRules.available_moves(cells)
        if not legal_moves:
            return None

        simulations = 120 if self.is_trained else 25
        mcts = AlphaZeroMCTS(self.model, simulations=simulations, device=self.device)
        policy = mcts.run(cells, current_player)
        move = max(policy, key=policy.get) if policy else random.choice(legal_moves)
        row, col = divmod(move, len(cells))
        return {"row": row, "col": col}


class AgentRegistry:
    def __init__(self) -> None:
        self._factories = {}

    def register(self, agent_cls) -> None:
        descriptor = agent_cls.descriptor
        self._factories[descriptor.agent_id] = {
            "descriptor": descriptor,
            "factory": agent_cls,
        }

    def list_agents(self) -> List[Dict[str, str]]:
        return [
            {
                "id": entry["descriptor"].agent_id,
                "label": entry["descriptor"].label,
            }
            for entry in self._factories.values()
        ]

    def create(self, agent_id: str) -> BaseHexAgent:
        if agent_id not in self._factories:
            raise KeyError(f"Unknown agent id: {agent_id}")

        return self._factories[agent_id]["factory"]()


def build_default_registry() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register(RandomAgent)
    registry.register(AlphaZeroAgent)
    return registry
