from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional


Move = Dict[str, int]
BoardState = Dict[str, object]


@dataclass(frozen=True)
class AgentDescriptor:
    agent_id: str
    label: str


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

        for row_index, row in enumerate(cells):
            for col_index, cell in enumerate(row):
                if cell is None:
                    moves.append({"row": row_index, "col": col_index})

        return moves


class RandomHexAgent(BaseHexAgent):
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
        if not moves:
            return None

        return random.choice(moves)


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
    registry.register(RandomHexAgent)
    return registry
