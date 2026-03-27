from __future__ import annotations

import json
import pathlib
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict

from hex_agents import build_default_registry


ROOT = pathlib.Path(__file__).resolve().parent
REGISTRY = build_default_registry()
VALID_PLAYERS = {"blue", "red"}


def validate_board(board: Dict[str, Any]) -> None:
    size = board.get("size")
    cells = board.get("cells")

    if not isinstance(size, int) or size <= 0:
        raise ValueError("Board size must be a positive integer.")

    if not isinstance(cells, list) or len(cells) != size:
        raise ValueError("Board rows must match the declared size.")

    for row in cells:
        if not isinstance(row, list) or len(row) != size:
            raise ValueError("Board columns must match the declared size.")

        for cell in row:
            if cell is not None and cell not in VALID_PLAYERS:
                raise ValueError("Board cells must be null, 'blue', or 'red'.")


class HexRequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/api/agents":
            self.send_json({"agents": REGISTRY.list_agents()})
            return

        super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/api/move":
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown API route.")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length)
            payload = json.loads(body.decode("utf-8"))

            board = payload["board"]
            current_player = payload["currentPlayer"]
            opponent = payload["opponent"]
            agent_id = payload["agentId"]

            validate_board(board)

            if current_player not in VALID_PLAYERS or opponent not in VALID_PLAYERS:
                raise ValueError("Players must be 'blue' or 'red'.")

            agent = REGISTRY.create(agent_id)
            move = agent.choose_move(
                board=board,
                current_player=current_player,
                opponent=opponent,
            )
            self.send_json({"move": move})
        except KeyError as error:
            self.send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
        except ValueError as error:
            self.send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
        except json.JSONDecodeError:
            self.send_json({"error": "Request body must be valid JSON."}, status=HTTPStatus.BAD_REQUEST)

    def send_json(self, payload: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run() -> None:
    handler = partial(HexRequestHandler, directory=str(ROOT))
    server = ThreadingHTTPServer(("127.0.0.1", 8008), handler)
    print("Serving Hex app on http://127.0.0.1:8008")
    server.serve_forever()


if __name__ == "__main__":
    run()
