const PLAYERS = {
  human: "blue",
  agent: "red",
};

class HexBoard {
  constructor(size) {
    this.size = size;
    this.cells = Array.from({ length: size }, () => Array(size).fill(null));
  }

  isInside(row, col) {
    return row >= 0 && row < this.size && col >= 0 && col < this.size;
  }

  isEmpty(row, col) {
    return this.cells[row][col] === null;
  }

  place(row, col, player) {
    if (!this.isInside(row, col) || !this.isEmpty(row, col)) {
      return false;
    }

    this.cells[row][col] = player;
    return true;
  }

  getAvailableMoves() {
    const moves = [];
    for (let row = 0; row < this.size; row += 1) {
      for (let col = 0; col < this.size; col += 1) {
        if (this.cells[row][col] === null) {
          moves.push({ row, col });
        }
      }
    }
    return moves;
  }

  toPayload() {
    return {
      size: this.size,
      cells: this.cells.map((row) => [...row]),
    };
  }
}

class HexRules {
  static directions = [
    [-1, 0],
    [-1, 1],
    [0, -1],
    [0, 1],
    [1, -1],
    [1, 0],
  ];

  static hasWinner(board, player) {
    const frontier = [];
    const visited = new Set();

    if (player === PLAYERS.human) {
      for (let row = 0; row < board.size; row += 1) {
        if (board.cells[row][0] === player) {
          frontier.push({ row, col: 0 });
          visited.add(`${row},0`);
        }
      }
    } else {
      for (let col = 0; col < board.size; col += 1) {
        if (board.cells[0][col] === player) {
          frontier.push({ row: 0, col });
          visited.add(`0,${col}`);
        }
      }
    }

    while (frontier.length > 0) {
      const current = frontier.pop();
      const reachedGoal =
        player === PLAYERS.human
          ? current.col === board.size - 1
          : current.row === board.size - 1;

      if (reachedGoal) {
        return true;
      }

      for (const [rowOffset, colOffset] of HexRules.directions) {
        const nextRow = current.row + rowOffset;
        const nextCol = current.col + colOffset;
        const key = `${nextRow},${nextCol}`;

        if (
          board.isInside(nextRow, nextCol) &&
          board.cells[nextRow][nextCol] === player &&
          !visited.has(key)
        ) {
          visited.add(key);
          frontier.push({ row: nextRow, col: nextCol });
        }
      }
    }

    return false;
  }
}

class PythonAgentService {
  async fetchAgents() {
    const response = await fetch("/api/agents");
    return this.parseJson(response, "Could not load Python agents.");
  }

  async chooseMove({ agentId, board, currentPlayer, opponent }) {
    const response = await fetch("/api/move", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        agentId,
        board,
        currentPlayer,
        opponent,
      }),
    });

    return this.parseJson(response, "Could not get a move from the Python agent.");
  }

  async parseJson(response, fallbackMessage) {
    let payload = null;

    try {
      payload = await response.json();
    } catch (error) {
      if (!response.ok) {
        throw new Error(fallbackMessage);
      }

      throw error;
    }

    if (!response.ok) {
      throw new Error(payload.error || fallbackMessage);
    }

    return payload;
  }
}

class HexGameController {
  constructor({
    boardRoot,
    statusMessage,
    turnIndicator,
    sizeInput,
    agentInput,
    service,
  }) {
    this.boardRoot = boardRoot;
    this.statusMessage = statusMessage;
    this.turnIndicator = turnIndicator;
    this.sizeInput = sizeInput;
    this.agentInput = agentInput;
    this.service = service;

    this.agentEntries = [];
    this.board = null;
    this.isHumanTurn = true;
    this.gameOver = false;
    this.isAgentThinking = false;
  }

  async initialize() {
    const payload = await this.service.fetchAgents();
    this.agentEntries = payload.agents;
    this.populateAgentOptions();
    this.startNewGame();
  }

  populateAgentOptions() {
    this.agentInput.replaceChildren();

    for (const entry of this.agentEntries) {
      const option = document.createElement("option");
      option.value = entry.id;
      option.textContent = entry.label;
      this.agentInput.append(option);
    }
  }

  getSelectedAgent() {
    return this.agentEntries.find((entry) => entry.id === this.agentInput.value) ?? null;
  }

  startNewGame() {
    const size = Number(this.sizeInput.value);

    this.board = new HexBoard(size);
    this.isHumanTurn = true;
    this.gameOver = false;
    this.isAgentThinking = false;

    this.turnIndicator.textContent = "Your turn";
    this.statusMessage.textContent =
      "Blue connects left to right. Pick an open hex to make the first move.";

    this.render();
  }

  handleCellClick(row, col) {
    if (
      this.gameOver ||
      !this.isHumanTurn ||
      this.isAgentThinking ||
      !this.board.isEmpty(row, col)
    ) {
      return;
    }

    this.board.place(row, col, PLAYERS.human);
    this.render();

    if (HexRules.hasWinner(this.board, PLAYERS.human)) {
      this.finishGame("You win! Blue connected left to right.");
      return;
    }

    if (this.board.getAvailableMoves().length === 0) {
      this.finishGame("Board full. In Hex that should not happen, but no moves remain.");
      return;
    }

    this.isHumanTurn = false;
    this.isAgentThinking = true;

    const selectedAgent = this.getSelectedAgent();
    const agentLabel = selectedAgent ? selectedAgent.label : "Python";

    this.turnIndicator.textContent = `${agentLabel} agent is thinking`;
    this.statusMessage.textContent = "Waiting for the Python policy to choose a move.";
    this.render();

    window.setTimeout(() => {
      void this.playAgentTurn();
    }, 260);
  }

  async playAgentTurn() {
    if (this.gameOver) {
      return;
    }

    try {
      const selectedAgent = this.getSelectedAgent();
      const payload = await this.service.chooseMove({
        agentId: selectedAgent?.id ?? "random",
        board: this.board.toPayload(),
        currentPlayer: PLAYERS.agent,
        opponent: PLAYERS.human,
      });

      const move = payload.move;
      this.isAgentThinking = false;

      if (!move) {
        this.finishGame("No valid moves remain for the agent.");
        return;
      }

      this.board.place(move.row, move.col, PLAYERS.agent);

      if (HexRules.hasWinner(this.board, PLAYERS.agent)) {
        this.render();
        this.finishGame("Red wins. The agent connected top to bottom.");
        return;
      }

      this.isHumanTurn = true;
      this.turnIndicator.textContent = "Your turn";
      this.statusMessage.textContent = `Agent played row ${move.row + 1}, column ${move.col + 1}.`;
      this.render();
    } catch (error) {
      this.gameOver = true;
      this.isAgentThinking = false;
      this.turnIndicator.textContent = "Python agent unavailable";
      this.statusMessage.textContent =
        error instanceof Error
          ? `${error.message} Restart the Python server and begin a new game.`
          : "The Python agent could not be reached. Restart the server and begin a new game.";
      this.render();
    }
  }

  finishGame(message) {
    this.gameOver = true;
    this.isAgentThinking = false;
    this.turnIndicator.textContent = "Game over";
    this.statusMessage.textContent = message;
    this.render();
  }

  render() {
    this.boardRoot.replaceChildren();

    for (let row = 0; row < this.board.size; row += 1) {
      const rowElement = document.createElement("div");
      rowElement.className = "board-row";
      rowElement.dataset.row = String(row);
      rowElement.style.setProperty("--row-index", String(row));
      rowElement.setAttribute("role", "row");

      for (let col = 0; col < this.board.size; col += 1) {
        const button = document.createElement("button");
        const owner = this.board.cells[row][col];
        const isPlayable =
          owner === null && !this.gameOver && this.isHumanTurn && !this.isAgentThinking;

        button.type = "button";
        button.className = "hex-cell";
        button.dataset.owner = owner ?? "empty";
        button.dataset.playable = String(isPlayable);
        button.setAttribute("role", "gridcell");
        button.setAttribute(
          "aria-label",
          `Row ${row + 1}, column ${col + 1}${owner ? `, occupied by ${owner}` : ", empty"}`
        );
        button.disabled = !isPlayable;
        button.addEventListener("click", () => this.handleCellClick(row, col));

        rowElement.append(button);
      }

      this.boardRoot.append(rowElement);
    }
  }
}

const boardRoot = document.querySelector("#board");
const statusMessage = document.querySelector("#status-message");
const turnIndicator = document.querySelector("#turn-indicator");
const sizeInput = document.querySelector("#board-size");
const agentInput = document.querySelector("#agent-select");
const resetButton = document.querySelector("#reset-button");

const game = new HexGameController({
  boardRoot,
  statusMessage,
  turnIndicator,
  sizeInput,
  agentInput,
  service: new PythonAgentService(),
});

sizeInput.addEventListener("change", () => game.startNewGame());
agentInput.addEventListener("change", () => game.startNewGame());
resetButton.addEventListener("click", () => game.startNewGame());

void game.initialize().catch((error) => {
  turnIndicator.textContent = "Python agent unavailable";
  statusMessage.textContent =
    error instanceof Error
      ? `${error.message} Start the Python server to play.`
      : "Start the Python server to play.";
});
