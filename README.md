# Hex Game

- Hex Game, project for CSCI 680: Decision Making Under Certainty.


## Run

```bash
python3 server.py
```

Then visit `http://localhost:8008`.

## Train AlphaZero

The AlphaZero agent uses a policy-value neural network trained from self-play.
If no checkpoint exists yet, the agent still appears in the UI but plays with an
untrained network.

```bash
python3 train_alphazero.py --board-size 7 --iterations 20 --games-per-iteration 16 --simulations 80
```

This writes `models/alphazero_hex.pt`, which `server.py` loads automatically.
For a faster but weaker run, reduce `--iterations`, `--games-per-iteration`, or
`--simulations`.
