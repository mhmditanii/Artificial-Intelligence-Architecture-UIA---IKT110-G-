import random
import os
import math
import json
import numpy as np
from dnd_auction_game import AuctionGameClient
from helper import (
    get_number_of_rounds,
    get_current_bank_stats,
    get_winning_bid_stats,
    compute_historical_winning_stats,
)


############################################################################################
#
# FortunaAgent
#   Learns to bid on auctions using the Fortuna algorithm
#
############################################################################################


class FortunaAgent:
    def __init__(
        self,
        theta: list[float],
        loadModel: bool,
        min_bid: int,
        bid_step: int,
        lambda_base: float,
        lambda_ramp: float,
    ):
        self.theta = theta
        self.total_rounds = 0
        self.load = loadModel
        self.min_bid = min_bid
        self.bid_step = bid_step
        self.lambda_base = lambda_base
        self.lambda_ramp = lambda_ramp
        self.best_run = {"bestU": -1e9, "bestB": 0, "auction": None}
        self.win_bids: list[int] = []  # Historical winners across the whole run

    def auction_estimated_value(self, a: dict) -> float:
        # EV = E[sum of dice] + bonus = num * (die+1)/2 + bonus
        return a["num"] * (a["die"] + 1) / 2.0 + a["bonus"]

    @staticmethod
    def utility(p: float, EV: float, b: int, λ: float) -> float:
        return p * EV - λ * b * (0.4 + 0.6 * p)

    def choose_lambda(
        self, interest: float, round_idx: int, total_rounds: int
    ) -> float:
        # simple, stable λ (increase slowly with rounds)
        base = self.lambda_base
        ramp = self.lambda_ramp * (round_idx / max(1, total_rounds - 1))
        return base + ramp

    def sigmoid(self, z: float) -> float:
        z = max(-30.0, min(30.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def predict(self, b: int) -> float:
        x = b / 300.0
        return self.sigmoid(self.theta[0] + self.theta[1] * x)

    # TODO: Implement this
    def learn_from_prev(self):
        self.theta = self.load_model_from_file() + [
            random.uniform(-10, 10),
            random.uniform(-10, 10),
        ]

    def load_model_from_file(self):
        path = "/home/ignacio/Proyects/2025/AI-Architecture/fortuna_model.json"
        with open(path, "r") as f:
            content = f.read()
        records = json.loads(content)
        best = max(records, key=lambda r: r["bestU"]) if records else None
        if best:
            return best["theta"]
        return [0, 5.0]

    def save_model_to_file(self, record: dict):
        """Persist one record per run inside a valid JSON array file."""
        path = "fortuna_model.json"
        existing: list = []
        if os.path.exists(path):
            with open(path, "r") as f:
                content = f.read()
            if content.strip():
                existing = json.loads(content)
        existing.append(record)
        with open(path, "w") as f:
            json.dump(existing, f, indent=2)

    def add_to_historical_winners(self, prev_auctions: dict):
        if not prev_auctions:
            return
        for _, a in prev_auctions.items():
            bids = a.get("bids", [])
            if not bids:
                continue
            b_max = int(bids[0]["gold"])  # clearing price
            self.win_bids.append(b_max)

    """
    EV = expected points for the auction
    p(b) = P(win | b) from the learned model.
    λ = the shadow price of gold (opportunity cost), capturing bank interest

    Expected points-minus-cost of bidding b on one auction:
    U(b) = p(b) * EV - λ * b (0.4 + 0.6 * p(b))

    Objective:
    Maximize the expected points-minus-cost U(b)
    """

    def bid(
        self,
        agent_id: str,
        current_round: int,
        states: dict,
        auctions: dict,
        prev_auctions: dict,
        bank_state: dict,
    ):
        if current_round == 0:
            if self.load:
                self.theta = self.load_model_from_file()
            self.total_rounds = get_number_of_rounds(bank_state)
            self.best_run = {"bestU": -1e9, "bestB": 0, "auction": None}

        current_interest_rate, current_bank_limit, current_gold_income = (
            get_current_bank_stats(bank_state)
        )

        # Accumulate historical winners from last round's auctions
        self.add_to_historical_winners(prev_auctions)

        # Historical max/mean
        hist_max_winning_bet, hist_mean_gold = compute_historical_winning_stats(
            self.win_bids
        )

        self.min_bid = int(hist_mean_gold)

        λ = self.choose_lambda(current_interest_rate, current_round, self.total_rounds)

        agent_state = states[agent_id]
        current_gold = agent_state["gold"]
        bids = {}

        for a_id, a in auctions.items():
            EV = self.auction_estimated_value(a)
            bestU, bestB = -1e9, 0

            # Max bid determined by historical mean and max
            per_cap = int(hist_max_winning_bet)

            # Where learning happens, we evaluate the utility of each bid.
            for b in range(self.min_bid, per_cap, self.bid_step):
                p = self.predict(b)
                U = self.utility(p, EV, b, λ)
                if U > bestU:
                    bestU, bestB = U, b

            # Update global best for the run
            if bestU > self.best_run["bestU"]:
                self.best_run = {"bestU": float(bestU), "bestB": int(bestB)}

            if bestU > 0 and bestB >= self.min_bid:
                if current_gold >= bestB:
                    bids[a_id] = bestB
                    current_gold -= bestB
                if current_gold < self.min_bid:
                    break

        # On final round, persist one JSON line with model + best performance this run
        if current_round == self.total_rounds and not self.load:
            record = {
                "theta": [float(v) for v in self.theta],
                "bestU": float(self.best_run["bestU"]),
                "bestB": int(self.best_run["bestB"]),
            }
            self.save_model_to_file(record)

        return bids


if __name__ == "__main__":
    host = "localhost"
    agent_name = "ignacio fortuna"
    player_id = "Ignacio"
    port = 8000

    game = AuctionGameClient(
        host=host, agent_name=agent_name, player_id=player_id, port=port
    )
    agent = FortunaAgent(
        theta=[-50, 500],
        loadModel=False,
        min_bid=300,
        bid_step=10,
        lambda_base=0.025,
        lambda_ramp=0.01,
    )
    try:
        game.run(agent.bid)
    except KeyboardInterrupt:
        print("<interrupt - shutting down>")

    print("<game is done>")
