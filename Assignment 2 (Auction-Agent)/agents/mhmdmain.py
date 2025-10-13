import random
import os
import numpy as np
from dnd_auction_game import AuctionGameClient
import math


class MyAgent:
    def __init__(self, top_fraction=0.5, total_rounds=1000):
        self.my_id = ""
        self.top_fraction = top_fraction
        self.total_rounds = total_rounds

    def set_id(self, agent_id: str):
        self.my_id = agent_id

    def make_bid(
        self, agent_id, current_round, states, auctions, prev_auctions, bank_state
    ):
        self.set_id(agent_id)
        agent = states[self.my_id]
        gold = agent["gold"]
        bids = {}

        if not auctions or gold <= 0:
            return {}

        auction_ids = np.array(list(auctions.keys()))
        expected_values = np.array(
            [
                (info["num"] * (info["die"] + 1) / 2) + info["bonus"]
                for info in auctions.values()
            ]
        )

        num_auctions = len(auction_ids)
        num_top = max(1, int(num_auctions * self.top_fraction))
        top_indices = np.argsort(expected_values)[-num_top:][::-1]  # descending
        top_auction_ids = auction_ids[top_indices]
        top_values = expected_values[top_indices]

        # ---------------------------
        # Aggression and spend curve (rescaled)
        # ---------------------------
        progress = current_round / self.total_rounds
        # Non-linear scaling for aggressive spending
        aggression = 0.7 + 4.0 * (progress**2.0)
        spend_rate = min(
            1.0, 0.05 + (math.exp(progress * 5) - 1) / (math.e**5 - 1)
        )  # aggressive even early

        # ---------------------------
        # Allocate gold proportionally to top auctions
        # ---------------------------
        portions = top_values / top_values.sum()
        raw_bids = gold * portions * spend_rate * aggression
        raw_bids *= np.random.uniform(0.9, 1.1, size=raw_bids.shape)  # randomness

        # Scale to ensure we don't overspend
        reserve_fraction = max(
            0.01, 0.05 * (1 - progress)
        )  # nearly nothing left at end
        total_bid = raw_bids.sum()
        if total_bid > gold * (1 - reserve_fraction):
            raw_bids *= (gold * (1 - reserve_fraction)) / total_bid

        # Assign bids
        for aid, bid in zip(top_auction_ids, raw_bids):
            bids[aid] = bid

            # print(
            # f"Round {current_round:4d}/{self.total_rounds} | Agg={aggression:.2f} | Top {num_top}/{num_auctions} | Spend={sum(bids.values()):.1f}/{gold:.1f}"
        # )
        return bids


if __name__ == "__main__":
    host = "localhost"
    port = 8000
    agent_name = "itani"
    player_id = "itani"

    agent = MyAgent(top_fraction=0.5)

    game = AuctionGameClient(
        host=host, agent_name=agent_name, player_id=player_id, port=port
    )

    try:
        game.run(agent.make_bid)
    except KeyboardInterrupt:
        print("<interrupt - shutting down>")

    print("<game is done>")
