import random
import os
import numpy as np
from dnd_auction_game import AuctionGameClient
import math


class MyAgent:
    def __init__(self, top_fraction=0.5):
        self.my_id = ""
        self.top_fraction = top_fraction

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

        no_rounds = 1000
        progress = current_round / no_rounds
        self.top_fraction -= progress / no_rounds

        no_auctions = len(auction_ids)
        no_top = max(1, int(no_auctions * self.top_fraction))
        top_indices = np.argsort(expected_values)[-no_top:][::-1]
        top_auction_ids = auction_ids[top_indices]
        top_values = expected_values[top_indices]

        aggression = 0.5 + 3.5 * (progress**2.2)
        spend_rate = min(1.0, 0.02 + (math.exp(progress * 4) - 1) / (math.e**4 - 1))

        portions = top_values / top_values.sum()
        raw_bids = gold * portions * spend_rate * aggression

        raw_bids *= np.random.uniform(0.9, 1.1, size=raw_bids.shape)

        savings = max(0.01, 0.15 * (1 - progress))
        total_bid = raw_bids.sum()
        if total_bid > gold * (1 - savings):
            raw_bids *= (gold * (1 - savings)) / total_bid

        for a_id, bid in zip(top_auction_ids, raw_bids):
            bids[a_id] = bid

        return bids


if __name__ == "__main__":
    host = "opentsetlin.com"
    port = 8000
    agent_name = "O-block"
    player_id = "Mohammad Itani"

    game = AuctionGameClient(
        host=host, agent_name=agent_name, player_id=player_id, port=port
    )
    agent = MyAgent(0.12)

    try:
        game.run(agent.make_bid)
    except KeyboardInterrupt:
        print("<interrupt - shutting down>")

    print("<game is done>")
