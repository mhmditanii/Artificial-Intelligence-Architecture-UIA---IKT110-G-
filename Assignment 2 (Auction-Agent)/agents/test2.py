import random
from dnd_auction_game import AuctionGameClient
import math


class MyAgent:
    def __init__(self):
        self.my_id = ""

    def set_id(self, agent_id: str):
        self.my_id = agent_id

    def make_bid(
        self, agent_id, current_round, states, auctions, prev_auctions, bank_state
    ):
        self.set_id(agent_id)

        agent = states[self.my_id]
        gold = agent["gold"]
        bids = {}
        n_rounds = 1000
        progress = current_round / n_rounds

        # spending and risk increase with rounds
        risk = 0.5 + 3.5 * (progress**2.2)
        spend_rate = min(1.0, 0.02 + (math.exp(progress * 4) - 1) / (math.e**4 - 1))

        total_value = 0
        values = {}
        for a_id, info in auctions.items():
            expected_value = (info["num"] * (info["die"] + 1) / 2) + info["bonus"]
            values[a_id] = expected_value
            total_value += expected_value

        if total_value == 0:
            for a_id in auctions.keys():
                bids[a_id] = 0
            return bids

        # distributing the gold with a 10% randomness
        for a_id, value in values.items():
            portion = value / total_value
            bid = gold * portion * spend_rate * risk
            bid *= random.uniform(0.9, 1.1)
            bids[a_id] = min(
                bid, gold
            )  # clip to gold if I bid was more than what I have

        # do not save money at the end of the game
        savings = max(0.15 * (1 - progress), 0.01)
        total_bid = sum(bids.values())

        if total_bid > gold * (1 - savings):
            scale = (gold * (1 - savings)) / total_bid
            for a_id in bids:
                bids[a_id] *= scale
        return bids


if __name__ == "__main__":
    host = "localhost"
    port = 8000
    agent_name = "geedorah"
    player_id = "ito"

    game = AuctionGameClient(
        host=host, agent_name=agent_name, player_id=player_id, port=port
    )
    agent = MyAgent()

    try:
        game.run(agent.make_bid)
    except KeyboardInterrupt:
        print("<interrupt - shutting down>")

    print("<game is done>")
