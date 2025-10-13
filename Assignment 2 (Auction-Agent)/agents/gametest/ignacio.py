import os
import random
from collections import defaultdict, deque
from dnd_auction_game import AuctionGameClient
from helper import (
    get_other_agents_stats,
    get_current_bank_stats,
    get_next_round_gold,
    calculate_auction_expected_value,
    estimated_price,
    update_price_history,
)


class FirstAgent:
    """
    Smart 'many small bids' bot with market-awareness.

    Key ideas:
    - Maintain a moving average of winning prices per auction signature (die,num,bonus).
    - Rank auctions by EV / (price_estimate+1) and spread bids across the top ones.
    - Keep a reserve to benefit from bank interest caps and avoid going broke.
    """

    def __init__(
        self,
        min_auctions: int = 5,
        overpay_margin: float = 0.18,
        price_memory: int = 18,  # how many past wins we remember per auction type
        base_max_bid: int = 60,  # like tiny_bid, but adaptive
        rich_max_bid: int = 500,  # when next income is high
        min_bid: int = 12,
        rounds_qt: int = 12,
    ):
        self.base_max_bid = base_max_bid
        self.rich_max_bid = rich_max_bid
        self.bank_state = {}
        self.price_history = defaultdict(lambda: deque(maxlen=price_memory))
        self.min_bid = min_bid
        self.min_auctions = min_auctions
        self.overpay_margin = overpay_margin
        self.rounds_qt = rounds_qt

    def isHighestInterestRate(self, interest_rate: float):
        return max(self.bank_state.get("bank_interest_per_round", [])) == interest_rate

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
            self.bank_state = bank_state
            self.rounds_qt = len(bank_state["bank_interest_per_round"])

        # Update price history from previous round
        if prev_auctions:
            update_price_history(self.price_history, prev_auctions)

        me = states[agent_id]
        gold = int(me["gold"])

        # Get additional insights using helper functions
        others_gold, others_points = get_other_agents_stats(agent_id, states)

        # Bank / income info using helper functions
        next_income = get_next_round_gold(bank_state)
        interest_rate, bank_limit, gold_income = get_current_bank_stats(bank_state)

        # I spend all my gold except the bank limit, so that I maximize my interest profit
        # WARNING: Spendable is not necessarily the optimum amount to bid.
        # Because you may win more money bidding more than saving it for interest.
        if current_round == 0:
            spendable = gold
        else:
            spendable = gold - bank_limit

        # 2) Decide per-auction bid cap
        # TODO: IMPROVE ON HOW TO DECIDE THE MAX BID
        per_auction_max = self.rich_max_bid if next_income > 1050 else self.base_max_bid

        # Score each auction by EV / estimated price
        scored = []
        for a_id, a in auctions.items():
            # On average how many points will I get from this auction?
            ev = calculate_auction_expected_value(a)
            # How much gold do I think I need to win this auction.
            est = estimated_price(
                self.price_history,
                a["die"],
                a["num"],
                a["bonus"],
                max(others_gold) if others_gold else gold,
            )
            # simple score: EV per unit price; add tiny jitter to avoid ties
            # HIGHER SCORE: Means that the auction gives me more points for my gold.
            # LOWER SCORE: Means that the auction gives me less points for my gold.
            score = ev / (est + 1.0) + random.uniform(0, 1e-6)
            scored.append((score, a_id, ev, est, a))

        scored.sort(reverse=True)

        # Allocate bids greedily across top auctions
        bids = {}

        # Ensure we have enough to bid
        if spendable < self.min_bid:
            print(f"Not enough gold to bid (have {spendable}, need {self.min_bid})")
            return {}

        # ensure we try at least a few auctions, but stop if funds are low
        max_targets = min(
            len(scored), max(self.min_auctions, spendable // self.min_bid)
        )

        for i in range(max_targets):
            if i >= len(scored):
                break
            score, a_id, ev, est, a = scored[i]

            # If the EV is extremely poor relative to cost, skip
            if ev < 1.0:
                continue

            # Calculate bid amount - be more aggressive
            target_bid = min(per_auction_max, spendable // (max_targets - i))
            target_bid = max(target_bid, self.min_bid)  # Ensure minimum bid

            # Add overpay margin to increase winning chances
            target_bid = int(target_bid * (1 + self.overpay_margin))

            if target_bid <= spendable:
                bids[a_id] = target_bid
                spendable -= target_bid

        return bids


if __name__ == "__main__":
    host = "localhost"
    port = 8000
    agent_name = "Ignacio main"
    player_id = "Ignacio"

    game = AuctionGameClient(
        host=host, agent_name=agent_name, player_id=player_id, port=port
    )
    agent = FirstAgent()
    try:
        game.run(agent.bid)
    except KeyboardInterrupt:
        print("<interrupt - shutting down>")
    print("<game is done>")
