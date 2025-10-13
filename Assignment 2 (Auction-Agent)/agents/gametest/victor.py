from typing import Dict, Any, Tuple, List

import os
import random

from dnd_auction_game import AuctionGameClient


average_roll_for_die = {
    2: 1.5,
    3: 2.0,
    4: 2.5,
    6: 3.5,
    8: 4.5,
    10: 5.5,
    12: 6.5,
    20: 10.5,
}


class Agent:
    def bid(self, auction_info: Dict[str, Any]) -> Dict[str, int]:
        raise NotImplementedError


class FortunaHybridAgent(Agent):
    def __init__(
        self,
        base_aggressiveness: float = 0.65,
        min_bid: int = 5,
        max_spend_fraction: float = 0.72,
        risk_balance: float = 0.5,
        min_aggr: float = 0.35,
        max_aggr: float = 0.9,
    ):
        self.base_aggressiveness = max(0.0, min(1.0, base_aggressiveness))
        self.current_aggressiveness = self.base_aggressiveness
        self.min_bid = max(0, int(min_bid))
        self.max_spend_fraction = max(0.0, min(1.0, max_spend_fraction))
        self.risk_balance = max(
            0.0, min(1.0, risk_balance)
        )  # 0 => conserva oro, 1 => maximiza puntos
        self.min_aggr = max(0.0, min(1.0, min_aggr))
        self.max_aggr = max(0.0, min(1.0, max_aggr))

        self.last_round_revenue = 0
        self.last_round_spent = 0

    def fortuna(self, die: int, num: int, bonus: int) -> float:
        base = average_roll_for_die.get(die, 0.0)
        return (base * num) + bonus

    def _estimate_win_probability(self, my_bid: int, others_gold: List[int]) -> float:
        if my_bid <= 0:
            return 0.0
        if not others_gold:
            return 1.0
        stronger = sum(1 for g in others_gold if g >= my_bid)
        total = len(others_gold)
        p = 1.0 - (stronger / max(1, total))
        return max(0.0, min(1.0, p))

    def _adjust_aggressiveness_with_history(self):
        effective_loss = max(
            0, self.last_round_spent - int(0.6 * self.last_round_spent)
        )
        score = self.last_round_revenue - 0.35 * effective_loss
        if score > 0:
            self.current_aggressiveness = min(
                self.max_aggr, self.current_aggressiveness + 0.04
            )
        else:
            self.current_aggressiveness = max(
                self.min_aggr, self.current_aggressiveness - 0.04
            )

    def _extract_prev_round_stats(
        self, prev_auctions: Dict[str, Any], agent_id: str
    ) -> Tuple[int, int]:
        revenue = 0
        spent = 0
        for a_id, a in prev_auctions.items():
            bids = a.get("bids", [])
            for b in bids:
                if b.get("a_id") == agent_id:
                    gold_bid = int(b.get("gold", 0))
                    spent += gold_bid
                    if bids and bids[0].get("a_id") == agent_id:
                        revenue += int(a.get("reward", 0))
                    else:
                        spent -= int(0.6 * gold_bid)
                    break
        return revenue, max(0, spent)

    def bid(self, auction_info: Dict[str, Any]) -> Dict[str, int]:
        agent_id: str = auction_info["agent_id"]
        states: Dict[str, Dict[str, int]] = auction_info["states"]
        auctions: Dict[str, Dict[str, int]] = auction_info["auctions"]
        prev_auctions: Dict[str, Any] = auction_info.get("prev_auctions", {})
        bank_state: Dict[str, Any] = auction_info.get("bank_state", {})

        if not auctions:
            return {}

        rev, spent = self._extract_prev_round_stats(prev_auctions, agent_id)
        self.last_round_revenue = rev
        self.last_round_spent = spent
        self._adjust_aggressiveness_with_history()

        current_gold = int(states[agent_id]["gold"])
        gi = bank_state.get("gold_income_per_round", [])
        next_income = int(gi[0]) if isinstance(gi, list) and len(gi) > 0 else 0

        spendable = int(current_gold + 0.35 * next_income)
        spendable = max(0, spendable)
        budget = int(self.max_spend_fraction * spendable)
        if budget <= 0:
            return {}

        others_gold = [s["gold"] for a, s in states.items() if a != agent_id]
        richest_other = max(others_gold) if others_gold else 1
        max_per_bid_cap = max(1, int(richest_other))

        auction_evs: List[Tuple[str, float]] = []
        for auction_id, a in auctions.items():
            ev = self.fortuna(a["die"], a["num"], a["bonus"])
            auction_evs.append((auction_id, ev))

        auction_evs.sort(key=lambda x: x[1], reverse=True)
        total_value = sum(max(0.0, ev) for _, ev in auction_evs)
        if total_value <= 0:
            return {}

        bids: Dict[str, int] = {}
        remaining = budget

        for auction_id, ev in auction_evs:
            if remaining <= 0:
                break
            if ev <= 0:
                continue

            share = ev / total_value
            greedy_target = int(
                max(self.min_bid, self.current_aggressiveness * share * budget)
            )
            conservative_target = int(max(self.min_bid, 0.5 * share * budget))
            target_bid = int(
                (1 - self.risk_balance) * conservative_target
                + self.risk_balance * greedy_target
            )
            target_bid = min(target_bid, remaining, max_per_bid_cap)
            if target_bid <= 0:
                continue

            p_win = self._estimate_win_probability(target_bid, others_gold)
            ev_points = ev
            ev_gold_preserve = 0.6 * (1 - p_win) * target_bid
            ev_points_component = p_win * ev_points
            score = (
                self.risk_balance * ev_points_component
                + (1 - self.risk_balance) * ev_gold_preserve
            )

            if score > 0:
                bids[auction_id] = target_bid
                remaining -= target_bid

        return bids


if __name__ == "__main__":
    host = os.environ.get("DND_HOST", "localhost")
    port = int(os.environ.get("DND_PORT", "8000"))
    agent_name = f"victor salamanca{random.randint(1, 100000)}"
    player_id = "player_fortuna_hybrid_agent"

    if AuctionGameClient is None:
        raise RuntimeError(
            "AuctionGameClient not available; install/update dnd-auction-game."
        )

    game = AuctionGameClient(
        host=host, agent_name=agent_name, player_id=player_id, port=port
    )
    adapter = FortunaHybridAgent()
    try:
        game.run(adapter.bid)
    except KeyboardInterrupt:
        print("<interrupt - shutting down>")
    print("<game is done>")
