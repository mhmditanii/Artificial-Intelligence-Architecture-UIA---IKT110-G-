import os
import sys
import math
import random
from typing import Dict, Any, Tuple, List, Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from dnd_auction_game import AuctionGameClient
except Exception:
    AuctionGameClient = None


class Agent:
    def bid(self, auction_info: Dict[str, Any]) -> Dict[str, int]:
        raise NotImplementedError


def expected_value_for_auction(die: int, num: int, bonus: int) -> float:
    return num * (die + 1) / 2.0 + bonus


class StrategicAgent(Agent):
    def __init__(
        self,
        base_aggressiveness: float = 0.72,
        min_bid: int = 6,
        max_spend_fraction: float = 0.90,
        risk_balance: float = 0.60,
        min_aggr: float = 0.30,
        max_aggr: float = 0.92,
        lose_cashback_fraction: float = 0.60,
    ):
        self.base_aggressiveness = max(0.0, min(1.0, base_aggressiveness))
        self.current_aggressiveness = self.base_aggressiveness
        self.min_bid = max(1, int(min_bid))
        self.max_spend_fraction = max(0.0, min(1.0, max_spend_fraction))
        self.risk_balance = max(0.0, min(1.0, risk_balance))
        self.min_aggr = max(0.0, min(1.0, min_aggr))
        self.max_aggr = max(0.0, min(1.0, max_aggr))
        self.lose_cashback_fraction = max(0.0, min(1.0, lose_cashback_fraction))

        self.last_round_points_gained = 0
        self.last_round_gold_net_spent = 0

    def _estimate_win_probability(self, my_bid: int, others_gold: List[int]) -> float:
        if my_bid <= 0:
            return 0.0
        if not others_gold:
            return 1.0
        rivals_able = sum(1 for g in others_gold if g >= my_bid)
        total = len(others_gold)
        p = 1.0 - (rivals_able / max(1, total))
        return max(0.0, min(1.0, p))

    def _adjust_aggressiveness_with_history(self):
        score = self.last_round_points_gained - 0.35 * self.last_round_gold_net_spent
        if score > 0:
            self.current_aggressiveness = min(self.max_aggr, self.current_aggressiveness + 0.04)
        else:
            self.current_aggressiveness = max(self.min_aggr, self.current_aggressiveness - 0.04)

    def _extract_prev_round_stats(self, prev_auctions: Dict[str, Any], agent_id: str) -> Tuple[int, int]:
        points = 0
        net_spent = 0
        for _, a in prev_auctions.items():
            bids = a.get("bids", [])
            if not bids:
                continue
            for b in bids:
                if b.get("a_id") == agent_id:
                    bid_gold = int(b.get("gold", 0))
                    winner_id = bids[0].get("a_id")
                    if winner_id == agent_id:
                        points += int(a.get("reward", 0))
                        net_spent += bid_gold
                    else:
                        net_spent += bid_gold
                        net_spent -= int(self.lose_cashback_fraction * bid_gold)
                    break
        return max(0, points), max(0, net_spent)

    def _compute_spend_budget(self, my_gold: int, bank_state: Dict[str, Any], am_i_losing: bool) -> int:
        gi = bank_state.get("gold_income_per_round", [])
        next_income = int(gi[0]) if isinstance(gi, list) and len(gi) > 0 else 0
        next_income_factor = 0.55 if am_i_losing else 0.30
        spendable = max(0, int(my_gold + next_income_factor * next_income))
        return int(self.max_spend_fraction * spendable)

    def _am_i_losing(self, agent_id: str, states: Dict[str, Dict[str, int]]) -> bool:
        my_points = int(states[agent_id]["points"])
        best_other = 0
        for a, s in states.items():
            if a == agent_id:
                continue
            best_other = max(best_other, int(s["points"]))
        return my_points < best_other

    def bid(self, auction_info: Dict[str, Any]) -> Dict[str, int]:
        agent_id: str = auction_info["agent_id"]
        states: Dict[str, Dict[str, int]] = auction_info["states"]
        auctions: Dict[str, Dict[str, int]] = auction_info["auctions"]
        prev_auctions: Dict[str, Any] = auction_info.get("prev_auctions", {})
        bank_state: Dict[str, Any] = auction_info.get("bank_state", {})

        if not auctions:
            return {}

        gained, net_spent = self._extract_prev_round_stats(prev_auctions, agent_id)
        self.last_round_points_gained = gained
        self.last_round_gold_net_spent = net_spent
        self._adjust_aggressiveness_with_history()

        my_gold = int(states[agent_id]["gold"])
        am_losing = self._am_i_losing(agent_id, states)
        budget = self._compute_spend_budget(my_gold, bank_state, am_losing)
        if budget <= 0:
            return {}

        others_gold = [int(s["gold"]) for a, s in states.items() if a != agent_id]
        richest_other = max(others_gold) if others_gold else 1
        max_per_bid_cap = max(1, int(richest_other))

        auction_evs: List[Tuple[str, float]] = []
        for auction_id, a in auctions.items():
            ev = expected_value_for_auction(int(a["die"]), int(a["num"]), int(a["bonus"]))
            if ev > 0:
                auction_evs.append((auction_id, ev))

        if not auction_evs:
            return {}

        auction_evs.sort(key=lambda x: x[1], reverse=True)
        total_ev = sum(ev for _, ev in auction_evs)
        if total_ev <= 0:
            return {}

        aggr = self.current_aggressiveness
        if am_losing:
            aggr = min(self.max_aggr, aggr + 0.08)
        else:
            aggr = max(self.min_aggr, aggr - 0.05)

        bids: Dict[str, int] = {}
        remaining = budget

        # Focus on top-2 high EV auctions for stronger contention
        top_candidates = auction_evs[:2]
        # Heuristic estimate of strong rival bid level
        estimated_rival_bid = int(0.6 * max_per_bid_cap)

        for idx, (auction_id, ev) in enumerate(top_candidates):
            if remaining <= 0:
                break

            share = ev / total_ev
            # Allocate a large chunk to the best auction, smaller to the second
            weight = 0.75 if idx == 0 else 0.25
            base_target = int(max(self.min_bid, weight * aggr * budget * max(0.35, share)))

            # Encourage competitive bids around rival capacity, but never exceed caps
            competitive_boost = int(0.4 * estimated_rival_bid)
            target_bid = base_target + competitive_boost
            target_bid = min(target_bid, remaining, max_per_bid_cap)

            # Safety floor to ensure we actually contend
            target_bid = max(target_bid, min(self.min_bid, remaining))
            if target_bid <= 0:
                continue

            bids[auction_id] = target_bid
            remaining -= target_bid

        # If budget remains, sprinkle small bids on remaining positive-EV auctions
        if remaining > 0:
            for auction_id, ev in auction_evs[2:]:
                if remaining <= 0:
                    break
                if ev <= 0:
                    continue
                small = min(remaining, max(self.min_bid, int(0.05 * budget)))
                if small > 0:
                    bids.setdefault(auction_id, small)
                    remaining -= small

        return bids


class StrategicLive:
    def __init__(self):
        self.agent = StrategicAgent()
        self.agent_id: Optional[str] = None

    def bid_callback(self, agent_id: str, current_round: int, states: dict, auctions: dict, prev_auctions: dict, bank_state: dict):
        if self.agent_id is None:
            self.agent_id = agent_id
        auction_info = {
            "agent_id": agent_id,
            "states": states,
            "auctions": auctions,
            "prev_auctions": prev_auctions,
            "bank_state": bank_state,
        }
        return self.agent.bid(auction_info)


if __name__ == "__main__":
    host = os.environ.get("DND_HOST", "localhost")
    port = int(os.environ.get("DND_PORT", "8000"))
    agent_name = f"strategic_agent_{random.randint(1, 100000)}"
    player_id = "player_strategic_agent"

    if AuctionGameClient is None:
        raise RuntimeError("AuctionGameClient not available; install/update dnd-auction-game.")

    game = AuctionGameClient(host=host, agent_name=agent_name, player_id=player_id, port=port)
    adapter = StrategicLive()
    try:
        game.run(adapter.bid_callback)
    except KeyboardInterrupt:
        print("<interrupt - shutting down>")
    print("<game is done>")


