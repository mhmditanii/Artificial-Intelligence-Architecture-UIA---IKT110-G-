"""
Helper functions for auction game agents.

This module provides utility functions for analyzing game state,
calculating statistics, and making bidding decisions.
"""

import numpy as np
from typing import Dict, Tuple, Deque, List
from collections import deque


def get_other_agents_stats(agent_id: str, states: Dict) -> Tuple[list, list]:
    """
    Calculate the mean gold and points for all other players.
    """
    gold = []
    points = []
    
    for other_agent_id, state in states.items():
        if other_agent_id == agent_id:
            continue
        gold.append(state["gold"])
        points.append(state["points"])
    
    return gold, points


def get_wealthiest_agent(states: Dict) -> Tuple[str, int]:
    """
    Find the agent with the most gold.
    """
    wealthiest_id = max(states.keys(), key=lambda aid: states[aid]["gold"])
    return wealthiest_id, states[wealthiest_id]["gold"]


def get_next_round_gold(bank_state: Dict) -> int:
    """
    Get the gold amount we receive next round.
    """
    gold_income = bank_state.get("gold_income_per_round", [])
    return gold_income[0] if gold_income else 0

def get_number_of_rounds(bank_state: Dict) -> int:
    """
    Get the number of rounds in total
    """
    gold_income = bank_state.get("gold_income_per_round", [])
    return len(gold_income)


def get_current_bank_stats(bank_state: Dict) -> Tuple[float, float, int]:
    """
    Get the current bank statistics.
    """
    interest_rate = bank_state.get("bank_interest_per_round", [])
    bank_limit = bank_state.get("bank_limit_per_round", [])
    gold_income = bank_state.get("gold_income_per_round", [])

    if (len(interest_rate) == 0):
        return 0, 0, 0
    
    return interest_rate[0], bank_limit[0], gold_income[0]

def get_winning_bid_stats(prev_auctions: dict) -> tuple[int, float, int]:
    # returns (max_gold, mean_gold, count)
    total = 0
    count = 0
    max_gold = 0
    for _, auction in prev_auctions.items():
        bids = auction.get("bids", [])
        if not bids:
            continue
        # if winner is first:
        winner = bids[0]
        g = int(winner.get("gold", 0)) if isinstance(winner, dict) else int(winner)
        max_gold = max(max_gold, g)
        total += g
        count += 1
    mean = float(total) / count if count > 0 else 0.0
    return max_gold, mean, count
    
def calculate_auction_expected_value(auction: Dict) -> float:
    """
    Calculate the expected value of an auction based on dice statistics.
    """
    average_roll_for_die = {
        2: 1.5, 3: 2.0, 4: 2.5, 6: 3.5, 8: 4.5, 10: 5.5, 12: 6.5, 20: 10.5
    }
    
    die = auction["die"]
    num = auction["num"]
    bonus = auction["bonus"]
    
    return (average_roll_for_die[die] * num) + bonus


def update_price_history(price_history: Dict[Tuple[int, int, int], Deque[int]], prev_auctions: Dict) -> None:
    """
    Update a shared price history with the winning bid (gold) from the previous round's auctions.

    Parameters
    - price_history: mapping (die, num, bonus) -> deque of past winning bid amounts
    - prev_auctions: structure from logs with keys like:
        { auction_id: { "die": int, "num": int, "bonus": int,
                         "bids": [ {"a_id": str, "gold": int}, ... ] } }
      The first element in "bids" is the winning bid.
    """
    if not prev_auctions:
        return
    for _, auction_data in prev_auctions.items():
        bids = auction_data.get("bids", [])
        if not bids:
            continue
        winning_bid_data = bids[0]
        if isinstance(winning_bid_data, dict):
            winning_bid_amount = int(winning_bid_data.get("gold", 0))
        else:
            # Backward compatibility if structure was a raw number
            winning_bid_amount = int(winning_bid_data)
        key = (
            int(auction_data.get("die", 0)),
            int(auction_data.get("num", 0)),
            int(auction_data.get("bonus", 0)),
        )
        if key not in price_history:
            price_history[key] = deque(maxlen=12)
        price_history[key].append(winning_bid_amount)


def estimated_price(price_history: Dict[Tuple[int, int, int], Deque[int]],
                    die: int,
                    num: int,
                    bonus: int,
                    others_max_gold: int) -> float:
    """
    Estimate a fair price for an auction signature based on historical winning bids.

    Parameters
    - price_history: mapping (die, num, bonus) -> deque of past winning bid amounts
    - die, num, bonus: the auction signature
    - others_max_gold: cap the estimate by what others can likely afford

    Returns a float estimate capped by others_max_gold, with cold-start fallback.
    """
    key = (int(die), int(num), int(bonus))
    history = price_history.get(key)
    
    if not history or len(history) == 0:
        return min(30.0, float(others_max_gold))
    est = sum(history) / len(history)
    return min(float(est), float(others_max_gold))


def compute_historical_winning_stats(win_bids: List[int]) -> Tuple[int, float]:
    """
    Given a running list of historical winning bids (clearing prices),
    return (hist_max, hist_mean).

    - hist_max never decreases across rounds because it's computed over
      the cumulative list.
    - hist_mean is the arithmetic mean over all entries.
    """
    if not win_bids:
        return 0, 0.0
    hist_max = max(win_bids)
    hist_mean = float(sum(win_bids)) / float(len(win_bids))
    return hist_max, hist_mean
