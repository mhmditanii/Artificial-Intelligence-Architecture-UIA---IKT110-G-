import random
import os
import numpy as np

from dnd_auction_game import AuctionGameClient

MIN_START_GPP = 60


def _extract_bid(entry):
    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        return entry[1]
    if isinstance(entry, dict):
        return entry.get("bid")
    return None


def _ev_from_simple_schema(info):
    if not isinstance(info, dict):
        return None
    num = info.get("num")
    die = info.get("die")
    bonus = info.get("bonus", 0)
    if isinstance(num, (int, float)) and isinstance(die, (int, float)):
        try:
            ev = float(num) * ((float(die) + 1.0) / 2.0) + float(bonus or 0.0)
            return ev
        except Exception:
            return None
    return None


def jamie_dimon(
    agent_id: str,
    current_round: int,
    states: dict,
    auctions: dict,
    prev_auctions: dict,
    bank_state: dict,
):
    # --- trygge oppslag ---
    agent_state = (states or {}).get(agent_id, {}) or {}
    current_gold = float(agent_state.get("gold", 0))

    gold_income = (bank_state or {}).get("gold_income_per_round", []) or []
    next_income = float(gold_income[0]) if gold_income else 0.0

    # bank_limits = (bank_state or {}).get("bank_limit_per_round", []) or []
    # current_bank_limit = bank_limits[0] if bank_limits else current_gold
    # try:
    #    current_bank_limit = float(current_bank_limit)
    # except Exception:
    #    current_bank_limit = float(current_gold)

    budget = max(0.0, current_gold)
    if budget < 1:
        print(f"[round {current_round}] gold={int(current_gold)} budget=0 (ingen bud)")
        return {}

    # --- helper for å hente bud fra ulike formater ---

    # --- estimer market GPP fra forrige runde ---
    ratios = []
    for _aid, a in (prev_auctions or {}).items():
        if not isinstance(a, dict):
            continue
        win_bid = a.get("winning_bid")

        if not isinstance(win_bid, (int, float)):
            all_bids = a.get("all_bids") or a.get("bids") or []
            if isinstance(all_bids, (list, tuple)) and all_bids:
                bids = [_extract_bid(x) for x in all_bids]
                bids = [b for b in bids if isinstance(b, (int, float))]
                if bids:
                    win_bid = max(bids)

        reward = (
            a.get("reward") or a.get("got_reward") or a.get("value") or a.get("result")
        )
        if (
            isinstance(win_bid, (int, float))
            and isinstance(reward, (int, float))
            and reward > 0
        ):
            ratios.append(win_bid / reward)

    ratios.sort()
    if ratios:
        mid = len(ratios) // 2
        market_gpp = (
            ratios[mid] if len(ratios) % 2 else 0.5 * (ratios[mid - 1] + ratios[mid])
        )
    else:
        market_gpp = float(MIN_START_GPP)

    # --- samle auksjoner og EV ---
    items = []
    for a_id, info in (auctions or {}).items():
        ev = _ev_from_simple_schema(info) if isinstance(info, dict) else None
        items.append((a_id, ev))

    # høy EV først; None sist
    items.sort(
        key=lambda t: (
            t[1] is None,
            -(t[1] if isinstance(t[1], (int, float)) else -1e18),
            str(t[0]),
        )
    )

    # basisbud om EV mangler – liten andel av neste inntekt
    base_bid_unknown_ev = max(1, int(max(5.0, next_income * 0.02)))

    # --- fordel budsjett og bygg bud-dict ---
    remaining = int(budget)
    n = len(items)
    bids: dict = {}
    # --- logging (header) ---
    print(
        f"[round {current_round}] gold={int(current_gold)} "
        f"budget={int(budget)} market_gpp≈{market_gpp:.2f} auctions={len(items)}"
    )

    for idx, (a_id, ev) in enumerate(items, start=1):
        auctions_left = n - (idx - 1)
        if remaining < 1 or auctions_left <= 0:
            break

        fair_slice = max(1, int(remaining / auctions_left))
        if isinstance(ev, (int, float)):
            target = max(1, int(ev * market_gpp))
        else:
            target = base_bid_unknown_ev

        bid = int(min(target, fair_slice, remaining))

        ev_str = f"{ev:5.2f}" if isinstance(ev, (int, float)) else "  n/a"
        print(
            f"  -> BID {bid:6d} on {str(a_id):>5s} | EV={ev_str} "
            f"target={target:6d} slice={fair_slice:5d} remaining {remaining:6d} -> {remaining - bid:7d}"
        )
        if bid >= 1:
            bids[a_id] = bid
            remaining -= bid

    return bids


"""
    best_a_id, best_ev = None, float("-inf")
    for a_id, info in (auctions or {}).items():
        #ev=info.get("expected_value")
        ev=(info or {}).get("expected_value")
        if isinstance(ev, (int, float)) and ev > best_ev:
            best_ev = ev
            best_a_id = a_id

"""
"""
    bank_limits = (bank_state or {}).get("bank_limit_per_round", [])
    current_bank_limit=bank_limits[0] if isinstance(bank_limits, (list, tuple)) and bank_limits else current_gold
    if not isinstance(current_bank_limit, (int, float)):
        try:
            current_bank_limit = float(current_bank_limit)
        except Exception:
            current_bank_limit = current_gold
    cap = max(0.0, min(float(current_gold), float(current_bank_limit)))



    bids={}
    if best_a_id is not None and isinstance(best_ev, (int, float)) and best_ev > 0 and cap > 0:
        planned_bid= int(min(best_ev * market_gpp, cap))
        if planned_bid >=1:
            bids[best_a_id]=planned_bid
    return bids
"""
"""
    return {
        "plan": {
            "target_auction": best_a_id,
            "expected_value": best_ev if best_a_id else None,
            "market_gpp": market_gpp,
            "planned_bid_cap": cap,
            "planned_bid": planned_bid,
        },
        "bids": bids
    }
"""

if __name__ == "__main__":
    host = "localhost"
    agent_name = "magnus"
    player_id = "id_of_human_player"
    port = 8000

    game = AuctionGameClient(
        host=host, agent_name=agent_name, player_id=player_id, port=port
    )
    try:
        game.run(jamie_dimon)
    except KeyboardInterrupt:
        print("<interrupt - shutting down>")

    print("<game is done>")

