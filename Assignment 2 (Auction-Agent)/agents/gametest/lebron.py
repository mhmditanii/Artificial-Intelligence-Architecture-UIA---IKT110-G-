import random
from dnd_auction_game import AuctionGameClient
import numpy as np

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

# Global state
aggression = 0.6
loss_streak = 0
round_history = []


def calculate_ev(auction):
    """Beregn forventet verdi for en auksjon"""
    return (average_roll_for_die[auction["die"]] * auction["num"]) + auction["bonus"]


def estimate_future_gold(current_gold, bank_state, rounds_ahead):
    """Estimer hvor mye gull vi vil ha i fremtiden med bank og inntekt"""
    gold = current_gold

    for i in range(min(rounds_ahead, len(bank_state.get("gold_income_per_round", [])))):
        # Legg til inntekt
        income = bank_state["gold_income_per_round"][i]
        gold += income

        # Legg til renter
        interest_rate = bank_state["bank_interest_per_round"][i]
        bank_limit = bank_state["bank_limit_per_round"][i]

        # Renter gis kun på gull opp til grensen
        interest_gold = min(gold, bank_limit)
        gold += interest_gold * interest_rate

    return gold


def should_save_for_bank(current_gold, current_round, bank_state, my_points):
    """Avgjør om vi burde spare gull for bankrente"""

    if not bank_state:
        return False, 0

    # Hent neste runde sin info
    next_interest = (
        bank_state["bank_interest_per_round"][0]
        if bank_state.get("bank_interest_per_round")
        else 0
    )
    next_limit = (
        bank_state["bank_limit_per_round"][0]
        if bank_state.get("bank_limit_per_round")
        else 0
    )
    rounds_left = (
        bank_state.get("gold_income_per_round", [1])[0]
        if isinstance(bank_state.get("gold_income_per_round"), list)
        else 1
    )

    # Ikke spar hvis det er få runder igjen
    if len(bank_state.get("bank_interest_per_round", [])) < 3:
        return False, 0

    # Hvis vi har mye gull og god rente, spar noe
    if next_interest >= 0.10 and current_gold > next_limit * 0.7:
        # Beregn fremtidig verdi av å spare
        future_interest = current_gold * next_interest

        # Hvis vi kan tjene mer enn 100 gull i renter, spar litt
        if future_interest > 100:
            # Spar nok til å nå bank limit
            save_amount = min(current_gold * 0.3, next_limit - current_gold * 0.7)
            return True, max(0, save_amount)

    # Hvis vi har nok poeng og god rente fremover, bygg bank
    if my_points >= 8:
        avg_interest = np.mean(bank_state.get("bank_interest_per_round", [0])[:3])
        if avg_interest > 0.08:
            return True, current_gold * 0.4

    return False, 0


def analyze_competition(prev_auctions, current_ev):
    """Analyser tidligere auksjoner for å estimere konkurranse"""
    if not prev_auctions:
        return 1.0  # Default multiplikator

    winning_ratios = []

    for auction_id, data in prev_auctions.items():
        bids_list = data.get("bids", [])
        if not bids_list:
            continue

        winning_bid = bids_list[0].get("bid", 0)
        prev_ev = calculate_ev(data)

        if prev_ev > 0:
            ratio = winning_bid / prev_ev
            winning_ratios.append(ratio)

    if winning_ratios:
        # Bruk 70. persentil for å være konkurransedyktig
        percentile_70 = np.percentile(winning_ratios, 70)
        return percentile_70

    return 1.0


def smart_bidder(agent_id, current_round, states, auctions, prev_auctions, bank_state):
    global aggression, loss_streak, round_history

    agent_state = states[agent_id]
    current_gold = agent_state["gold"]
    my_points = agent_state["points"]
    bids = {}

    print(f"\n{'=' * 60}")
    print(f"🎲 RUNDE {current_round} 🎲")
    print(f"{'=' * 60}")
    print(f"💰 Gull: {current_gold} | 🎯 Poeng: {my_points}")

    # --- Lær av forrige runde ---
    if prev_auctions:
        won_something = False
        my_bids_last_round = {}

        for auction_id, data in prev_auctions.items():
            bids_list = data.get("bids", [])

            # Finn våre bud
            for bid_info in bids_list:
                if bid_info.get("agent_id") == agent_id:
                    my_bids_last_round[auction_id] = bid_info.get("bid", 0)

            # Sjekk om vi vant
            if bids_list and bids_list[0].get("agent_id") == agent_id:
                won_something = True
                reward = data.get("reward", 0)
                print(f"✅ Vant {auction_id}! Fikk {reward} poeng")

        if not won_something and my_bids_last_round:
            loss_streak += 1
            print(f"❌ Tapte alle auksjoner. Tap i rad: {loss_streak}")
        else:
            loss_streak = max(0, loss_streak - 1)
            print(f"🔥 Vant! Tap-streak tilbakestilt: {loss_streak}")

    # --- Juster aggresjon basert på situasjon ---
    base_aggression = 0.5 + (0.08 * loss_streak)

    # Mer aggressiv hvis vi trenger poeng
    if my_points < 6:
        base_aggression += 0.15
    elif my_points < 10:
        base_aggression += 0.08

    # Mindre aggressiv hvis vi leder
    if my_points >= 12:
        base_aggression -= 0.1

    aggression = min(0.95, max(0.3, base_aggression))
    print(f"⚔️  Aggresjon: {aggression:.2%} (base + tap-streak)")

    # --- Bank analyse ---
    if bank_state:
        rounds_left = len(bank_state.get("gold_income_per_round", [1]))
        next_income = (
            bank_state["gold_income_per_round"][0]
            if bank_state.get("gold_income_per_round")
            else 0
        )
        next_interest = (
            bank_state["bank_interest_per_round"][0]
            if bank_state.get("bank_interest_per_round")
            else 0
        )
        next_limit = (
            bank_state["bank_limit_per_round"][0]
            if bank_state.get("bank_limit_per_round")
            else 0
        )

        print(f"\n🏦 BANK INFO:")
        print(f"   Runder igjen: {rounds_left}")
        print(f"   Neste inntekt: {next_income} gull")
        print(f"   Neste rente: {next_interest:.1%} (maks {next_limit} gull)")

        # Estimer fremtidig gull
        future_gold = estimate_future_gold(current_gold, bank_state, 3)
        print(f"   Estimert gull om 3 runder: {future_gold:.0f}")

        should_save, save_amount = should_save_for_bank(
            current_gold, current_round, bank_state, my_points
        )
        if should_save:
            print(f"   💎 SPARER {save_amount:.0f} gull for bank!")
            current_gold -= save_amount

    # --- Sjekk om siste runde ---
    rounds_remaining = (
        len(bank_state.get("gold_income_per_round", [1])) if bank_state else 1
    )
    if rounds_remaining <= 1:
        # Finn beste auksjon
        best_auction = max(auctions.items(), key=lambda x: calculate_ev(x[1]))
        bids[best_auction[0]] = current_gold
        print(f"\n⚡ SISTE RUNDE - ALL IN!")
        print(f"   {best_auction[0]}: {current_gold} gull")
        return bids

    # --- Beregn forventet verdi for alle auksjoner ---
    auction_analysis = []
    for auction_id, auction in auctions.items():
        ev = calculate_ev(auction)

        # Beregn varians (risiko)
        die = auction["die"]
        num = auction["num"]
        variance = num * ((die**2 - 1) / 12)
        std_dev = variance**0.5

        auction_analysis.append(
            {"id": auction_id, "ev": ev, "std_dev": std_dev, "auction": auction}
        )

    # Sorter etter forventet verdi
    auction_analysis.sort(key=lambda x: x["ev"], reverse=True)

    print(f"\n📊 AUKSJONS-ANALYSE:")
    for i, a in enumerate(auction_analysis[:3]):
        print(f"   #{i + 1} {a['id']}: EV={a['ev']:.1f} (±{a['std_dev']:.1f})")

    # --- Estimer konkurransedyktig bud-nivå ---
    competition_multiplier = analyze_competition(prev_auctions, 0)
    print(f"\n🎯 Konkurransemultiplikator: {competition_multiplier:.2f}x")

    # --- Fordel gull basert på strategi ---
    total_ev = sum(a["ev"] for a in auction_analysis)
    available_gold = int(current_gold * aggression)

    print(f"\n💵 Fordeler {available_gold}/{current_gold} gull")

    # Strategi: Vekt mot topp auksjoner
    weights = []
    for i, a in enumerate(auction_analysis):
        if i == 0:
            weight = 0.5  # 50% til beste
        elif i == 1:
            weight = 0.3  # 30% til nest beste
        elif i == 2:
            weight = 0.15  # 15% til tredje beste
        else:
            weight = 0.05 / max(1, len(auction_analysis) - 3)  # Rest fordeles
        weights.append(weight)

    gold_remaining = available_gold

    for i, (a, weight) in enumerate(zip(auction_analysis, weights)):
        if gold_remaining <= 0:
            break

        # Kalkuler bud basert på EV, konkurranse og vår vekt
        base_bid = a["ev"] * competition_multiplier * 40  # 40 gull per forventet poeng
        weighted_bid = available_gold * weight

        # Bruk gjennomsnitt av de to tilnærmingene
        bid = int((base_bid + weighted_bid) / 2)
        bid = max(20, min(bid, gold_remaining))  # Minimum 20 gull per bud

        if bid > 0:
            bids[a["id"]] = bid
            gold_remaining -= bid
            print(f"   ➜ {a['id']}: {bid} gull")

    # --- Sikkerhetsnett: Alltid ha minst ett bud ---
    if not bids and current_gold > 0:
        best_id = auction_analysis[0]["id"]
        bids[best_id] = max(20, current_gold // 2)
        print(f"\n⚠️  FALLBACK BUD: {best_id} = {bids[best_id]} gull")

    total_bid = sum(bids.values())
    print(f"\n💰 TOTALT BUD: {total_bid} gull")
    print(f"   Gjenstår: {current_gold - total_bid} gull (60% refund hvis tap)")
    print(f"{'=' * 60}\n")

    return bids


if __name__ == "__main__":
    host = "localhost"
    agent_name = f"lebron{random.randint(1, 1000)}"
    player_id = "Maksymilian"
    port = 8000
    game = AuctionGameClient(
        host=host, agent_name=agent_name, player_id=player_id, port=port
    )
    try:
        game.run(smart_bidder)
    except KeyboardInterrupt:
        print("\n<interrupt - shutting down>")
    print("<game is done>")
