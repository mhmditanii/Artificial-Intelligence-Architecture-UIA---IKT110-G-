import random
import numpy as np
from dnd_auction_game import AuctionGameClient
import matplotlib.pyplot as plt

# Constants
TOTAL_ROUNDS = 1000 # IMPORTANT! don't forget to adapt!!!
N_ROUNDS_TRAINING = 10


class BidPredictor:
    """Class to store opponent bid patterns and predict winning bids"""

    def __init__(self, degree=2):
        self.bid_history = []  # List of (average_reward, winning_bid) tuples
        self.agent_bids = []
        self.agent_winning_bids = []
        self.coefficients = None
        self.degree = degree

    def add_observation(self, average_reward, winning_bid):
        """Add a new observation from previous round"""
        self.bid_history.append((average_reward, winning_bid))

    def add_agent_bid(self, average_reward, my_bid):
        """Add a new observation from previous round"""
        self.agent_bids.append((average_reward, my_bid))

    def add_agent_winning_bid(self, average_reward, my_bid):
        """Add a new observation from previous round"""
        self.agent_winning_bids.append((average_reward, my_bid))

    def train_model(self):
        """Fit regression on observed bids"""
        if len(self.bid_history) < self.degree + 2:
            # Not enough data, use heuristic
            return False

        x = np.array([reward for reward, bid in self.bid_history])
        y = np.array([bid for reward, bid in self.bid_history])

        # Using numpy polyfit (degree 1 = linear)
        self.coefficients = np.polyfit(x, y, self.degree)
        return True

    def predict_bid(self, average_reward):
        """Predict winning bid for given average reward"""
        if self.coefficients is None:
            return average_reward * 5

        predicted = np.polyval(self.coefficients, average_reward)

        # Sanity check: bid should be reasonable
        return max(0, predicted)


# Global predictor instance
predictor = BidPredictor()


def smart_bid(agent_id: str, current_round: int, states: dict, auctions: dict,
              prev_auctions: dict, bank_state: dict):
    # ===== PART 1: TRAIN MODEL FROM PREVIOUS AUCTIONS =====
    if prev_auctions:
        for auction_id, auction_data in prev_auctions.items():
            bids = auction_data.get("bids", [])
            if not bids:
                continue

            # Calculate average reward for this auction
            die = auction_data["die"]
            num = auction_data["num"]
            bonus = auction_data["bonus"]
            avg_reward = (((die + 1) / 2.0) * num) + bonus

            # Get winning bid
            winning_bid = bids[0]["gold"]
            if bids[0]["a_id"] == agent_id:
                predictor.add_agent_winning_bid(avg_reward, winning_bid)

            # Add to training data
            predictor.add_observation(avg_reward, winning_bid)

        # Retrain model every N rounds
        if current_round % N_ROUNDS_TRAINING == 0:
            predictor.train_model()

    # Get current state
    current_gold = states[agent_id]["gold"]


    # ===== PART 2: CALCULATE ESTIMATED VALUE FOR EACH AUCTION =====
    auctions_filtered = []

    for auction_id, auction in auctions.items():
        minimum_reward = auction["num"] + auction["bonus"]

        # Discard where minimum reward is unimportant
        if minimum_reward < 1:
            continue

        auction_copy = auction.copy()
        auction_copy["id"] = auction_id
        auction_copy["minimum_reward"] = minimum_reward
        auction_copy["average_reward"] = (((auction["die"] + 1) / 2.0) * auction["num"]) + auction["bonus"]

        # ===== FROM PART 1: GET SUITABLE GOLD BIDDING FROM MODEL =====
        predicted_winning_bid = predictor.predict_bid(auction_copy["average_reward"])
        auction_copy["predicted_bid"] = predicted_winning_bid

        auctions_filtered.append(auction_copy)

    if not auctions_filtered:
        return {}

    # Strategy: Use greedy approach, bid on auction in that order, until reach gold reserve
    auctions_filtered.sort(key=lambda x: x["average_reward"], reverse=True)


    # ===== PART 3: GOLD RESERVE =====
    # determine gold reserve depending on round number
    """
    Round 0 → reserve ≈ 40% of gold
    Round 50 → reserve ≈ 30%
    Round 100 → reserve ≈ 0%
    """
    progress = current_round / TOTAL_ROUNDS
    reserve_ratio = max(0.0, (-0.5 * progress**2 + 0.1 * progress + 0.5))

    reserve = current_gold * reserve_ratio
    available_gold = current_gold - reserve


    # ===== PART 4: CALCULATE AGGRESSION LEVEL =====
    # Aggression function:
    """
    Early rounds → 0.8–0.9
    Mid-game → ~1.0
    End-game → ~1.5
    """
    base_aggression = 0.8 + 0.7 * progress**3.5


    # ===== PART 5: SELECT AND BID ON AUCTIONS =====
    bids = {}
    remaining_gold = available_gold

    # bid on best auctions, until gold reserve reached
    for auction in auctions_filtered:
        if remaining_gold <= 0:
            break

        # Calculate our bid: predicted winning bid * aggression
        base_bid = auction["predicted_bid"] * base_aggression
        final_bid = min(base_bid, remaining_gold)
        final_bid = int(final_bid)

        bids[auction["id"]] = final_bid
        remaining_gold -= final_bid
        predictor.add_agent_bid(auction["average_reward"], final_bid)

    return bids




def plot_learning_results(predictor: BidPredictor):
    """Plot the relationship between average reward and winning bid (supports any polynomial degree)"""
    if not predictor.bid_history:
        print("No data to plot.")
        return

    x = np.array([r for r, _ in predictor.bid_history])
    y = np.array([b for _, b in predictor.bid_history])

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color="blue", label="Observed bids", alpha=0.6)

    # Plot this agent's bids
    if predictor.agent_bids:
        x_agent = np.array([r for r, _ in predictor.agent_bids])
        y_agent = np.array([b for _, b in predictor.agent_bids])
        plt.scatter(x_agent, y_agent, color="green", label="Agent's bids", alpha=0.6, marker='x')

    # Plot this agent's bids
    if predictor.agent_winning_bids:
        x_agent = np.array([r for r, _ in predictor.agent_winning_bids])
        y_agent = np.array([b for _, b in predictor.agent_winning_bids])
        plt.scatter(x_agent, y_agent, color="orange", label="Agent's winning bids", alpha=0.6, marker='o')

    # Plot fitted polynomial curve if available
    if predictor.coefficients is not None:
        degree = predictor.degree
        x_line = np.linspace(min(x), max(x), 200)
        y_line = np.polyval(predictor.coefficients, x_line)
        plt.plot(x_line, y_line, color="red", label=f"Fitted degree-{degree} curve")

    plt.title(f"Learned Relationship: Average Reward vs Winning Bid (degree={predictor.degree})")
    plt.xlabel("Average Reward")
    plt.ylabel("Winning Bid")
    plt.legend()
    plt.grid(True)
    plt.show()




if __name__ == "__main__":
    host = "localhost"
    agent_name = "give_me_all_the_points_{}".format(random.randint(1, 1000))
    player_id = "Raphael Najee Monteiro"
    port = 8000

    game = AuctionGameClient(host=host,
                             agent_name=agent_name,
                             player_id=player_id,
                             port=port)
    try:
        game.run(smart_bid)
    except KeyboardInterrupt:
        print("<interrupt - shutting down>")

    print("<game is done>")
    plot_learning_results(predictor)