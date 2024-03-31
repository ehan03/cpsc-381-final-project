# standard library imports
import itertools
from typing import Tuple

# third party imports
import cvxpy as cp
import numpy as np

# local imports


class SimultaneousKelly:
    """
    Simultaneous Kelly bet sizing strategy for multiple bouts
    including a risk free asset
    """

    def __init__(
        self,
        red_probs: np.ndarray,
        blue_probs: np.ndarray,
        red_odds: np.ndarray,  # Decimal odds
        blue_odds: np.ndarray,  # Decimal odds
        current_bankroll: float,
        fraction: float = 0.10,
        min_bet: float = 0.10,
    ):
        """
        Initialize the SimultaneousKelly object
        """

        self.red_probs = red_probs
        self.blue_probs = blue_probs
        self.red_odds = red_odds
        self.blue_odds = blue_odds
        self.current_bankroll = current_bankroll
        self.fraction = fraction  # Default is 1/10
        self.min_bet = min_bet  # DraftKings requires a minimum $0.10 bet

        self.n = len(red_probs)
        self.variations = np.array(list(itertools.product([1, 0], repeat=self.n)))

    def create_returns_matrix(self) -> np.ndarray:
        """
        Create returns matrix R
        """

        returns_matrix = np.zeros(shape=(self.variations.shape[0], 2 * self.n + 1))
        returns_matrix[:, -1] = 1
        for j in range(self.n):
            returns_matrix[:, 2 * j] = np.where(
                self.variations[:, j] == 1, self.red_odds[j], 0
            )
            returns_matrix[:, 2 * j + 1] = np.where(
                self.variations[:, j] == 0, self.blue_odds[j], 0
            )

        return returns_matrix

    def create_probabilities_vector(self) -> np.ndarray:
        """
        Create probabilities vector p, contains probability combinations
        for all possible overall event outcomes
        """

        prob_vector = np.ones(shape=(1, self.variations.shape[0]))
        for j in range(self.n):
            prob_vector[0, :] = np.where(
                self.variations[:, j] == 1,
                prob_vector * self.red_probs[j],
                prob_vector * self.blue_probs[j],
            )

        return prob_vector

    def calculate_optimal_wagers(self) -> np.ndarray:
        """
        Calculate optimal fractions
        """

        R = self.create_returns_matrix()
        p = self.create_probabilities_vector()
        b = cp.Variable(2 * self.n + 1)

        objective = cp.Maximize(p @ cp.log(R @ b))
        constraints = [
            b >= 0,
            cp.sum(b) == 1,
        ]
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.CLARABEL)
            return b.value
        except:
            return np.zeros(2 * self.n + 1)

    def __call__(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate optimal wager amounts in dollars
        """

        fractions = self.calculate_optimal_wagers()
        wagers = self.fraction * self.current_bankroll * fractions[:-1]
        wagers_rounded = np.round(wagers, 2)
        wagers_clipped = np.where(wagers_rounded < self.min_bet, 0, wagers_rounded)

        red_wagers, blue_wagers = wagers_clipped[::2], wagers_clipped[1::2]

        return red_wagers, blue_wagers
