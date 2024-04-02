# standard library imports

# third-party imports
import numpy as np
import pandas as pd

# local imports
from src.bet_sizing import SimultaneousKelly

# TODO: Refactor Matt's code lol


class BacktestFramework:
    """
    This class is for evaluating betting performance of the models
    """

    def __init__(self, data, bankroll=100):
        """Initializes BacktestFramework with a list of probabilities"""
        self.data = data  # contains the testing probabilities -- should be 915x2
        read_csv = pd.read_csv("../data/processed/backtest_odds.csv", na_values="")
        read_csv.dropna(subset=["RED_WIN"], inplace=True)
        fight_information = read_csv[
            ["EVENT_ID", "RED_FIGHTER_ODDS", "BLUE_FIGHTER_ODDS", "RED_WIN"]
        ].to_numpy()
        # Check if the shapes are compatible
        if fight_information.shape[0] == self.data.shape[0]:
            # Concatenate EVENT_ID as the first column, followed by self.data
            combined_data = np.hstack((fight_information, self.data))
            self.data = combined_data
        else:
            raise ValueError(
                "The number of rows in the CSV does not match the number of rows in `data`."
            )
        self.bankroll = bankroll

    def split_by_event_id(self):
        # Initialize a list to hold the arrays
        grouped_data = []

        # Track the current event_id and a temporary list to store rows for the current event_id
        current_event_id = None
        current_group = []

        # Iterate through each row in the combined data
        for row in self.data:
            # Check if we're still on the same event_id
            if row[0] == current_event_id:
                # If yes, append the row to the current group
                current_group.append(row)
            else:
                # If no, this means we're encountering a new event_id
                # Check if current_group is not empty (which would be the case for the first row)
                if current_group:
                    # Convert the current_group to an array and append to grouped_data
                    grouped_data.append(np.array(current_group))
                    # Reset current_group for the next event_id
                    current_group = []
                # Update the current event_id and initialize the current group with the current row
                current_event_id = row[0]
                current_group.append(row)

        # After the loop, add the last group to the list (if not empty)
        if current_group:
            grouped_data.append(np.array(current_group))

        return grouped_data

    def test_events(self):
        grouped_data = self.split_by_event_id()
        temporary_bank = self.bankroll
        loop = 1
        for group in grouped_data:
            # remember it's (EVENT_ID, RED_FIGHT_ODDS, BLUe_ODDS, RED_WIN, red_prob, blue_prob)
            print(f"testing event {loop} with bankroll == {temporary_bank}")
            kelly = SimultaneousKelly(
                group[:, 4], group[:, 5], group[:, 1], group[:, 2], temporary_bank
            )
            red_wagers, blue_wagers = kelly()
            for (
                red_wager,
                blue_wager,
                outcome,
                red_betting_odds,
                blue_betting_odds,
            ) in zip(red_wagers, blue_wagers, group[:, 3], group[:, 1], group[:, 2]):
                if outcome:
                    temporary_bank -= red_wager + blue_wager
                    if outcome == 1:
                        temporary_bank += red_wager * red_betting_odds
                    elif outcome == 0:
                        temporary_bank += blue_wager * blue_betting_odds
                    else:
                        print("problem...")

            loop += 1

        return temporary_bank
