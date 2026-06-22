"""
===========================================================
Reputation Manager
===========================================================

Maintains historical reputation information
for every federated client.

Features
--------
1. Current Reputation
2. Trust History
3. Average Reputation
4. Communication Rounds
5. Consecutive Low Trust Counter
6. Reputation Summary

Project:
Adaptive Trust-Aware Federated Intrusion Detection

Dataset:
FLNET2023
"""

from collections import defaultdict


class ReputationManager:

    def __init__(self):
        """
        Initialize the Reputation Manager.
        """

        # Current reputation score of each client
        self.reputation = {}

        # Complete trust history
        self.history = defaultdict(list)

        # Number of communication rounds
        self.rounds = defaultdict(int)

        # Consecutive low trust counter
        self.low_trust_counter = defaultdict(int)

    # ------------------------------------------------------

    def initialize_client(
        self,
        client_id,
        initial_reputation=0.80
    ):
        """
        Register a new client.
        """

        if client_id not in self.reputation:

            self.reputation[client_id] = initial_reputation

    # ------------------------------------------------------

    def update(
        self,
        client_id,
        trust_score
    ):
        """
        Update client reputation after one FL round.
        """

        self.initialize_client(client_id)

        self.reputation[client_id] = trust_score

        self.history[client_id].append(trust_score)

        self.rounds[client_id] += 1

        # Consecutive low trust tracking
        if trust_score < 0.40:

            self.low_trust_counter[client_id] += 1

        else:

            self.low_trust_counter[client_id] = 0

    # ------------------------------------------------------

    def get_reputation(
        self,
        client_id
    ):
        """
        Return latest reputation.
        """

        return self.reputation.get(client_id, 0.0)

    # ------------------------------------------------------

    def get_history(
        self,
        client_id
    ):
        """
        Return trust history.
        """

        return self.history.get(client_id, [])

    # ------------------------------------------------------

    def get_average_reputation(
        self,
        client_id
    ):
        """
        Compute average reputation.
        """

        history = self.get_history(client_id)

        if len(history) == 0:

            return 0.0

        return sum(history) / len(history)

    # ------------------------------------------------------

    def get_rounds(
        self,
        client_id
    ):
        """
        Return communication rounds.
        """

        return self.rounds.get(client_id, 0)

    # ------------------------------------------------------

    def get_low_trust_count(
        self,
        client_id
    ):
        """
        Return consecutive low trust count.
        """

        return self.low_trust_counter.get(client_id, 0)

    # ------------------------------------------------------

    def reset_client(
        self,
        client_id,
        initial_reputation=0.80
    ):
        """
        Reset one client's reputation.
        """

        self.reputation[client_id] = initial_reputation

        self.history[client_id].clear()

        self.rounds[client_id] = 0

        self.low_trust_counter[client_id] = 0

    # ------------------------------------------------------

    def remove_client(
        self,
        client_id
    ):
        """
        Remove client completely.
        """

        self.reputation.pop(client_id, None)

        self.history.pop(client_id, None)

        self.rounds.pop(client_id, None)

        self.low_trust_counter.pop(client_id, None)

    # ------------------------------------------------------

    def get_all_clients(self):
        """
        Return all registered clients.
        """

        return list(self.reputation.keys())

    # ------------------------------------------------------

    def reputation_summary(self):
        """
        Return reputation statistics of every client.
        Useful for Member 3's experiments.
        """

        summary = {}

        for client in self.get_all_clients():

            summary[client] = {

                "reputation": self.get_reputation(client),

                "average": self.get_average_reputation(client),

                "rounds": self.get_rounds(client),

                "low_trust": self.get_low_trust_count(client)

            }

        return summary

    # ------------------------------------------------------

    def print_summary(self):
        """
        Print complete reputation summary.
        """

        print("\n")
        print("=" * 65)
        print("Reputation Summary")
        print("=" * 65)

        if len(self.reputation) == 0:

            print("No registered clients.")

            return

        for client in self.get_all_clients():

            print(f"\nClient : {client}")

            print("-" * 45)

            print(
                f"Current Reputation : "
                f"{self.get_reputation(client):.4f}"
            )

            print(
                f"Average Reputation : "
                f"{self.get_average_reputation(client):.4f}"
            )

            print(
                f"Communication Rounds : "
                f"{self.get_rounds(client)}"
            )

            print(
                f"Low Trust Counter : "
                f"{self.get_low_trust_count(client)}"
            )

    # ------------------------------------------------------

    def __len__(self):
        """
        Return number of registered clients.
        """

        return len(self.reputation)