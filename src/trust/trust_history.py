"""
===========================================================
Trust History Module
===========================================================

Maintains complete trust history of every client
throughout federated learning communication rounds.

Features
--------
1. Store trust evolution
2. Store reputation evolution
3. Store client status
4. Store blacklist history
5. Retrieve client history
6. Export complete history

Project:
Adaptive Trust-Aware Federated Intrusion Detection

Dataset:
FLNET2023
"""

from collections import defaultdict


class TrustHistory:

    def __init__(self):
        """
        Initialize Trust History.
        """

        self.history = defaultdict(list)

    # ------------------------------------------------------

    def add_record(
        self,
        round_number,
        client_id,
        trust_score,
        reputation,
        status,
        blacklisted
    ):
        """
        Store one communication round.
        """

        self.history[client_id].append({

            "round": round_number,

            "trust_score": trust_score,

            "reputation": reputation,

            "status": status,

            "blacklisted": blacklisted

        })

    # ------------------------------------------------------

    def get_client_history(
        self,
        client_id
    ):
        """
        Return complete history of one client.
        """

        return self.history.get(client_id, [])

    # ------------------------------------------------------

    def get_all_clients(self):
        """
        Return all registered clients.
        """

        return list(self.history.keys())

    # ------------------------------------------------------

    def get_round_history(
        self,
        round_number
    ):
        """
        Return all client records for a given round.
        """

        records = []

        for client in self.history:

            for item in self.history[client]:

                if item["round"] == round_number:

                    record = item.copy()

                    record["client_id"] = client

                    records.append(record)

        return records

    # ------------------------------------------------------

    def export_history(self):
        """
        Export complete history.
        """

        return dict(self.history)

    # ------------------------------------------------------

    def print_client_history(
        self,
        client_id
    ):
        """
        Display one client's trust evolution.
        """

        history = self.get_client_history(client_id)

        if len(history) == 0:

            print("No history available.")

            return

        print("\n")
        print("=" * 65)
        print(f"Trust History : {client_id}")
        print("=" * 65)

        for item in history:

            print(
                f"Round {item['round']}"
            )

            print(
                f"  Trust Score : {item['trust_score']:.4f}"
            )

            print(
                f"  Reputation  : {item['reputation']:.4f}"
            )

            print(
                f"  Status      : {item['status']}"
            )

            print(
                f"  Blacklisted : {item['blacklisted']}"
            )

            print()

    # ------------------------------------------------------

    def print_all(self):
        """
        Display complete history.
        """

        for client in self.get_all_clients():

            self.print_client_history(client)

    # ------------------------------------------------------

    def reset(self):
        """
        Clear history.
        """

        self.history.clear()

    # ------------------------------------------------------

    def __len__(self):

        return len(self.history)