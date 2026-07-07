"""
===========================================================
Trust Statistics Module
===========================================================

Computes statistical analysis of trust scores for every
communication round.

Features
--------
1. Average Trust Score
2. Maximum Trust Score
3. Minimum Trust Score
4. Standard Deviation
5. Client Status Counts
6. Export Statistics

Project:
Adaptive Trust-Aware Federated Intrusion Detection

Dataset:
FLNET2023
"""

import statistics


class TrustStatistics:

    def __init__(self):
        """
        Initialize Trust Statistics.
        """

        self.history = []

    # ------------------------------------------------------

    def calculate(
        self,
        round_number,
        client_results
    ):
        """
        Compute statistics for one communication round.

        client_results:
        [
            {
                "client_id": "...",
                "trust_score": ...,
                "status": "...",
                "blacklisted": ...
            },
            ...
        ]
        """

        if len(client_results) == 0:

            raise ValueError(
                "No client results available."
            )

        trust_scores = [

            client["trust_score"]

            for client in client_results

        ]

        average = statistics.mean(
            trust_scores
        )

        maximum = max(
            trust_scores
        )

        minimum = min(
            trust_scores
        )

        if len(trust_scores) > 1:

            std_dev = statistics.stdev(
                trust_scores
            )

        else:

            std_dev = 0.0

        trusted = sum(

            1

            for client in client_results

            if client["status"] == "Trusted"

        )

        reliable = sum(

            1

            for client in client_results

            if client["status"] == "Reliable"

        )

        suspicious = sum(

            1

            for client in client_results

            if client["status"] == "Suspicious"

        )

        malicious = sum(

            1

            for client in client_results

            if client["status"] == "Malicious"

        )

        blacklisted = sum(

            1

            for client in client_results

            if client["blacklisted"]

        )

        result = {

            "round": round_number,

            "average_trust": round(
                average,
                4
            ),

            "maximum_trust": round(
                maximum,
                4
            ),

            "minimum_trust": round(
                minimum,
                4
            ),

            "std_dev": round(
                std_dev,
                4
            ),

            "trusted": trusted,

            "reliable": reliable,

            "suspicious": suspicious,

            "malicious": malicious,

            "blacklisted": blacklisted

        }

        self.history.append(
            result
        )

        return result

    # ------------------------------------------------------

    def get_history(self):
        """
        Return statistics from all rounds.
        """

        return self.history

    # ------------------------------------------------------

    def latest(self):
        """
        Return latest communication round statistics.
        """

        if len(self.history) == 0:

            return None

        return self.history[-1]

    # ------------------------------------------------------

    def print_latest(self):
        """
        Display latest statistics.
        """

        stats = self.latest()

        if stats is None:

            print("No statistics available.")

            return

        print("\n")
        print("=" * 65)
        print(
            f"Communication Round {stats['round']}"
        )
        print("=" * 65)

        print(
            f"Average Trust : "
            f"{stats['average_trust']:.4f}"
        )

        print(
            f"Maximum Trust : "
            f"{stats['maximum_trust']:.4f}"
        )

        print(
            f"Minimum Trust : "
            f"{stats['minimum_trust']:.4f}"
        )

        print(
            f"Std Deviation : "
            f"{stats['std_dev']:.4f}"
        )

        print()

        print(
            f"Trusted      : "
            f"{stats['trusted']}"
        )

        print(
            f"Reliable     : "
            f"{stats['reliable']}"
        )

        print(
            f"Suspicious   : "
            f"{stats['suspicious']}"
        )

        print(
            f"Malicious    : "
            f"{stats['malicious']}"
        )

        print(
            f"Blacklisted  : "
            f"{stats['blacklisted']}"
        )

    # ------------------------------------------------------

    def reset(self):
        """
        Clear all stored statistics.
        """

        self.history.clear()

    # ------------------------------------------------------

    def __len__(self):

        return len(
            self.history
        )