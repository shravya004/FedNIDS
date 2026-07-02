"""
===========================================================
Trust Integration Module
===========================================================

Integrates all Member 2 components into one pipeline.

Components
----------
1. Adaptive Trust Score
2. Reputation Memory
3. Automatic Blacklisting
4. Trust-aware Aggregation

This module is used by the Federated Learning server.

Project:
Adaptive Trust-Aware Federated Intrusion Detection

Dataset:
FLNET2023
"""

from src.trust.trust_score import evaluate_client_trust
from src.trust.reputation import ReputationManager
from src.trust.blacklist import BlacklistManager
from src.trust.aggregation import TrustAggregator


class TrustAwareFL:
    """
    Central integration module for the complete
    Trust-Aware Federated Learning framework.
    """

    def __init__(self):
        """
        Initialize all trust management components.
        """

        self.reputation = ReputationManager()

        self.blacklist = BlacklistManager()

        self.aggregator = TrustAggregator()

    # =====================================================
    # Process One Client
    # =====================================================

    def process_client(
        self,
        client_id,
        client_embedding,
        global_embedding,
        validation_accuracy,
        current_round,
    ):
        """
        Process one federated client.

        Steps
        -----
        1. Compute trust score
        2. Update reputation
        3. Detect malicious behaviour
        4. Blacklist if necessary
        5. Return trust information
        """

        previous_reputation = self.reputation.get_reputation(
            client_id
        )

        result = evaluate_client_trust(

            client_embedding=client_embedding,

            global_embedding=global_embedding,

            validation_accuracy=validation_accuracy,

            previous_reputation=previous_reputation,

        )

        trust_score = result["trust_score"]

        # ---------------------------------------------
        # Update Reputation
        # ---------------------------------------------

        self.reputation.update(

            client_id=client_id,

            trust_score=trust_score,

            current_round=current_round,

        )

        current_reputation = self.reputation.get_reputation(
            client_id
        )

        # ---------------------------------------------
        # Automatic Blacklisting
        # ---------------------------------------------

        if (
            self.reputation.get_low_trust_count(client_id)
            >= 3
        ):

            if not self.blacklist.is_blacklisted(
                client_id
            ):

                self.blacklist.add_client(

                    client_id=client_id,

                    reason="Three Consecutive Low Trust Rounds",

                    round_number=current_round,

                    trust_score=trust_score,

                )

        blacklisted = self.blacklist.is_blacklisted(
            client_id
        )

        # ---------------------------------------------
        # Update Result Dictionary
        # ---------------------------------------------

        result["historical_reputation"] = round(
            current_reputation,
            4,
        )

        result["client_id"] = client_id

        result["round"] = current_round

        result["blacklisted"] = blacklisted

        if blacklisted:

            result["status"] = "Blacklisted"

        return result

    # =====================================================
    # Trust-aware Aggregation
    # =====================================================

    def aggregate(
        self,
        client_models,
        trust_scores,
    ):
        """
        Aggregate trusted client models.
        """

        global_model = self.aggregator.aggregate(

            client_models=client_models,

            trust_scores=trust_scores,

            blacklist=self.blacklist,

        )

        aggregation_info = self.aggregator.export_statistics()

        return global_model, aggregation_info
    # =====================================================
    # Process Multiple Clients
    # =====================================================

    def process_all_clients(
        self,
        client_results,
        global_embedding,
        current_round,
    ):
        """
        Evaluate trust for all clients
        participating in the current round.

        client_results format:

        {
            "client_id": ...,
            "embedding": ...,
            "accuracy": ...
        }
        """

        outputs = []

        for client in client_results:

            result = self.process_client(

                client_id=client["client_id"],

                client_embedding=client["embedding"],

                global_embedding=global_embedding,

                validation_accuracy=client["accuracy"],

                current_round=current_round,

            )

            outputs.append(result)

        return outputs
        # =====================================================
    # Print Reputation
    # =====================================================

    def print_reputation(self):
        """
        Print complete reputation summary.
        """

        self.reputation.print_summary()

    # =====================================================
    # Print Blacklist
    # =====================================================

    def print_blacklist(self):
        """
        Print blacklist summary.
        """

        self.blacklist.print_blacklist()

    # =====================================================
    # Statistics
    # =====================================================

    def reputation_statistics(self):
        """
        Return reputation statistics.
        """

        return self.reputation.export_statistics()

    def blacklist_statistics(self):
        """
        Return blacklist statistics.
        """

        return self.blacklist.export_blacklist()

    # =====================================================
    # Utility Functions
    # =====================================================

    def is_client_blacklisted(
        self,
        client_id,
    ):
        """
        Check whether a client is blacklisted.
        """

        return self.blacklist.is_blacklisted(
            client_id
        )

    def get_client_reputation(
        self,
        client_id,
    ):
        """
        Return the latest reputation of a client.
        """

        return self.reputation.get_reputation(
            client_id
        )

    def get_blacklisted_clients(self):
        """
        Return all blacklisted clients.
        """

        return self.blacklist.get_blacklisted_clients()

    def get_registered_clients(self):
        """
        Return all registered clients.
        """

        return self.reputation.get_all_clients()

    # =====================================================
    # Reset Framework
    # =====================================================

    def reset(self):
        """
        Reset the complete trust framework before
        starting a new experiment.
        """

        self.reputation = ReputationManager()

        self.blacklist = BlacklistManager()

        self.aggregator = TrustAggregator()

    # =====================================================
    # Framework Summary
    # =====================================================

    def summary(self):
        """
        Display overall trust framework summary.
        """

        print("\n")
        print("=" * 70)
        print("Trust Framework Summary")
        print("=" * 70)

        print(
            f"Registered Clients  : "
            f"{len(self.reputation)}"
        )

        print(
            f"Blacklisted Clients : "
            f"{len(self.blacklist)}"
        )

        print(
            f"Average Reputation  : "
            f"{self.reputation.get_global_average():.4f}"
        )

        best_client = self.reputation.get_best_client()

        if best_client is not None:

            print(
                f"Best Client         : "
                f"{best_client}"
            )

        worst_client = self.reputation.get_worst_client()

        if worst_client is not None:

            print(
                f"Worst Client        : "
                f"{worst_client}"
            )

        print("=" * 70)

    # =====================================================
    # Utility
    # =====================================================

    def __len__(self):
        """
        Return the total number of registered clients.
        """

        return len(self.reputation)