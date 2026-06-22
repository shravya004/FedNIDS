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

    def __init__(self):

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
        current_round
    ):
        """
        Compute trust score, update reputation,
        and blacklist malicious clients.
        """

        previous_reputation = self.reputation.get_reputation(
            client_id
        )

        result = evaluate_client_trust(

            client_embedding=client_embedding,

            global_embedding=global_embedding,

            validation_accuracy=validation_accuracy,

            previous_reputation=previous_reputation

        )

        trust_score = result["trust_score"]

        # ---------------------------------------------
        # Update Reputation
        # ---------------------------------------------

        self.reputation.update(

            client_id,

            trust_score

        )

        # ---------------------------------------------
        # Automatic Blacklisting
        # ---------------------------------------------

        if self.reputation.get_low_trust_count(client_id) >= 3:

            if not self.blacklist.is_blacklisted(client_id):

                self.blacklist.add_client(

                    client_id=client_id,

                    reason="Consecutive Low Trust",

                    round_number=current_round,

                    trust_score=trust_score

                )

        result["historical_reputation"] = \
            self.reputation.get_reputation(client_id)

        if self.blacklist.is_blacklisted(client_id):

            result["status"] = "Blacklisted"

        elif trust_score >= 0.75:

            result["status"] = "Trusted"

        elif trust_score >= 0.50:

            result["status"] = "Suspicious"

        else:

            result["status"] = "Malicious"

        return result

    # =====================================================
    # Aggregate Client Models
    # =====================================================

    def aggregate(
        self,
        client_models,
        trust_scores
    ):
        """
        Perform Trust-Aware Aggregation.
        """

        return self.aggregator.aggregate(

            client_models=client_models,

            trust_scores=trust_scores,

            blacklist=self.blacklist

        )

    # =====================================================
    # Print Reputation
    # =====================================================

    def print_reputation(self):

        self.reputation.print_summary()

    # =====================================================
    # Print Blacklist
    # =====================================================

    def print_blacklist(self):

        self.blacklist.print_blacklist()

    # =====================================================
    # Utility Functions
    # =====================================================

    def is_client_blacklisted(
        self,
        client_id
    ):

        return self.blacklist.is_blacklisted(
            client_id
        )

    def get_client_reputation(
        self,
        client_id
    ):

        return self.reputation.get_reputation(
            client_id
        )

    def get_blacklisted_clients(self):

        return self.blacklist.get_blacklisted_clients()

    def reset(self):
        """
        Reset the framework before a new experiment.
        """

        self.reputation = ReputationManager()

        self.blacklist = BlacklistManager()

    def summary(self):

        print("\n")
        print("=" * 65)
        print("Trust Framework Summary")
        print("=" * 65)

        print(
            f"Total Clients : {len(self.reputation)}"
        )

        print(
            f"Blacklisted   : {self.blacklist.total_blacklisted()}"
        )

        print("=" * 65)