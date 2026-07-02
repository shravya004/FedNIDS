"""
===========================================================
Trust-Aware Aggregation Module
===========================================================

Aggregates trusted client models using
Adaptive Trust Scores.

Features
--------
1. Trust Score Normalization
2. Weighted Model Aggregation
3. Blacklist Filtering
4. Aggregation Statistics
5. Export Aggregation Report

Project:
Adaptive Trust-Aware Federated Intrusion Detection

Dataset:
FLNET2023
"""

import torch


class TrustAggregator:

    def __init__(self):
        """
        Initialize Trust Aggregator.
        """

        self.last_weights = []

        self.included_clients = []

        self.excluded_clients = []

        self.last_statistics = {}

    # =======================================================
    # Normalize Trust Scores
    # =======================================================

    def normalize_weights(
        self,
        trust_scores,
    ):
        """
        Normalize trust scores so they sum to one.
        """

        if len(trust_scores) == 0:

            raise ValueError(
                "No trust scores provided."
            )

        total = sum(trust_scores)

        if total <= 0:

            raise ValueError(
                "Sum of trust scores must be positive."
            )

        return [

            score / total

            for score in trust_scores

        ]

    # =======================================================
    # Aggregate One Layer
    # =======================================================

    def aggregate_tensor(
        self,
        client_models,
        weights,
    ):
        """
        Aggregate one model layer.
        """

        if len(client_models) == 0:

            raise ValueError(
                "No client models available."
            )

        if len(client_models) != len(weights):

            raise ValueError(
                "Number of models and weights must match."
            )

        expected_shape = client_models[0].shape

        for tensor in client_models:

            if tensor.shape != expected_shape:

                raise ValueError(
                    "Client tensor shapes do not match."
                )

        global_tensor = torch.zeros_like(
            client_models[0]
        )

        for tensor, weight in zip(
            client_models,
            weights,
        ):

            global_tensor += tensor * weight

        return global_tensor

    # =======================================================
    # Trust-Aware Aggregation
    # =======================================================

    def aggregate(
        self,
        client_models,
        trust_scores,
        blacklist=None,
    ):
        """
        Perform trust-aware aggregation.
        """

        if len(client_models) != len(trust_scores):

            raise ValueError(
                "Models and trust scores must have same length."
            )

        included_models = []
        included_scores = []

        self.included_clients = []
        self.excluded_clients = []

        for i, (model, score) in enumerate(

            zip(client_models, trust_scores)

        ):

            client_id = f"Client_{i+1}"

            if (

                blacklist is not None

                and blacklist.is_blacklisted(client_id)

            ):

                self.excluded_clients.append(
                    client_id
                )

                continue

            included_models.append(model)

            included_scores.append(score)

            self.included_clients.append(
                client_id
            )

        if len(included_models) == 0:

            raise ValueError(
                "No trusted clients available."
            )

        weights = self.normalize_weights(
            included_scores
        )

        self.last_weights = weights

        aggregated = self.aggregate_tensor(

            included_models,

            weights,

        )

        self.last_statistics = {

            "included_clients":
                self.included_clients,

            "excluded_clients":
                self.excluded_clients,

            "trust_weights":
                weights,

            "num_clients":
                len(client_models),

            "num_trusted":
                len(self.included_clients),

            "num_blacklisted":
                len(self.excluded_clients),

        }

        return aggregated

    # =======================================================
    # Print Aggregation Summary
    # =======================================================

    def print_summary(self):
        """
        Print aggregation summary.
        """

        print("\n")
        print("=" * 65)
        print("Aggregation Summary")
        print("=" * 65)

        print(
            f"Participating Clients : "
            f"{len(self.included_clients)}"
        )

        print(
            f"Excluded Clients      : "
            f"{len(self.excluded_clients)}"
        )

        print("\nIncluded Clients")

        for client, weight in zip(

            self.included_clients,

            self.last_weights,

        ):

            print(

                f"{client:<12}"

                f"Weight : {weight:.4f}"

            )

        if len(self.excluded_clients):

            print("\nExcluded Clients")

            for client in self.excluded_clients:

                print(client)

        print("=" * 65)

    # =======================================================
    # Export Statistics
    # =======================================================

    def export_statistics(self):
        """
        Export aggregation statistics.
        """

        return self.last_statistics

    # =======================================================
    # Utility
    # =======================================================

    def __len__(self):

        return len(self.included_clients)