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
4. Aggregation Summary

Project:
Adaptive Trust-Aware Federated Intrusion Detection

Dataset:
FLNET2023
"""

import torch


class TrustAggregator:

    def __init__(self):
        pass

    # =======================================================
    # Normalize Trust Scores
    # =======================================================

    def normalize_weights(
        self,
        trust_scores
    ):
        """
        Normalize trust scores so they sum to 1.
        """

        total = sum(trust_scores)

        if total == 0:

            raise ValueError(
                "Sum of trust scores cannot be zero."
            )

        return [

            score / total

            for score in trust_scores

        ]

    # =======================================================
    # Aggregate Model Parameters
    # =======================================================

    def aggregate_tensor(
        self,
        client_models,
        weights
    ):
        """
        Aggregate model tensors using weighted averaging.
        """

        if len(client_models) == 0:

            raise ValueError(
                "No client models available."
            )

        # Check tensor shapes

        expected_shape = client_models[0].shape

        for model in client_models:

            if model.shape != expected_shape:

                raise ValueError(
                    "All client tensors must have identical shapes."
                )

        global_model = torch.zeros_like(
            client_models[0]
        )

        for model, weight in zip(
            client_models,
            weights
        ):

            global_model += weight * model

        return global_model

    # =======================================================
    # Trust-Aware Aggregation
    # =======================================================

    def aggregate(
        self,
        client_models,
        trust_scores,
        blacklist=None
    ):
        """
        Perform Trust-Aware Aggregation.
        """

        included_models = []
        included_scores = []

        included_clients = []
        excluded_clients = []

        for i, (model, score) in enumerate(
            zip(client_models, trust_scores)
        ):

            client_id = f"Client_{i+1}"

            if (
                blacklist is not None
                and blacklist.is_blacklisted(client_id)
            ):

                excluded_clients.append(
                    client_id
                )

                continue

            included_models.append(model)

            included_scores.append(score)

            included_clients.append(client_id)

        print("\nIncluded Clients")
        print("-" * 30)

        if included_clients:

            for client in included_clients:

                print(client)

        else:

            print("None")

        print("\nExcluded Clients")
        print("-" * 30)

        if excluded_clients:

            for client in excluded_clients:

                print(client)

        else:

            print("None")

        if len(included_models) == 0:

            raise ValueError(
                "No trusted clients available for aggregation."
            )

        weights = self.normalize_weights(
            included_scores
        )

        print("\nFinal Trust Weights")
        print("-" * 30)

        for client, weight in zip(
            included_clients,
            weights
        ):

            print(
                f"{client} : {weight:.4f}"
            )

        return self.aggregate_tensor(
            included_models,
            weights
        )

    # =======================================================
    # Display Weights
    # =======================================================

    def print_weights(
        self,
        trust_scores
    ):
        """
        Display normalized trust weights.
        """

        weights = self.normalize_weights(
            trust_scores
        )

        print("\n")
        print("=" * 60)
        print("Normalized Trust Weights")
        print("=" * 60)

        for i, weight in enumerate(weights):

            print(
                f"Client_{i+1}"
                f" : "
                f"{weight:.4f}"
            )

    # =======================================================
    # Utility
    # =======================================================

    def __len__(self):

        return 1