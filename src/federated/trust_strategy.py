"""
===========================================================
Trust-Aware Flower Strategy
===========================================================

Custom Flower FedAvg strategy integrating the complete
Adaptive Trust-Aware Federated Learning framework.

Features
--------
1. Adaptive Multi-Factor Trust Scoring (AMTS)
2. Reputation Management
3. Automatic Blacklisting
4. Trust-Aware Aggregation
5. Experiment Logging
6. Performance Monitoring

Project:
Adaptive Trust-Aware Federated Intrusion Detection System

Dataset:
FLNET2023
"""

from typing import Dict, List, Tuple

import numpy as np
import torch

import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from src.trust.integration import TrustAwareFL
from src.utils.project_logger import ProjectLogger


class TrustAwareStrategy(FedAvg):
    """
    Custom Flower strategy implementing the complete
    Trust-Aware Federated Learning pipeline.
    """

    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
    ):

        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
        )

        # --------------------------------------------------
        # Trust Framework
        # --------------------------------------------------

        self.trust_framework = TrustAwareFL()

        # --------------------------------------------------
        # Logger
        # --------------------------------------------------

        self.logger = ProjectLogger()

        # --------------------------------------------------
        # Runtime Statistics
        # --------------------------------------------------

        self.current_round = 0

        self.global_accuracy = 0.0

        self.average_trust = 0.0

        self.global_model = None

        self.client_results = []

        self.trust_scores = []

        self.aggregation_info = {}

        # --------------------------------------------------
        # Experiment Statistics
        # --------------------------------------------------

        self.total_clients_processed = 0

        self.total_blacklisted = 0

        self.round_statistics = []

        print("\n" + "=" * 70)
        print("Trust-Aware Federated Learning Strategy Initialized")
        print("=" * 70)

        print(f"Minimum Fit Clients       : {min_fit_clients}")
        print(f"Minimum Evaluate Clients  : {min_evaluate_clients}")
        print(f"Minimum Available Clients : {min_available_clients}")

        print("=" * 70)

        # =====================================================
    # Aggregate Client Models
    # =====================================================

    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ):
        """
        Aggregate client updates using the complete
        Trust-Aware Federated Learning framework.
        """

        if not results:

            return None, {}

        self.current_round = server_round

        print("\n" + "=" * 70)
        print(f"FEDERATED ROUND {server_round}")
        print("=" * 70)

        self.logger.start_round(server_round)

        # --------------------------------------------------
        # Storage
        # --------------------------------------------------

        client_models = []

        client_embeddings = []

        client_information = []

        self.trust_scores = []

        self.client_results = []

        # --------------------------------------------------
        # Pass 1
        # Collect models, metrics and embeddings
        # --------------------------------------------------

        for index, (_, fit_res) in enumerate(results):

            metrics = fit_res.metrics

            client_id = metrics.get(

                "client_id",

                f"Client_{index + 1}"

            )

            loss = float(

                metrics.get(

                    "loss",

                    0.0

                )

            )

            accuracy = float(

                metrics.get(

                    "accuracy",

                    0.0

                )

            )

            precision = float(

                metrics.get(

                    "precision",

                    0.0

                )

            )

            recall = float(

                metrics.get(

                    "recall",

                    0.0

                )

            )

            f1 = float(

                metrics.get(

                    "f1",

                    0.0

                )

            )

            embedding_path = metrics.get(

                "embedding_path",

                None

            )

            if embedding_path is None:

                raise ValueError(

                    f"Embedding not received from {client_id}"

                )

            # ------------------------------------------
            # Load Client Embedding
            # ------------------------------------------

            client_embedding = torch.load(

                embedding_path,

                map_location="cpu"

            )

            client_embeddings.append(

                client_embedding

            )

            # ------------------------------------------
            # Store Client Model
            # ------------------------------------------

            client_models.append(

                parameters_to_ndarrays(

                    fit_res.parameters

                )

            )

            # ------------------------------------------
            # Store Client Information
            # ------------------------------------------

            client_information.append(

                {

                    "client_id": client_id,

                    "loss": loss,

                    "accuracy": accuracy,

                    "precision": precision,

                    "recall": recall,

                    "f1": f1,

                    "embedding": client_embedding,

                }

            )
            # --------------------------------------------------
        # Compute Global Embedding
        # --------------------------------------------------

        global_embedding = torch.stack(

            client_embeddings,

            dim=0,

        ).mean(dim=0)

        print("\n")
        print("=" * 70)
        print("Adaptive Multi-Factor Trust Evaluation")
        print("=" * 70)

        # --------------------------------------------------
        # Pass 2
        # Evaluate Trust For Every Client
        # --------------------------------------------------

        for client in client_information:

            result = self.trust_framework.process_client(

                client_id=client["client_id"],

                client_embedding=client["embedding"],

                global_embedding=global_embedding,

                validation_accuracy=client["accuracy"],

                current_round=server_round,

            )

            trust_score = result["trust_score"]

            reputation = result["historical_reputation"]

            status = result["status"]

            blacklisted = result["blacklisted"]

            self.trust_scores.append(

                trust_score

            )

            self.client_results.append(

                result

            )

            # ------------------------------------------
            # Logger
            # ------------------------------------------

            self.logger.log_metrics_csv(
                round_num=server_round,
                client=client["client_id"],
                loss=client["loss"],
                accuracy=client["accuracy"],
                precision=client["precision"],
                recall=client["recall"],
                f1=client["f1"],
            )

            self.logger.log_trust(

                round_num=server_round,

                client=client["client_id"],

                embedding_similarity=result["embedding_similarity"],

                validation_score=result["validation_score"],

                anomaly_score=result["anomaly_score"],

                reputation=result["historical_reputation"],

                trust_score=result["trust_score"],

                confidence=result["confidence"],

                status=result["status"],

            )

            self.logger.log_trust_csv(
                round_num=server_round,
                client=client["client_id"],
                embedding_similarity=result["embedding_similarity"],
                validation_score=result["validation_score"],
                anomaly_score=result["anomaly_score"],
                reputation=result["historical_reputation"],
                trust_score=result["trust_score"],
                confidence=result["confidence"],
                status=result["status"],
            )

            if blacklisted:

                self.logger.log_blacklist(

                    round_num=server_round,

                    client=client["client_id"],

                    trust_score=trust_score,

                    reason="Low Trust Score",

                )

            print(f"\n{client['client_id']}")

            print("-" * 45)

            print(
                f"Accuracy             : "
                f"{client['accuracy']:.4f}"
            )

            print(
                f"Precision            : "
                f"{client['precision']:.4f}"
            )

            print(
                f"Recall               : "
                f"{client['recall']:.4f}"
            )

            print(
                f"F1 Score             : "
                f"{client['f1']:.4f}"
            )

            print(
                f"Trust Score          : "
                f"{trust_score:.4f}"
            )

            print(
                f"Historical Reputation: "
                f"{reputation:.4f}"
            )

            print(
                f"Status               : "
                f"{status}"
            )

            if blacklisted:

                print(
                    "Client added to blacklist."
                )

        print("\n")
        print("=" * 70)
        print("Trust Evaluation Completed")
        print("=" * 70)

            # --------------------------------------------------
        # Trust-Aware Model Aggregation
        # --------------------------------------------------

        print("\n")
        print("=" * 70)
        print("Trust-Aware Model Aggregation")
        print("=" * 70)

        aggregated_weights = []

        total_layers = len(client_models[0])

        # --------------------------------------------------
        # Aggregate Every Layer
        # --------------------------------------------------

        for layer in range(total_layers):

            layer_models = []

            for model in client_models:

                layer_models.append(

                    torch.tensor(

                        model[layer],

                        dtype=torch.float32,

                    )

                )

            # ------------------------------------------
            # Trust-Aware Aggregation
            # ------------------------------------------

            global_layer, aggregation_info = (

                self.trust_framework.aggregate(

                    client_models=layer_models,

                    trust_scores=self.trust_scores,

                )

            )

            aggregated_weights.append(

                global_layer.cpu().numpy()

            )

        # --------------------------------------------------
        # Save Aggregation Statistics
        # --------------------------------------------------

        self.aggregation_info = aggregation_info

        self.global_model = aggregated_weights

        print("\nAggregation Summary")

        print("-" * 50)

        for key, value in aggregation_info.items():

            print(f"{key} : {value}")

        print("=" * 70)

            # --------------------------------------------------
        # Convert Aggregated Weights
        # --------------------------------------------------

        parameters = ndarrays_to_parameters(

            aggregated_weights

        )

        # --------------------------------------------------
        # Global Statistics
        # --------------------------------------------------

        self.global_accuracy = sum(

            client["accuracy"]

            for client in client_information

        ) / len(client_information)

        self.average_trust = sum(

            self.trust_scores

        ) / len(self.trust_scores)

        self.total_clients_processed += len(

            client_information

        )

        self.total_blacklisted = len(

            self.trust_framework.get_blacklisted_clients()

        )

        # --------------------------------------------------
        # Store Round Statistics
        # --------------------------------------------------

        self.round_statistics.append(

            {

                "round": server_round,

                "accuracy": self.global_accuracy,

                "average_trust": self.average_trust,

                "blacklisted": self.total_blacklisted,

            }

        )

        # --------------------------------------------------
        # Logger
        # --------------------------------------------------

        self.logger.end_round(

            round_num=server_round,

            global_accuracy=self.global_accuracy,

            average_trust=self.average_trust,

        )

        print("\n")
        print("=" * 70)
        print("Round Summary")
        print("=" * 70)

        print(

            f"Global Accuracy     : "

            f"{self.global_accuracy:.4f}"

        )

        print(

            f"Average Trust Score : "

            f"{self.average_trust:.4f}"

        )

        print(

            f"Registered Clients  : "

            f"{len(self.trust_framework)}"

        )

        print(

            f"Blacklisted Clients : "

            f"{self.total_blacklisted}"

        )

        print("=" * 70)

        return parameters, {}
    
        # =====================================================
    # Aggregate Evaluation
    # =====================================================

    def aggregate_evaluate(
        self,
        server_round,
        results,
        failures,
    ):
        """
        Aggregate evaluation metrics.
        """

        if not results:

            return None, {}

        loss = sum(

            evaluate_res.loss

            for _, evaluate_res in results

        ) / len(results)

        metrics = {

            "accuracy": sum(

                evaluate_res.metrics.get(

                    "accuracy",

                    0.0,

                )

                for _, evaluate_res in results

            ) / len(results),

            "precision": sum(

                evaluate_res.metrics.get(

                    "precision",

                    0.0,

                )

                for _, evaluate_res in results

            ) / len(results),

            "recall": sum(

                evaluate_res.metrics.get(

                    "recall",

                    0.0,

                )

                for _, evaluate_res in results

            ) / len(results),

            "f1": sum(

                evaluate_res.metrics.get(

                    "f1",

                    0.0,

                )

                for _, evaluate_res in results

            ) / len(results),

        }

        print("\n")
        print("=" * 70)
        print("Global Evaluation")
        print("=" * 70)

        print(f"Loss      : {loss:.4f}")
        print(f"Accuracy  : {metrics['accuracy']:.4f}")
        print(f"Precision : {metrics['precision']:.4f}")
        print(f"Recall    : {metrics['recall']:.4f}")
        print(f"F1 Score  : {metrics['f1']:.4f}")

        print("=" * 70)

        return loss, metrics

    # =====================================================
    # Central Evaluation
    # =====================================================

    def evaluate(
        self,
        server_round,
        parameters,
    ):
        """
        No centralized evaluation.
        """

        return None

    # =====================================================
    # Framework Summary
    # =====================================================

    def summary(self):
        """
        Print complete trust framework summary.
        """

        print("\n")
        print("=" * 70)
        print("Trust-Aware Federated Learning Summary")
        print("=" * 70)

        print(

            f"Rounds Completed      : "

            f"{self.current_round}"

        )

        print(

            f"Clients Processed     : "

            f"{self.total_clients_processed}"

        )

        print(

            f"Blacklisted Clients   : "

            f"{self.total_blacklisted}"

        )

        print(

            f"Average Trust Score   : "

            f"{self.average_trust:.4f}"

        )

        print(

            f"Global Accuracy       : "

            f"{self.global_accuracy:.4f}"

        )

        print("=" * 70)

        self.trust_framework.summary()

    # =====================================================
    # Reset Strategy
    # =====================================================

    def reset(self):
        """
        Reset the strategy for a new experiment.
        """

        self.current_round = 0

        self.global_accuracy = 0.0

        self.average_trust = 0.0

        self.global_model = None

        self.client_results.clear()

        self.trust_scores.clear()

        self.round_statistics.clear()

        self.total_clients_processed = 0

        self.total_blacklisted = 0

        self.aggregation_info = {}

        self.trust_framework.reset()

        self.logger = ProjectLogger()

    # =====================================================
    # Utility Functions
    # =====================================================

    def get_trust_scores(self):

        return self.trust_scores

    def get_round_statistics(self):

        return self.round_statistics

    def get_blacklisted_clients(self):

        return self.trust_framework.get_blacklisted_clients()

    def get_reputation_statistics(self):

        return self.trust_framework.reputation_statistics()

    def get_blacklist_statistics(self):

        return self.trust_framework.blacklist_statistics()

    def __len__(self):

        return len(self.trust_framework)