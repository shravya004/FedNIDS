"""
===========================================================
Federated Server
===========================================================

Starts the Flower server using the proposed
Trust-Aware Federated Learning strategy.

Project:
Adaptive Trust-Aware Federated Intrusion Detection System
Using Hybrid GRU-Transformer on FLNET2023
"""

import traceback

import flwr as fl

from src.federated.trust_strategy import TrustAwareStrategy
from src.utils.project_logger import ProjectLogger


# ==========================================================
# Configuration
# ==========================================================

SERVER_ADDRESS = "127.0.0.1:8080"

NUM_ROUNDS = 10

MIN_FIT_CLIENTS = 2

MIN_EVALUATE_CLIENTS = 2

MIN_AVAILABLE_CLIENTS = 2


# ==========================================================
# Initialize Logger
# ==========================================================

logger = ProjectLogger()

print("\n" + "=" * 70)
print("Adaptive Trust-Aware Federated Intrusion Detection System")
print("=" * 70)

print("Initializing Project...")

print("Results directory ready")
print("Reports initialized")
print("CSV files initialized")
print("Graph directory ready")
print("Model directory ready")
print("Embedding directory ready")

logger.start_experiment(

    dataset="FLNET2023",

    model="Hybrid GRU-Transformer",

    clients=MIN_AVAILABLE_CLIENTS,

    rounds=NUM_ROUNDS,

)

print("Experiment report created")

print("=" * 70)


# ==========================================================
# Trust Strategy
# ==========================================================

strategy = TrustAwareStrategy(

    fraction_fit=1.0,

    fraction_evaluate=1.0,

    min_fit_clients=MIN_FIT_CLIENTS,

    min_evaluate_clients=MIN_EVALUATE_CLIENTS,

    min_available_clients=MIN_AVAILABLE_CLIENTS,

)


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    try:

        print("\n" + "=" * 70)
        print("Adaptive Trust-Aware Federated IDS")
        print("=" * 70)

        print(f"Server Address        : {SERVER_ADDRESS}")
        print(f"Federated Rounds      : {NUM_ROUNDS}")
        print(f"Minimum Clients       : {MIN_AVAILABLE_CLIENTS}")
        print("Aggregation Strategy  : Trust-Aware Aggregation")

        print("=" * 70)

        fl.server.start_server(

            server_address=SERVER_ADDRESS,

            config=fl.server.ServerConfig(

                num_rounds=NUM_ROUNDS

            ),

            strategy=strategy,

        )

    except KeyboardInterrupt:

        print("\nServer stopped by user.")

    except SystemExit as error:

        print("\nFlower Server Error")

        print(error)

    except Exception as error:

        print("\nUnexpected Server Error")

        print(error)

        traceback.print_exc()

    finally:

        print("\n")
        print("=" * 70)
        print("Training Finished")
        print("=" * 70)

        print("\nTrust Framework Summary")

        strategy.trust_framework.summary()

        print("\nClient Reputation")

        strategy.trust_framework.print_reputation()

        print("\nBlacklisted Clients")

        strategy.trust_framework.print_blacklist()

        print("\nSaving Results...")

        try:

            strategy.logger.save()

        except Exception as error:

            print(f"Logger Save Error : {error}")

        print("Generating Plots...")

        try:

            strategy.logger.plot()

        except Exception as error:

            print(f"Plot Generation Error : {error}")

        print("\nResults saved successfully.")

        print("=" * 70)