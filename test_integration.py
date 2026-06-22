import torch

from src.trust.integration import TrustAwareFL

print("=" * 65)
print("Adaptive Trust Integration Test")
print("=" * 65)

# =====================================================
# Create Trust Framework
# =====================================================

manager = TrustAwareFL()

# =====================================================
# Global Embedding
# =====================================================

global_embedding = torch.tensor(
    [
        0.80,
        0.30,
        0.60,
        0.10
    ],
    dtype=torch.float32
)

# =====================================================
# Simulated Clients
# =====================================================

clients = {

    "Client_1": {

        "embedding": torch.tensor(
            [
                0.81,
                0.31,
                0.62,
                0.11
            ],
            dtype=torch.float32
        ),

        "accuracy": 96

    },

    "Client_2": {

        "embedding": torch.tensor(
            [
                0.55,
                0.38,
                0.42,
                0.18
            ],
            dtype=torch.float32
        ),

        "accuracy": 87

    },

    "Client_3": {

        "embedding": torch.tensor(
            [
                -0.60,
                -0.45,
                -0.55,
                -0.72
            ],
            dtype=torch.float32
        ),

        "accuracy": 42

    }

}

# =====================================================
# Simulate FL Rounds
# =====================================================

TOTAL_ROUNDS = 5

for round_number in range(1, TOTAL_ROUNDS + 1):

    print("\n")
    print("=" * 65)
    print(f"Communication Round {round_number}")
    print("=" * 65)

    for client_id, info in clients.items():

        result = manager.process_client(

            client_id=client_id,

            client_embedding=info["embedding"],

            global_embedding=global_embedding,

            validation_accuracy=info["accuracy"],

            current_round=round_number

        )

        print(f"\nClient : {client_id}")

        print("-" * 45)

        print(
            f"Trust Score           : "
            f"{result['trust_score']:.4f}"
        )

        print(
            f"Historical Reputation : "
            f"{result['historical_reputation']:.4f}"
        )

        print(
            f"Status                : "
            f"{result['status']}"
        )

# =====================================================
# Aggregation Test
# =====================================================

print("\n")
print("=" * 65)
print("Testing Trust-Aware Aggregation")
print("=" * 65)

client_models = [

    torch.tensor(
        [1.0, 2.0, 3.0],
        dtype=torch.float32
    ),

    torch.tensor(
        [2.0, 3.0, 4.0],
        dtype=torch.float32
    ),

    torch.tensor(
        [10.0, 10.0, 10.0],
        dtype=torch.float32
    )

]

trust_scores = [

    manager.get_client_reputation("Client_1"),

    manager.get_client_reputation("Client_2"),

    manager.get_client_reputation("Client_3")

]

global_model = manager.aggregate(

    client_models=client_models,

    trust_scores=trust_scores

)

print("\nAggregated Global Model")

print(global_model)

# =====================================================
# Reputation Summary
# =====================================================

print("\n")
print("=" * 65)
print("Reputation Summary")
print("=" * 65)

manager.print_reputation()

# =====================================================
# Blacklist Summary
# =====================================================

print("\n")
print("=" * 65)
print("Blacklist Summary")
print("=" * 65)

manager.print_blacklist()

# =====================================================
# Framework Summary
# =====================================================

print("\n")
print("=" * 65)
print("Framework Summary")
print("=" * 65)

manager.summary()

print("\n")
print("=" * 65)
print("Integration Test Completed Successfully")
print("=" * 65)