import torch

from src.trust.trust_score import evaluate_client_trust

print("=" * 60)
print("Adaptive Multi-Factor Trust Scoring Test")
print("=" * 60)

# ---------------------------------------------------
# Simulated Global Embedding
# ---------------------------------------------------

global_embedding = torch.tensor([
    0.80,
    0.30,
    0.60,
    0.10
])

# ---------------------------------------------------
# Simulated Client Data
# ---------------------------------------------------

clients = {

    "Client_1": {
        "embedding": torch.tensor([
            0.81,
            0.29,
            0.61,
            0.12
        ]),
        "accuracy": 96,
        "reputation": 0.90
    },

    "Client_2": {
        "embedding": torch.tensor([
            0.50,
            0.41,
            0.31,
            0.24
        ]),
        "accuracy": 87,
        "reputation": 0.82
    },

    "Client_3": {
        "embedding": torch.tensor([
            -0.40,
            -0.55,
            -0.10,
            -0.72
        ]),
        "accuracy": 45,
        "reputation": 0.55
    }

}

for client_name, info in clients.items():

    result = evaluate_client_trust(
        client_embedding=info["embedding"],
        global_embedding=global_embedding,
        validation_accuracy=info["accuracy"],
        previous_reputation=info["reputation"]
    )

    print(f"\n{client_name}")
    print("-" * 40)

    print(f"Embedding Similarity : {result['embedding_similarity']:.4f}")
    print(f"Validation Score     : {result['validation_score']:.4f}")
    print(f"Historical Reputation: {result['historical_reputation']:.4f}")
    print(f"Anomaly Score        : {result['anomaly_score']:.4f}")
    print(f"Trust Score          : {result['trust_score']:.4f}")
    print(f"Status               : {result['status']}")