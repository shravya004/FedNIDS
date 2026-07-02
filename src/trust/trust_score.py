"""
===========================================================
Adaptive Multi-Factor Trust Scoring (AMTS)
===========================================================

Computes the trust score for each Federated Learning client
using four trust factors:

1. Embedding Similarity
2. Validation Performance
3. Historical Reputation
4. Anomaly Score

Project:
Adaptive Trust-Aware Federated Intrusion Detection

Dataset:
FLNET2023
"""

import torch
import torch.nn.functional as F

# ===========================================================
# Trust Weights
# ===========================================================

TRUST_WEIGHTS = {

    "embedding":0.30,

    "validation":0.30,

    "reputation":0.20,

    "anomaly":0.20

}

EMBEDDING_WEIGHT = TRUST_WEIGHTS["embedding"]
VALIDATION_WEIGHT = TRUST_WEIGHTS["validation"]
REPUTATION_WEIGHT = TRUST_WEIGHTS["reputation"]
ANOMALY_WEIGHT = TRUST_WEIGHTS["anomaly"]

TOTAL_WEIGHT = (
    EMBEDDING_WEIGHT
    + VALIDATION_WEIGHT
    + REPUTATION_WEIGHT
    + ANOMALY_WEIGHT
)

assert abs(TOTAL_WEIGHT - 1.0) < 1e-6, \
    "Trust weights must sum to 1.0"

# ===========================================================
# Embedding Similarity
# ===========================================================

def calculate_embedding_similarity(
    client_embedding: torch.Tensor,
    global_embedding: torch.Tensor,
) -> float:
    """
    Computes cosine similarity between client embedding
    and global embedding.

    Returns:
        float in range [0,1]
    """

    client_embedding = client_embedding.float()
    global_embedding = global_embedding.float()

    if client_embedding.shape != global_embedding.shape:

        raise ValueError(
            "Embedding dimensions do not match."
        )

    similarity = F.cosine_similarity(
        client_embedding.unsqueeze(0),
        global_embedding.unsqueeze(0)
    ).item()

    similarity = (similarity + 1.0) / 2.0

    similarity = max(0.0, min(1.0, similarity))

    return float(similarity)


# ===========================================================
# Validation Score
# ===========================================================

def calculate_validation_score(
    validation_accuracy: float
) -> float:

    validation_accuracy = max(
        0.0,
        min(validation_accuracy, 1.0)
    )

    return float(validation_accuracy)


# ===========================================================
# Historical Reputation
# ===========================================================

def update_reputation(
    previous_reputation: float,
    current_score: float,
    beta: float = 0.80
) -> float:
    """
    Exponential Moving Average (EMA)
    """

    reputation = (
        beta * previous_reputation
        +
        (1.0 - beta) * current_score
    )

    reputation = max(0.0, min(1.0, reputation))

    return float(reputation)


# ===========================================================
# Anomaly Score
# ===========================================================

def calculate_anomaly_score(
    client_embedding: torch.Tensor,
    global_embedding: torch.Tensor
) -> float:
    """
    Computes anomaly score using Euclidean distance.

    Smaller distance -> Higher trust.
    """

    client_embedding = client_embedding.float()
    global_embedding = global_embedding.float()

    if client_embedding.shape != global_embedding.shape:

        raise ValueError(
            "Embedding dimensions do not match."
        )

    distance = torch.norm(
        client_embedding - global_embedding,
        p=2
    ).item()

    # Prevent extremely large distances from dominating
    distance = min(distance, 100.0)

    anomaly_score = 1.0 / (1.0 + distance)

    anomaly_score = max(
        0.0,
        min(anomaly_score, 1.0)
    )

    return float(anomaly_score)

# ===========================================================
# Trust Score
# ===========================================================

def calculate_trust_score(
    embedding_similarity: float,
    validation_score: float,
    historical_reputation: float,
    anomaly_score: float
) -> float:
    """
    Computes final trust score.
    """

    trust_score = (

        EMBEDDING_WEIGHT * embedding_similarity

        +

        VALIDATION_WEIGHT * validation_score

        +

        REPUTATION_WEIGHT * historical_reputation

        +

        ANOMALY_WEIGHT * anomaly_score

    )

    trust_score = max(
        0.0,
        min(1.0, trust_score)
    )

    return round(float(trust_score), 4)


# ===========================================================
# Client Classification
# ===========================================================

def classify_client(
    trust_score: float
) -> str:
    """
    Categorizes client based on trust score.
    """

    if trust_score >= 0.80:

        return "Trusted"

    elif trust_score >= 0.60:

        return "Reliable"

    elif trust_score >= 0.40:

        return "Suspicious"

    else:

        return "Malicious"


# ===========================================================
# Complete Trust Evaluation
# ===========================================================

def evaluate_client_trust(
    client_embedding: torch.Tensor,
    global_embedding: torch.Tensor,
    validation_accuracy: float,
    previous_reputation: float
):
    """
    Complete Adaptive Trust Evaluation Pipeline.
    """
    if torch.isnan(client_embedding).any():

        raise ValueError(
            "Client embedding contains NaN values."
        )

    if torch.isnan(global_embedding).any():

        raise ValueError(
            "Global embedding contains NaN values."
        )

    embedding_similarity = calculate_embedding_similarity(
        client_embedding,
        global_embedding
    )

    validation_score = calculate_validation_score(
        validation_accuracy
    )

    historical_reputation = update_reputation(
        previous_reputation,
        validation_score
    )

    anomaly_score = calculate_anomaly_score(
        client_embedding,
        global_embedding
    )

    trust_score = calculate_trust_score(
        embedding_similarity,
        validation_score,
        historical_reputation,
        anomaly_score
    )

    client_status = classify_client(
        trust_score
    )

    return {

        "embedding_similarity": round(
            embedding_similarity,
            4
        ),

        "validation_score": round(
            validation_score,
            4
        ),

        "historical_reputation": round(
            historical_reputation,
            4
        ),

        "anomaly_score": round(
            anomaly_score,
            4
        ),

        "trust_score": round(
            trust_score,
            4
        ),

        "confidence": round(

            EMBEDDING_WEIGHT * embedding_similarity +

            VALIDATION_WEIGHT * validation_score +

            REPUTATION_WEIGHT * historical_reputation +

            ANOMALY_WEIGHT * anomaly_score,

            4

        ),

        "status": client_status,

        "weights": TRUST_WEIGHTS
    }

# ===========================================================
# Utility Function
# ===========================================================

def print_trust_report(result: dict):
    """
    Displays trust evaluation result.
    """

    print("\n" + "=" * 60)
    print("Adaptive Trust Evaluation")
    print("=" * 60)
    print(f"Embedding Similarity : {result['embedding_similarity']:.4f}")

    print(f"Validation Score     : {result['validation_score']:.4f}")

    print(f"Historical Reputation: {result['historical_reputation']:.4f}")

    print(f"Anomaly Score        : {result['anomaly_score']:.4f}")

    print(f"Trust Score          : {result['trust_score']:.4f}")

    print(f"Confidence           : {result['confidence']:.4f}")

    print(f"Client Status        : {result['status']}")

    print("=" * 60)