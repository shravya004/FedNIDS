import torch

from src.trust.aggregation import TrustAggregator
from src.trust.blacklist import BlacklistManager

aggregator = TrustAggregator()
blacklist = BlacklistManager()

print("=" * 60)
print("Trust-Aware Aggregation Test")
print("=" * 60)

# --------------------------------------------------
# Simulated Client Models
# --------------------------------------------------

client_models = [

    torch.tensor([
        0.80,
        0.50,
        0.30,
        0.10
    ]),

    torch.tensor([
        0.70,
        0.40,
        0.20,
        0.15
    ]),

    torch.tensor([
        -0.50,
        -0.60,
        -0.70,
        -0.80
    ])

]

# --------------------------------------------------
# Trust Scores
# --------------------------------------------------

trust_scores = [

    0.96,

    0.84,

    0.20

]

print("\nBefore Blacklisting")
print("-" * 40)

global_model, aggregation_info = aggregator.aggregate(

    client_models=client_models,

    trust_scores=trust_scores

)

print("\nAggregated Model")

print(global_model)

print("\nAggregation Information")

print(aggregation_info)

# --------------------------------------------------
# Blacklist Client 3
# --------------------------------------------------

blacklist.add_client(

    client_id="Client_3",

    reason="Consecutive Low Trust",

    round_number=3,

    trust_score=0.20

)

print("\n")
print("=" * 60)
print("Aggregation After Blacklisting Client_3")
print("=" * 60)

global_model, aggregation_info = aggregator.aggregate(

    client_models=client_models,

    trust_scores=trust_scores,

    blacklist=blacklist

)

print("\nAggregated Model")

print(global_model)

print("\nAggregation Information")

print(aggregation_info)
