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

# ==================================================
# Before Blacklisting
# ==================================================

print("\nBefore Blacklisting")
print("-" * 40)

global_model = aggregator.aggregate(

    client_models=client_models,

    trust_scores=trust_scores

)

aggregation_info = aggregator.export_statistics()

print("\nAggregated Model")
print(global_model)

print("\nAggregation Information")
print("-" * 40)

print(f"Included Clients   : {aggregation_info['included_clients']}")
print(f"Excluded Clients   : {aggregation_info['excluded_clients']}")
print(f"Trust Weights      : {aggregation_info['trust_weights']}")
print(f"Total Clients      : {aggregation_info['num_clients']}")
print(f"Trusted Clients    : {aggregation_info['num_trusted']}")
print(f"Blacklisted Clients: {aggregation_info['num_blacklisted']}")

print("\n")
aggregator.print_summary()

# ==================================================
# Blacklist Client 3
# ==================================================

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

global_model = aggregator.aggregate(

    client_models=client_models,

    trust_scores=trust_scores,

    blacklist=blacklist

)

aggregation_info = aggregator.export_statistics()

print("\nAggregated Model")
print(global_model)

print("\nAggregation Information")
print("-" * 40)

print(f"Included Clients   : {aggregation_info['included_clients']}")
print(f"Excluded Clients   : {aggregation_info['excluded_clients']}")
print(f"Trust Weights      : {aggregation_info['trust_weights']}")
print(f"Total Clients      : {aggregation_info['num_clients']}")
print(f"Trusted Clients    : {aggregation_info['num_trusted']}")
print(f"Blacklisted Clients: {aggregation_info['num_blacklisted']}")

print("\n")
aggregator.print_summary()

print("\n")
print("=" * 60)
print("Aggregation Test Completed Successfully")
print("=" * 60)