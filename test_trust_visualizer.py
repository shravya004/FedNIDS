from src.trust.trust_history import TrustHistory
from src.trust.trust_statistics import TrustStatistics
from src.trust.trust_visualizer import TrustVisualizer

print("=" * 70)
print("Trust Visualizer Test")
print("=" * 70)

# ============================================================
# Initialize
# ============================================================

history = TrustHistory()

statistics = TrustStatistics()

visualizer = TrustVisualizer()

# ============================================================
# Simulate Communication Rounds
# ============================================================

for round_number in range(1, 6):

    client_results = []

    client_data = [

        ("Client_1", 0.82 + round_number * 0.03, "Trusted", False),

        ("Client_2", 0.72 + round_number * 0.03, "Trusted", False),

        (
            "Client_3",
            0.32 - round_number * 0.02,
            "Blacklisted" if round_number >= 3 else "Malicious",
            True if round_number >= 3 else False
        )

    ]

    for client_id, trust, status, blacklisted in client_data:

        history.add_record(

            round_number=round_number,

            client_id=client_id,

            trust_score=trust,

            reputation=trust,

            status=status,

            blacklisted=blacklisted

        )

        client_results.append({

            "client_id": client_id,

            "trust_score": trust,

            "status": status,

            "blacklisted": blacklisted

        })

    statistics.calculate(

        round_number,

        client_results

    )

# ============================================================
# Generate Figures
# ============================================================

print("\nGenerating Figures...\n")

visualizer.export_all(

    history,

    statistics

)

visualizer.summary()

print("\n")
print("=" * 70)
print("Trust Visualizer Test Completed Successfully")
print("=" * 70)