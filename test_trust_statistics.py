from src.trust.trust_statistics import TrustStatistics

print("=" * 65)
print("Trust Statistics Module Test")
print("=" * 65)

stats = TrustStatistics()

# -------------------------------------------------------
# Round 1
# -------------------------------------------------------

round1 = [

    {
        "client_id": "Client_1",
        "trust_score": 0.95,
        "status": "Trusted",
        "blacklisted": False
    },

    {
        "client_id": "Client_2",
        "trust_score": 0.83,
        "status": "Trusted",
        "blacklisted": False
    },

    {
        "client_id": "Client_3",
        "trust_score": 0.35,
        "status": "Malicious",
        "blacklisted": False
    }

]

print("\nRound 1")

result = stats.calculate(
    round_number=1,
    client_results=round1
)

stats.print_latest()

# -------------------------------------------------------
# Round 2
# -------------------------------------------------------

round2 = [

    {
        "client_id": "Client_1",
        "trust_score": 0.97,
        "status": "Trusted",
        "blacklisted": False
    },

    {
        "client_id": "Client_2",
        "trust_score": 0.88,
        "status": "Trusted",
        "blacklisted": False
    },

    {
        "client_id": "Client_3",
        "trust_score": 0.22,
        "status": "Blacklisted",
        "blacklisted": True
    }

]

print("\nRound 2")

result = stats.calculate(
    round_number=2,
    client_results=round2
)

stats.print_latest()

# -------------------------------------------------------
# History
# -------------------------------------------------------

print("\n")
print("=" * 65)
print("Stored Statistics")
print("=" * 65)

for item in stats.get_history():

    print(item)

print("\n")
print("=" * 65)
print("Trust Statistics Test Completed Successfully")
print("=" * 65)