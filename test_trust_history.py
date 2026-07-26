from src.trust.trust_history import TrustHistory

print("=" * 65)
print("Trust History Module Test")
print("=" * 65)

history = TrustHistory()

# ==========================================================
# Simulate Communication Rounds
# ==========================================================

history.add_record(
    round_number=1,
    client_id="Client_1",
    trust_score=0.95,
    reputation=0.95,
    status="Trusted",
    blacklisted=False
)

history.add_record(
    round_number=1,
    client_id="Client_2",
    trust_score=0.84,
    reputation=0.84,
    status="Trusted",
    blacklisted=False
)

history.add_record(
    round_number=1,
    client_id="Client_3",
    trust_score=0.32,
    reputation=0.32,
    status="Malicious",
    blacklisted=False
)

history.add_record(
    round_number=2,
    client_id="Client_1",
    trust_score=0.97,
    reputation=0.97,
    status="Trusted",
    blacklisted=False
)

history.add_record(
    round_number=2,
    client_id="Client_2",
    trust_score=0.88,
    reputation=0.88,
    status="Trusted",
    blacklisted=False
)

history.add_record(
    round_number=2,
    client_id="Client_3",
    trust_score=0.24,
    reputation=0.24,
    status="Blacklisted",
    blacklisted=True
)

# ==========================================================
# Display Individual Client History
# ==========================================================

print("\n")
print("=" * 65)
print("Client Histories")
print("=" * 65)

history.print_client_history("Client_1")

history.print_client_history("Client_2")

history.print_client_history("Client_3")

# ==========================================================
# Display Round History
# ==========================================================

print("\n")
print("=" * 65)
print("Round 2 Records")
print("=" * 65)

for record in history.get_round_history(2):

    print(record)

# ==========================================================
# Display Exported History
# ==========================================================

print("\n")
print("=" * 65)
print("Complete History")
print("=" * 65)

print(history.export_history())

# ==========================================================
# Summary
# ==========================================================

print("\n")
print("=" * 65)
print("Registered Clients")
print("=" * 65)

print(history.get_all_clients())

print("\n")
print("=" * 65)
print(f"Total Clients Stored : {len(history)}")
print("=" * 65)

print("\n")
print("=" * 65)
print("Trust History Test Completed Successfully")
print("=" * 65)