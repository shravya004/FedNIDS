from src.trust.blacklist import BlacklistManager

manager = BlacklistManager()

print("=" * 60)
print("Blacklist Manager Test")
print("=" * 60)

manager.add_client(
    client_id="Client_3",
    reason="Consecutive Low Trust",
    round_number=5,
    trust_score=0.31
)

manager.add_client(
    client_id="Client_7",
    reason="Model Poisoning",
    round_number=8,
    trust_score=0.18
)

manager.print_blacklist()

print("\nTotal Blacklisted :", manager.total_blacklisted())

print("\nClient_3 :", manager.is_blacklisted("Client_3"))

print("Client_5 :", manager.is_blacklisted("Client_5"))

print("\nRemoving Client_3...\n")

manager.remove_client("Client_3")

manager.print_blacklist()