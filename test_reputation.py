from src.trust.reputation import ReputationManager

manager = ReputationManager()

print("=" * 60)
print("Reputation Manager Test")
print("=" * 60)

# --------------------------------------------------
# Simulate Multiple Federated Learning Rounds
# --------------------------------------------------

manager.update("Client_1", 0.96)
manager.update("Client_1", 0.95)
manager.update("Client_1", 0.97)

manager.update("Client_2", 0.82)
manager.update("Client_2", 0.80)

manager.update("Client_3", 0.30)
manager.update("Client_3", 0.25)
manager.update("Client_3", 0.20)

print("\n========== Individual Client Details ==========\n")

for client in manager.get_all_clients():

    print(f"Client : {client}")

    print("-" * 45)

    print(
        f"Current Reputation : "
        f"{manager.get_reputation(client):.4f}"
    )

    print(
        f"Average Reputation : "
        f"{manager.get_average_reputation(client):.4f}"
    )

    print(
        f"Communication Rounds : "
        f"{manager.get_rounds(client)}"
    )

    print(
        f"Low Trust Counter : "
        f"{manager.get_low_trust_count(client)}"
    )

print("\n")

print("=" * 60)
print("Reputation Summary Dictionary")
print("=" * 60)

summary = manager.reputation_summary()

for client, info in summary.items():

    print(f"\n{client}")

    print("-" * 30)

    print(f"Current Reputation : {info['reputation']:.4f}")

    print(f"Average Reputation : {info['average']:.4f}")

    print(f"Rounds : {info['rounds']}")

    print(f"Low Trust Count : {info['low_trust']}")

print("\n")

manager.print_summary()

print("\n")

print("=" * 60)
print(f"Total Registered Clients : {len(manager)}")
print("=" * 60)