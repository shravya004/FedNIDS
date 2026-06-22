"""
===========================================================
Blacklist Manager
===========================================================

Responsible for managing malicious clients in the
Federated Learning system.

Features
--------
1. Add clients to blacklist
2. Remove clients
3. Check blacklist status
4. Store blacklist reason
5. Store communication round
6. Store trust score
7. Print blacklist summary
8. Recover clients (optional)
9. Return blacklist statistics

Project:
Adaptive Trust-Aware Federated Intrusion Detection

Dataset:
FLNET2023
"""


class BlacklistManager:

    def __init__(self):
        """
        Initialize blacklist manager.
        """

        self.blacklist = {}

    # ------------------------------------------------------

    def add_client(
        self,
        client_id,
        reason,
        round_number,
        trust_score
    ):
        """
        Add a client to the blacklist.
        """

        self.blacklist[client_id] = {

            "reason": reason,

            "round": round_number,

            "trust_score": trust_score

        }

    # ------------------------------------------------------

    def remove_client(
        self,
        client_id
    ):
        """
        Remove a client from the blacklist.
        """

        if client_id in self.blacklist:

            del self.blacklist[client_id]

    # ------------------------------------------------------

    def recover_client(
        self,
        client_id
    ):
        """
        Recover a previously blacklisted client.

        (Future enhancement)
        """

        self.remove_client(client_id)

    # ------------------------------------------------------

    def is_blacklisted(
        self,
        client_id
    ):
        """
        Check whether a client is blacklisted.
        """

        return client_id in self.blacklist

    # ------------------------------------------------------

    def get_blacklist_info(
        self,
        client_id
    ):
        """
        Return blacklist information of one client.
        """

        return self.blacklist.get(
            client_id,
            None
        )

    # ------------------------------------------------------

    def get_blacklisted_clients(self):
        """
        Return list of all blacklisted clients.
        """

        return list(
            self.blacklist.keys()
        )

    # ------------------------------------------------------

    def total_blacklisted(self):
        """
        Return total number of blacklisted clients.
        """

        return len(
            self.blacklist
        )

    # ------------------------------------------------------

    def clear_blacklist(self):
        """
        Remove all blacklisted clients.
        """

        self.blacklist.clear()

    # ------------------------------------------------------

    def blacklist_summary(self):
        """
        Return blacklist statistics.
        """

        return {

            "total_blacklisted": self.total_blacklisted(),

            "clients": self.get_blacklisted_clients()

        }

    # ------------------------------------------------------

    def print_blacklist(self):
        """
        Print blacklist summary.
        """

        print("\n")
        print("=" * 65)
        print("Blacklisted Clients")
        print("=" * 65)

        if self.total_blacklisted() == 0:

            print("No blacklisted clients.")

            return

        for client, info in self.blacklist.items():

            print(f"\nClient : {client}")

            print("-" * 45)

            print(f"Reason       : {info['reason']}")

            print(f"Round        : {info['round']}")

            print(f"Trust Score  : {info['trust_score']:.4f}")

    # ------------------------------------------------------

    def __len__(self):
        """
        Return number of blacklisted clients.
        """

        return self.total_blacklisted()