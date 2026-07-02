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
7. Store timestamp
8. Store violation count
9. Print blacklist summary
10. Export blacklist statistics

Project:
Adaptive Trust-Aware Federated Intrusion Detection

Dataset:
FLNET2023
"""

from datetime import datetime


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
        trust_score,
    ):
        """
        Add a client to the blacklist.
        """

        if client_id in self.blacklist:

            self.blacklist[client_id]["violations"] += 1

            self.blacklist[client_id]["reason"] = reason

            self.blacklist[client_id]["round"] = round_number

            self.blacklist[client_id]["trust_score"] = trust_score

            self.blacklist[client_id]["timestamp"] = datetime.now()

            return

        self.blacklist[client_id] = {

            "reason": reason,

            "round": round_number,

            "trust_score": trust_score,

            "timestamp": datetime.now(),

            "violations": 1,

        }

    # ------------------------------------------------------

    def remove_client(
        self,
        client_id,
    ):
        """
        Remove a client from blacklist.
        """

        self.blacklist.pop(client_id, None)

    # ------------------------------------------------------

    def recover_client(
        self,
        client_id,
    ):
        """
        Recover a previously blacklisted client.
        """

        self.remove_client(client_id)

    # ------------------------------------------------------

    def is_blacklisted(
        self,
        client_id,
    ):
        """
        Check blacklist status.
        """

        return client_id in self.blacklist

    # ------------------------------------------------------

    def get_blacklist_info(
        self,
        client_id,
    ):
        """
        Return blacklist information.
        """

        return self.blacklist.get(client_id, None)

    # ------------------------------------------------------

    def get_blacklisted_clients(self):
        """
        Return all blacklisted clients.
        """

        return list(self.blacklist.keys())

    # ------------------------------------------------------

    def total_blacklisted(self):
        """
        Return total number of blacklisted clients.
        """

        return len(self.blacklist)

    # ------------------------------------------------------

    def clear_blacklist(self):
        """
        Remove every client from blacklist.
        """

        self.blacklist.clear()

    # ------------------------------------------------------

    def blacklist_summary(self):
        """
        Return blacklist statistics.
        """

        return {

            "total_blacklisted": self.total_blacklisted(),

            "clients": self.get_blacklisted_clients(),

        }

    # ------------------------------------------------------

    def export_statistics(self):
        """
        Export blacklist information.
        """

        return {

            "total_blacklisted": self.total_blacklisted(),

            "clients": self.blacklist,

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

            print(f"Reason          : {info['reason']}")

            print(f"Round           : {info['round']}")

            print(f"Trust Score     : {info['trust_score']:.4f}")

            print(f"Violations      : {info['violations']}")

            print(f"Blacklisted On  : {info['timestamp']}")

        print("\n")
        print("=" * 65)

        print(
            f"Total Blacklisted Clients : {self.total_blacklisted()}"
        )

        print("=" * 65)

    # ------------------------------------------------------

    def __len__(self):
        """
        Return number of blacklisted clients.
        """

        return self.total_blacklisted()