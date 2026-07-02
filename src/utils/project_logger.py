"""
==============================================================
Project Logger

Handles complete experiment logging for the
Adaptive Trust-Aware Federated Intrusion Detection System.

Features
--------
1. Directory Creation
2. Report Generation
3. CSV Logging
4. Graph Generation
5. Model Saving
6. Embedding Saving
7. Experiment Summary
==============================================================
"""

import os
import csv
from datetime import datetime

import matplotlib.pyplot as plt
import torch


class ProjectLogger:

    def __init__(self):

        # ==================================================
        # Root Result Directory
        # ==================================================

        self.results_dir = "results"

        self.report_dir = os.path.join(
            self.results_dir,
            "reports"
        )

        self.csv_dir = os.path.join(
            self.results_dir,
            "csv"
        )

        self.graph_dir = os.path.join(
            self.results_dir,
            "graphs"
        )

        self.model_dir = os.path.join(
            self.results_dir,
            "models"
        )

        self.embedding_dir = os.path.join(
            self.results_dir,
            "embeddings"
        )

        self.experiment_info = {}

        self.create_directories()

        # ==================================================
        # Report Files
        # ==================================================

        self.experiment_report = os.path.join(
            self.report_dir,
            "experiment_report.txt"
        )

        self.configuration_report = os.path.join(
            self.report_dir,
            "configuration.txt"
        )

        self.trust_report = os.path.join(
            self.report_dir,
            "trust_report.txt"
        )

        self.aggregation_report = os.path.join(
            self.report_dir,
            "aggregation_report.txt"
        )

        self.blacklist_report = os.path.join(
            self.report_dir,
            "blacklist_report.txt"
        )

        self.error_report = os.path.join(
            self.report_dir,
            "error_report.txt"
        )

        self.summary_report = os.path.join(
            self.report_dir,
            "summary.txt"
        )

        # ==================================================
        # CSV Files
        # ==================================================

        self.metrics_csv = os.path.join(
            self.csv_dir,
            "metrics.csv"
        )

        self.trust_csv = os.path.join(
            self.csv_dir,
            "trust_results.csv"
        )

        self.reputation_csv = os.path.join(
            self.csv_dir,
            "reputation.csv"
        )

        self.blacklist_csv = os.path.join(
            self.csv_dir,
            "blacklist.csv"
        )

        # ==================================================
        # Experiment State
        # ==================================================

        self.current_round = 0

        self.start_time = None

        self.end_time = None

        # ==================================================
        # In-Memory Histories
        # ==================================================

        self.metric_history = []

        self.trust_history = []

        self.reputation_history = []

        self.blacklist_history = []

        self.aggregation_history = []

        self.client_history = []

        # ==================================================
        # Initialization
        # ==================================================

        self.initialize_reports()

        self.initialize_csv_files()

    # ======================================================
    # Create Directories
    # ======================================================

    def create_directories(self):

        os.makedirs(
            self.results_dir,
            exist_ok=True
        )

        os.makedirs(
            self.report_dir,
            exist_ok=True
        )

        os.makedirs(
            self.csv_dir,
            exist_ok=True
        )

        os.makedirs(
            self.graph_dir,
            exist_ok=True
        )

        os.makedirs(
            self.model_dir,
            exist_ok=True
        )

        os.makedirs(
            self.embedding_dir,
            exist_ok=True
        )

    # ======================================================
    # Initialize Reports
    # ======================================================

    def initialize_reports(self):

        report_files = [

            self.experiment_report,

            self.configuration_report,

            self.trust_report,

            self.aggregation_report,

            self.blacklist_report,

            self.error_report,

            self.summary_report

        ]

        for file in report_files:

            with open(
                file,
                "w",
                encoding="utf-8"
            ) as f:

                f.write("")

    # ======================================================
    # Initialize CSV Files
    # ======================================================

    def initialize_csv_files(self):

        # ---------------- Metrics ----------------

        with open(
            self.metrics_csv,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow([

                "Timestamp",

                "Round",

                "Client",

                "Loss",

                "Accuracy",

                "Precision",

                "Recall",

                "F1"

            ])

        # ---------------- Trust ----------------

        with open(
            self.trust_csv,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow([

                "Timestamp",

                "Round",

                "Client",

                "Embedding Similarity",

                "Validation Score",

                "Anomaly Score",

                "Reputation",

                "Trust Score",

                "Confidence",

                "Status"

            ])

        # ---------------- Reputation ----------------

        with open(
            self.reputation_csv,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow([

                "Timestamp",

                "Round",

                "Client",

                "Current Reputation",

                "Average Reputation",

                "Best Reputation",

                "Worst Reputation",

                "Low Trust Counter"

            ])

        # ---------------- Blacklist ----------------

        with open(
            self.blacklist_csv,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow([

                "Timestamp",

                "Round",

                "Client",

                "Reason",

                "Trust Score"

            ])

    # ======================================================
    # Helper Functions
    # ======================================================

    def current_timestamp(self):

        return datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    def _write(
        self,
        path,
        text,
    ):

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(text)

    def _append(
        self,
        path,
        text,
    ):

        with open(
            path,
            "a",
            encoding="utf-8"
        ) as file:

            file.write(text)

            # ======================================================
    # Experiment Lifecycle
    # ======================================================

    # ------------------------------------------------------

    def end_experiment(self):
        """
        Called once after the federated training finishes.
        """

        self.end_time = datetime.now()

        duration = self.end_time - self.start_time

        self._append(
            self.experiment_report,
            "\n" +
            "=" * 70 +
            "\nExperiment Finished\n" +
            "=" * 70 +
            "\n"
            f"End Time          : {self.current_timestamp()}\n"
            f"Execution Time    : {duration}\n"
            f"Total Rounds      : {self.current_round}\n"
            f"Clients Logged    : {len(self.client_history)}\n"
            f"Trust Evaluations : {len(self.trust_history)}\n"
            f"Blacklist Events  : {len(self.blacklist_history)}\n"
            + "=" * 70 +
            "\n"
        )

    # ======================================================
    # Round Lifecycle
    # ======================================================

    def start_round(
        self,
        round_num,
    ):
        """
        Called at the beginning of every federated round.
        """

        self.current_round = round_num

        self._append(
            self.experiment_report,
            "\n" +
            "=" * 70 +
            f"\nFederated Round {round_num}\n" +
            "=" * 70 +
            "\n"
        )

    # ------------------------------------------------------

    def end_round(
        self,
        round_num,
        global_accuracy,
        average_trust,
    ):
        """
        Called after aggregation finishes.
        """

        self._append(
            self.experiment_report,
            "\nRound Summary\n"
            + "-" * 50 +
            "\n"
            f"Round               : {round_num}\n"
            f"Global Accuracy     : {global_accuracy:.4f}\n"
            f"Average Trust Score : {average_trust:.4f}\n"
            + "-" * 50 +
            "\n"
        )

    # ======================================================
    # Save Experiment Summary
    # ======================================================


    # ======================================================
    # Reset Experiment
    # ======================================================

    def reset(self):
        """
        Reset logger before a new experiment.
        """

        self.current_round = 0

        self.start_time = None

        self.end_time = None

        self.metric_history.clear()

        self.trust_history.clear()

        self.reputation_history.clear()

        self.blacklist_history.clear()

        self.aggregation_history.clear()

        self.client_history.clear()

            # ======================================================
    # Client Logging
    # ======================================================

    def log_client(
        self,
        round_num,
        client,
        loss,
        accuracy,
        precision,
        recall,
        f1,
        trust,
        reputation,
        status,
    ):
        """
        Log complete client statistics.
        """

        # --------------------------------------------------
        # Store in Memory
        # --------------------------------------------------

        self.client_history.append({

            "round": round_num,

            "client": client,

            "loss": loss,

            "accuracy": accuracy,

            "precision": precision,

            "recall": recall,

            "f1": f1,

            "trust": trust,

            "reputation": reputation,

            "status": status

        })

        # --------------------------------------------------
        # Write Report
        # --------------------------------------------------

        report = []

        report.append("\n")
        report.append("=" * 70)
        report.append("\n")

        report.append(f"Client : {client}\n")

        report.append("=" * 70)
        report.append("\n")

        report.append(
            f"Timestamp      : {self.current_timestamp()}\n"
        )

        report.append(
            f"Round          : {round_num}\n"
        )

        report.append(
            f"Training Loss  : {loss:.4f}\n"
        )

        report.append(
            f"Accuracy       : {accuracy:.4f}\n"
        )

        report.append(
            f"Precision      : {precision:.4f}\n"
        )

        report.append(
            f"Recall         : {recall:.4f}\n"
        )

        report.append(
            f"F1 Score       : {f1:.4f}\n"
        )

        report.append(
            f"Trust Score    : {trust:.4f}\n"
        )

        report.append(
            f"Reputation     : {reputation:.4f}\n"
        )

        report.append(
            f"Status         : {status}\n"
        )

        report.append("-" * 70)
        report.append("\n")

        self._append(

            self.experiment_report,

            "".join(report)

        )

    # ======================================================
    # Trust Logging
    # ======================================================

    def log_trust(
        self,
        round_num,
        client,
        embedding_similarity,
        validation_score,
        anomaly_score,
        reputation,
        trust_score,
        confidence,
        status,
    ):
        """
        Save detailed trust evaluation.
        """

        # --------------------------------------------------
        # Store in Memory
        # --------------------------------------------------

        self.trust_history.append({

            "round": round_num,

            "client": client,

            "embedding_similarity": embedding_similarity,

            "validation_score": validation_score,

            "anomaly_score": anomaly_score,

            "reputation": reputation,

            "trust_score": trust_score,

            "confidence": confidence,

            "status": status

        })

        # --------------------------------------------------
        # Write Report
        # --------------------------------------------------

        report = []

        report.append("\n")
        report.append("=" * 70)
        report.append("\n")

        report.append(
            f"Trust Evaluation : {client}\n"
        )

        report.append("=" * 70)
        report.append("\n")

        report.append(
            f"Timestamp              : {self.current_timestamp()}\n"
        )

        report.append(
            f"Round                  : {round_num}\n"
        )

        report.append(
            f"Embedding Similarity   : {embedding_similarity:.4f}\n"
        )

        report.append(
            f"Validation Score       : {validation_score:.4f}\n"
        )

        report.append(
            f"Historical Reputation  : {reputation:.4f}\n"
        )

        report.append(
            f"Anomaly Score          : {anomaly_score:.4f}\n"
        )

        report.append(
            f"Final Trust Score      : {trust_score:.4f}\n"
        )

        report.append(
            f"Confidence             : {confidence:.4f}\n"
        )

        report.append(
            f"Client Status          : {status}\n"
        )

        report.append("-" * 70)
        report.append("\n")

        self._append(

            self.trust_report,

            "".join(report)

        )

            # ======================================================
    # Blacklist Logging
    # ======================================================

    def log_blacklist(
        self,
        round_num,
        client,
        reason,
        trust_score,
    ):
        """
        Log blacklisted client information.
        """

        # --------------------------------------------------
        # Store in Memory
        # --------------------------------------------------

        self.blacklist_history.append({

            "round": round_num,

            "client": client,

            "reason": reason,

            "trust_score": trust_score

        })

        # --------------------------------------------------
        # Write Report
        # --------------------------------------------------

        report = []

        report.append("\n")
        report.append("=" * 70)
        report.append("\n")

        report.append(
            f"Blacklisted Client : {client}\n"
        )

        report.append("=" * 70)
        report.append("\n")

        report.append(
            f"Timestamp   : {self.current_timestamp()}\n"
        )

        report.append(
            f"Round       : {round_num}\n"
        )

        report.append(
            f"Reason      : {reason}\n"
        )

        report.append(
            f"Trust Score : {trust_score:.4f}\n"
        )

        report.append("-" * 70)
        report.append("\n")

        self._append(

            self.blacklist_report,

            "".join(report)

        )

    # ======================================================
    # Aggregation Logging
    # ======================================================

    def log_aggregation(
        self,
        round_num,
        included_clients,
        excluded_clients,
        trust_weights,
    ):
        """
        Log trust-aware aggregation information.
        """

        # --------------------------------------------------
        # Store in Memory
        # --------------------------------------------------

        self.aggregation_history.append({

            "round": round_num,

            "included_clients": included_clients.copy(),

            "excluded_clients": excluded_clients.copy(),

            "trust_weights": trust_weights.copy()

        })

        # --------------------------------------------------
        # Write Report
        # --------------------------------------------------

        report = []

        report.append("\n")
        report.append("=" * 70)
        report.append("\n")

        report.append(
            f"Aggregation Round : {round_num}\n"
        )

        report.append("=" * 70)
        report.append("\n")

        report.append(
            f"Timestamp : {self.current_timestamp()}\n\n"
        )

        # Included Clients

        report.append("Included Clients\n")
        report.append("-" * 40)
        report.append("\n")

        if len(included_clients) == 0:

            report.append("None\n")

        else:

            for client in included_clients:

                report.append(f"{client}\n")

        report.append("\n")

        # Excluded Clients

        report.append("Excluded Clients\n")
        report.append("-" * 40)
        report.append("\n")

        if len(excluded_clients) == 0:

            report.append("None\n")

        else:

            for client in excluded_clients:

                report.append(f"{client}\n")

        report.append("\n")

        # Trust Weights

        report.append("Normalized Trust Weights\n")
        report.append("-" * 40)
        report.append("\n")

        if len(trust_weights) == 0:

            report.append("No weights available.\n")

        else:

            for client, weight in zip(
                included_clients,
                trust_weights
            ):

                report.append(
                    f"{client:<20} : {weight:.4f}\n"
                )

        report.append("\n")

        report.append(
            f"Total Included Clients : {len(included_clients)}\n"
        )

        report.append(
            f"Total Excluded Clients : {len(excluded_clients)}\n"
        )

        report.append(
            f"Aggregation Completed  : Yes\n"
        )

        report.append("-" * 70)
        report.append("\n")

        self._append(

            self.aggregation_report,

            "".join(report)

        )
            # ======================================================
    # Metrics CSV
    # ======================================================

    def log_metrics_csv(
        self,
        round_num,
        client,
        loss,
        accuracy,
        precision,
        recall,
        f1,
    ):
        """
        Save client metrics to metrics.csv
        """

        with open(
            self.metrics_csv,
            "a",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(file)

            writer.writerow([

                self.current_timestamp(),

                round_num,

                client,

                round(loss, 4),

                round(accuracy, 4),

                round(precision, 4),

                round(recall, 4),

                round(f1, 4),

            ])

        self.metric_history.append({

            "timestamp": self.current_timestamp(),

            "round": round_num,

            "client": client,

            "loss": loss,

            "accuracy": accuracy,

            "precision": precision,

            "recall": recall,

            "f1": f1,

        })

    # ======================================================
    # Trust CSV
    # ======================================================

    def log_trust_csv(
        self,
        round_num,
        client,
        embedding_similarity,
        validation_score,
        anomaly_score,
        reputation,
        trust_score,
        confidence,
        status,
    ):
        """
        Save trust evaluation to trust_results.csv
        """

        with open(
            self.trust_csv,
            "a",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(file)

            writer.writerow([

                self.current_timestamp(),

                round_num,

                client,

                round(embedding_similarity, 4),

                round(validation_score, 4),

                round(anomaly_score, 4),

                round(reputation, 4),

                round(trust_score, 4),

                round(confidence, 4),

                status,

            ])

        self.trust_history.append({

            "timestamp": self.current_timestamp(),

            "round": round_num,

            "client": client,

            "embedding_similarity": embedding_similarity,

            "validation_score": validation_score,

            "anomaly_score": anomaly_score,

            "reputation": reputation,

            "trust_score": trust_score,

            "confidence": confidence,

            "status": status,

        })

    # ======================================================
    # Reputation CSV
    # ======================================================

    def log_reputation_csv(
        self,
        round_num,
        client,
        reputation,
        average_reputation,
        best_reputation,
        worst_reputation,
        low_trust_counter,
    ):
        """
        Save reputation information.
        """

        with open(
            self.reputation_csv,
            "a",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(file)

            writer.writerow([

                self.current_timestamp(),

                round_num,

                client,

                round(reputation, 4),

                round(average_reputation, 4),

                round(best_reputation, 4),

                round(worst_reputation, 4),

                low_trust_counter,

            ])

        self.reputation_history.append({

            "timestamp": self.current_timestamp(),

            "round": round_num,

            "client": client,

            "reputation": reputation,

            "average": average_reputation,

            "best": best_reputation,

            "worst": worst_reputation,

            "low_trust_counter": low_trust_counter,

        })

    # ======================================================
    # Blacklist CSV
    # ======================================================

    def log_blacklist_csv(
        self,
        round_num,
        client,
        reason,
        trust_score,
    ):
        """
        Save blacklist information.
        """

        with open(
            self.blacklist_csv,
            "a",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(file)

            writer.writerow([

                self.current_timestamp(),

                round_num,

                client,

                reason,

                round(trust_score, 4),

            ])

        self.blacklist_history.append({

            "timestamp": self.current_timestamp(),

            "round": round_num,

            "client": client,

            "reason": reason,

            "trust_score": trust_score,

        })
        # ======================================================
    # Error Logging
    # ======================================================

    def log_error(
        self,
        exception,
        module="Unknown",
    ):
        """
        Log an exception to the error report.
        """

        report = []

        report.append("\n")
        report.append("=" * 70)
        report.append("\n")

        report.append("ERROR REPORT\n")

        report.append("=" * 70)
        report.append("\n")

        report.append(
            f"Timestamp : {self.current_timestamp()}\n"
        )

        report.append(
            f"Module    : {module}\n"
        )

        report.append(
            f"Exception : {str(exception)}\n"
        )

        report.append("=" * 70)
        report.append("\n")

        self._append(
            self.error_report,
            "".join(report)
        )

    # ======================================================
    # Experiment Configuration
    # ======================================================

    def log_configuration(
        self,
        dataset,
        model,
        clients,
        rounds,
        batch_size,
        learning_rate,
    ):
        """
        Save experiment configuration.
        """

        report = []

        report.append("=" * 70)
        report.append("\n")

        report.append("EXPERIMENT CONFIGURATION\n")

        report.append("=" * 70)
        report.append("\n")

        report.append(
            f"Generated On     : {self.current_timestamp()}\n"
        )

        report.append(
            f"Dataset          : {dataset}\n"
        )

        report.append(
            f"Model            : {model}\n"
        )

        report.append(
            f"Federated Clients: {clients}\n"
        )

        report.append(
            f"Training Rounds  : {rounds}\n"
        )

        report.append(
            f"Batch Size       : {batch_size}\n"
        )

        report.append(
            f"Learning Rate    : {learning_rate}\n"
        )

        report.append("=" * 70)
        report.append("\n")

        self._write(
            self.configuration_report,
            "".join(report)
        )

    # ======================================================
    # Console Printing
    # ======================================================

    def print_header(
        self,
        title,
    ):
        """
        Print formatted header.
        """

        print("\n")
        print("=" * 70)
        print(title)
        print("=" * 70)

    # ------------------------------------------------------

    def print_subheader(
        self,
        title,
    ):
        """
        Print formatted sub-header.
        """

        print("\n")
        print("-" * 50)
        print(title)
        print("-" * 50)

    # ------------------------------------------------------

    def separator(self):
        """
        Print separator.
        """

        print("-" * 70)

    # ======================================================
    # Console Messages
    # ======================================================

    def info(
        self,
        message,
    ):

        print(f"[INFO] {message}")

    # ------------------------------------------------------

    def success(
        self,
        message,
    ):

        print(f"[SUCCESS] {message}")

    # ------------------------------------------------------

    def warning(
        self,
        message,
    ):

        print(f"[WARNING] {message}")

    # ======================================================
    # Progress Display
    # ======================================================

    def print_progress(
        self,
        current,
        total,
        prefix="Progress",
    ):
        """
        Print experiment progress.
        """

        percentage = (current / total) * 100

        print(
            f"{prefix}: "
            f"{current}/{total} "
            f"({percentage:.1f}%)"
        )

    # ======================================================
    # Generic Report Writer
    # ======================================================

    def append_report(
        self,
        report_path,
        text,
    ):
        """
        Append custom text to any report.
        """

        self._append(
            report_path,
            text + "\n"
        )

    # ======================================================
    # Export Statistics
    # ======================================================

    def export_statistics(self):
        """
        Export overall experiment statistics.
        """

        return {

            "rounds": self.current_round,

            "clients_logged": len(self.client_history),

            "metric_entries": len(self.metric_history),

            "trust_entries": len(self.trust_history),

            "reputation_entries": len(self.reputation_history),

            "blacklist_entries": len(self.blacklist_history),

            "aggregation_entries": len(self.aggregation_history),

        }

    # ======================================================
    # Representation
    # ======================================================

    def __repr__(self):

        return (

            f"ProjectLogger("

            f"results='{self.results_dir}', "

            f"rounds={self.current_round})"

        )
  
    # ============================================================
    # Graph Generation
    # ============================================================

    def _prepare_graph(self):
        """
        Common graph settings.
        """

        plt.figure(figsize=(10, 6))

        plt.grid(True, linestyle="--", alpha=0.4)


    # ============================================================
    # Accuracy Graph
    # ============================================================

    def plot_accuracy(self):
        """
        Plot validation accuracy across rounds.
        """

        if len(self.metric_history) == 0:

            return

        rounds = [
            record["round"]
            for record in self.metric_history
        ]

        accuracy = [
            record["accuracy"]
            for record in self.metric_history
        ]

        self._prepare_graph()

        plt.plot(

            rounds,

            accuracy,

            marker="o",

            linewidth=2,

            label="Accuracy",

        )

        plt.xlabel("Federated Round")

        plt.ylabel("Accuracy")

        plt.title("Validation Accuracy")

        plt.legend()

        plt.tight_layout()

        plt.savefig(

            os.path.join(self.graph_dir, "accuracy.png")

        )

        plt.close()


    # ============================================================
    # Loss Graph
    # ============================================================

    def plot_loss(self):
        """
        Plot training loss across rounds.
        """

        if len(self.metric_history) == 0:

            return

        rounds = [
            record["round"]
            for record in self.metric_history
        ]

        losses = [
            record["loss"]
            for record in self.metric_history
        ]

        self._prepare_graph()

        plt.plot(

            rounds,

            losses,

            marker="o",

            linewidth=2,

            label="Loss",

        )

        plt.xlabel("Federated Round")

        plt.ylabel("Loss")

        plt.title("Training Loss")

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.graph_dir,
                "loss.png"
            )
        )

        plt.close()

    # ============================================================
    # Trust Score Graph
    # ============================================================

    def plot_trust(self):
        """
        Plot client trust scores across federated rounds.
        """

        if len(self.trust_history) == 0:

            return

        self._prepare_graph()

        clients = sorted(

            list(

                set(

                    record["client"]

                    for record in self.trust_history

                )

            )

        )

        for client in clients:

            client_records = [

                record

                for record in self.trust_history

                if record["client"] == client

            ]

            rounds = [

                record["round"]

                for record in client_records

            ]

            trust_scores = [

                record["trust_score"]

                for record in client_records

            ]

            plt.plot(

                rounds,

                trust_scores,

                marker="o",

                linewidth=2,

                label=client,

            )

        plt.xlabel("Federated Round")

        plt.ylabel("Trust Score")

        plt.title("Client Trust Score")

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.graph_dir,
                "trust_scores.png"
            )
        )

        plt.close()


    # ============================================================
    # Reputation Graph
    # ============================================================

    def plot_reputation(self):
        """
        Plot client reputation across federated rounds.
        """

        if len(self.reputation_history) == 0:

            return

        self._prepare_graph()

        clients = sorted(

            list(

                set(

                    record["client"]

                    for record in self.reputation_history

                )

            )

        )

        for client in clients:

            client_records = [

                record

                for record in self.reputation_history

                if record["client"] == client

            ]

            rounds = [

                record["round"]

                for record in client_records

            ]

            reputation = [

                record["reputation"]

                for record in client_records

            ]

            plt.plot(

                rounds,

                reputation,

                marker="o",

                linewidth=2,

                label=client,

            )

        plt.xlabel("Federated Round")

        plt.ylabel("Reputation")

        plt.title("Client Reputation")

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.graph_dir,
                "reputation.png"
            )
        )

        plt.close()


    # ============================================================
    # Blacklist Graph
    # ============================================================

    def plot_blacklist(self):
        """
        Plot cumulative number of blacklisted clients.
        """

        if len(self.blacklist_history) == 0:

            return

        self._prepare_graph()

        rounds = [

            record["round"]

            for record in self.blacklist_history

        ]

        totals = []

        count = 0

        for _ in self.blacklist_history:

            count += 1

            totals.append(count)

        plt.step(

            rounds,

            totals,

            where="post",

            linewidth=2,

            label="Blacklisted Clients",

        )

        plt.xlabel("Federated Round")

        plt.ylabel("Total Blacklisted")

        plt.title("Blacklist Progress")

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.graph_dir,
                "blacklist.png"
            )
        )

        plt.close()


    # ============================================================
    # Generate All Graphs
    # ============================================================

    def plot(self):
        """
        Generate every experiment graph.
        """

        print("\nGenerating Graphs...")

        self.plot_accuracy()

        self.plot_loss()

        self.plot_trust()

        self.plot_reputation()

        self.plot_blacklist()

        print("Graphs saved successfully.")

    # ============================================================
    # Experiment Initialization
    # ============================================================

    def start_experiment(
        self,
        dataset,
        model,
        clients,
        rounds,
    ):
        """
        Initialize a new experiment.
        """

        self.start_time = datetime.now()

        self.experiment_info = {

            "dataset": dataset,

            "model": model,

            "clients": clients,

            "rounds": rounds,

            "start_time": self.start_time,

        }

        print("\n" + "=" * 70)
        print("Experiment Started")
        print("=" * 70)

        print(f"Dataset            : {dataset}")

        print(f"Model              : {model}")

        print(f"Clients            : {clients}")

        print(f"Federated Rounds   : {rounds}")

        print(
            f"Start Time         : "
            f"{self.experiment_info['start_time']}"
        )

        print("=" * 70)


    # ============================================================
    # Save Experiment Summary
    # ============================================================

    def save_summary(self):
        """
        Save overall experiment summary.
        """

        end_time = datetime.now()

        duration = end_time - self.experiment_info["start_time"]

        summary_path = os.path.join(

            self.report_dir,

            "experiment_summary.txt"

        )


        with open(

            summary_path,

            "w",

            encoding="utf-8",

        ) as file:

            file.write("=" * 70 + "\n")

            file.write("Adaptive Trust-Aware Federated IDS\n")

            file.write("=" * 70 + "\n\n")

            file.write(

                f"Dataset              : "

                f"{self.experiment_info['dataset']}\n"

            )

            file.write(

                f"Model                : "

                f"{self.experiment_info['model']}\n"

            )

            file.write(

                f"Clients              : "

                f"{self.experiment_info['clients']}\n"

            )

            file.write(

                f"Federated Rounds     : "

                f"{self.experiment_info['rounds']}\n"

            )

            file.write(

                f"Start Time           : "

                f"{self.experiment_info['start_time']}\n"

            )

            file.write(

                f"End Time             : "

                f"{end_time}\n"

            )

            file.write(

                f"Total Duration       : "

                f"{duration}\n"

            )

            file.write("\n")

            file.write("=" * 70 + "\n")

            file.write("Statistics\n")

            file.write("=" * 70 + "\n\n")

            file.write(

                f"Metric Records       : "

                f"{len(self.metric_history)}\n"

            )

            file.write(

                f"Trust Records        : "

                f"{len(self.trust_history)}\n"

            )

            file.write(

                f"Reputation Records   : "

                f"{len(self.reputation_history)}\n"

            )

            file.write(

                f"Blacklist Records    : "

                f"{len(self.blacklist_history)}\n"

            )

        print(

            f"Experiment summary saved to:\n"

            f"{summary_path}"

        )


    # ============================================================
    # Save Detailed Experiment Report
    # ============================================================

    def save_experiment_report(self):
        """
        Save detailed experiment report.
        """

        report_path = os.path.join(

            self.report_dir,

            "experiment_report.txt"

        )

        with open(

            report_path,

            "w",

            encoding="utf-8",

        ) as file:

            file.write("=" * 80 + "\n")

            file.write("Federated Learning Experiment Report\n")

            file.write("=" * 80 + "\n\n")

            file.write("Experiment Information\n")

            file.write("-" * 40 + "\n")

            for key, value in self.experiment_info.items():

                file.write(

                    f"{key:<20}: {value}\n"

                )

            file.write("\n")

            file.write("=" * 80 + "\n")

            file.write("Metric History\n")

            file.write("=" * 80 + "\n\n")

            for record in self.metric_history:

                file.write(

                    f"Round {record['round']} | "

                    f"{record['client']} | "

                    f"Loss={record['loss']:.4f} | "

                    f"Accuracy={record['accuracy']:.4f} | "

                    f"Precision={record['precision']:.4f} | "

                    f"Recall={record['recall']:.4f} | "

                    f"F1={record['f1']:.4f}\n"

                )

            file.write("\n")

            file.write("=" * 80 + "\n")

            file.write("Trust History\n")

            file.write("=" * 80 + "\n\n")

            for record in self.trust_history:

                file.write(

                    f"Round {record['round']} | "

                    f"{record['client']} | "

                    f"Trust={record['trust_score']:.4f} | "

                    f"Reputation={record['reputation']:.4f} | "

                    f"Status={record['status']}\n"

                )

        print(

            f"Detailed report saved to:\n"

            f"{report_path}"

        )

    # ============================================================
    # Save Model
    # ============================================================

    def save_model(
        self,
        model,
        filename="global_model.pth",
    ):
        """
        Save the trained global model.
        """

        model_path = os.path.join(

            self.model_dir,

            filename

        )

        torch.save(

            model.state_dict(),

            model_path,

        )

        print(

            f"Global model saved to:\n"

            f"{model_path}"

        )


    # ============================================================
    # Save Client Embedding
    # ============================================================

    def save_embedding(
        self,
        embedding,
        client_id,
        round_num,
    ):
        """
        Save a client embedding.
        """

        embedding_path = os.path.join(

            self.embedding_dir,

            f"{client_id}_round_{round_num}.pt"

        )

        torch.save(

            embedding,

            embedding_path,

        )


    # ============================================================
    # Save All Results
    # ============================================================

    def save(self):
        """
        Save the complete experiment.
        """

        print("\nSaving Experiment...")

        self.save_summary()

        self.save_experiment_report()

        print("CSV files already updated during execution.")

        print("Experiment saved successfully.")


    # ============================================================
    # Close Logger
    # ============================================================

    def close(self):
        """
        Final cleanup.
        """

        print("\nLogger Closed.")


    # ============================================================
    # Utility
    # ============================================================

    def __len__(self):
        """
        Return number of logged metric records.
        """

        return len(self.metric_history)