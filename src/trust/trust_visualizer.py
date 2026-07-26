"""
===========================================================
Trust Visualizer
===========================================================

Generates publication-quality figures for the
Adaptive Trust Framework.

Features
--------
1. Trust Score Evolution
2. Reputation Evolution
3. Average Trust Evolution
4. Blacklist Timeline
5. Trust Distribution
6. Trust Score Box Plot

Project:
Adaptive Trust-Aware Federated Intrusion Detection

Dataset:
FLNET2023
"""

import os
import matplotlib.pyplot as plt


class TrustVisualizer:
    """
    Generates publication-quality visualizations
    for the Adaptive Trust Framework.
    """

    def __init__(
        self,
        output_directory="results/figures"
    ):
        """
        Initialize output folders.
        """

        self.output_directory = output_directory

        self.png_directory = os.path.join(
            output_directory,
            "png"
        )

        self.pdf_directory = os.path.join(
            output_directory,
            "pdf"
        )

        os.makedirs(
            self.png_directory,
            exist_ok=True
        )

        os.makedirs(
            self.pdf_directory,
            exist_ok=True
        )

    # ======================================================
    # Common Plot Style
    # ======================================================

    def _setup_style(
        self,
        title,
        xlabel,
        ylabel
    ):
        """
        Apply common plotting style.
        """

        plt.figure(
            figsize=(8, 5)
        )

        plt.title(
            title,
            fontsize=14,
            fontweight="bold"
        )

        plt.xlabel(
            xlabel,
            fontsize=12
        )

        plt.ylabel(
            ylabel,
            fontsize=12
        )

        plt.grid(
            True,
            linestyle="--",
            alpha=0.5
        )

    # ======================================================
    # Save Figure
    # ======================================================

    def _save_figure(
        self,
        filename
    ):
        """
        Save figure as PNG and PDF.
        """

        png_path = os.path.join(
            self.png_directory,
            filename + ".png"
        )

        pdf_path = os.path.join(
            self.pdf_directory,
            filename + ".pdf"
        )

        plt.tight_layout()

        plt.savefig(
            png_path,
            dpi=300
        )

        plt.savefig(
            pdf_path
        )

        plt.close()

        print("\nSaved Figure")

        print("-" * 40)

        print(f"PNG : {png_path}")

        print(f"PDF : {pdf_path}")

    # ======================================================
    # Trust Score Evolution
    # ======================================================

    def plot_trust_scores(
        self,
        trust_history
    ):
        """
        Plot Trust Score Evolution for every client.
        """

        self._setup_style(

            title="Trust Score Evolution",

            xlabel="Communication Round",

            ylabel="Trust Score"

        )

        for client in trust_history.get_all_clients():

            history = trust_history.get_client_history(
                client
            )

            rounds = [

                record["round"]

                for record in history

            ]

            trust_scores = [

                record["trust_score"]

                for record in history

            ]

            plt.plot(

                rounds,

                trust_scores,

                marker="o",

                linewidth=2,

                markersize=6,

                label=client

            )

        plt.ylim(0, 1)

        plt.legend()

        self._save_figure(
            "trust_score_evolution"
        )
        # ======================================================
    # Reputation Evolution
    # ======================================================

    def plot_reputation(
        self,
        trust_history
    ):
        """
        Plot Reputation Evolution for every client.
        """

        self._setup_style(

            title="Reputation Evolution",

            xlabel="Communication Round",

            ylabel="Reputation"

        )

        for client in trust_history.get_all_clients():

            history = trust_history.get_client_history(
                client
            )

            rounds = [

                record["round"]

                for record in history

            ]

            reputations = [

                record["reputation"]

                for record in history

            ]

            plt.plot(

                rounds,

                reputations,

                marker="o",

                linewidth=2,

                markersize=6,

                label=client

            )

        plt.ylim(0, 1)

        plt.legend()

        self._save_figure(
            "reputation_evolution"
        )

    # ======================================================
    # Average Trust Evolution
    # ======================================================

    def plot_average_trust(
        self,
        statistics
    ):
        """
        Plot Average Trust Score across communication rounds.
        """

        self._setup_style(

            title="Average Trust Score",

            xlabel="Communication Round",

            ylabel="Average Trust"

        )

        history = statistics.get_history()

        rounds = [

            item["round"]

            for item in history

        ]

        averages = [

            item["average_trust"]

            for item in history

        ]

        plt.plot(

            rounds,

            averages,

            marker="o",

            linewidth=2.5,

            markersize=7,

            label="Average Trust"

        )

        plt.ylim(0, 1)

        plt.legend()

        self._save_figure(
            "average_trust"
        )
        # ======================================================
    # Blacklist Timeline
    # ======================================================

    def plot_blacklist_timeline(
        self,
        trust_history
    ):
        """
        Plot blacklist status of clients over
        communication rounds.
        """

        self._setup_style(

            title="Blacklist Timeline",

            xlabel="Communication Round",

            ylabel="Client"

        )

        clients = trust_history.get_all_clients()

        for index, client in enumerate(clients):

            history = trust_history.get_client_history(
                client
            )

            rounds = []

            blacklist = []

            for record in history:

                rounds.append(
                    record["round"]
                )

                if record["blacklisted"]:

                    blacklist.append(
                        index + 1
                    )

                else:

                    blacklist.append(
                        None
                    )

            plt.scatter(

                rounds,

                blacklist,

                marker="X",

                s=120,

                label=client

            )

        plt.yticks(

            range(1, len(clients) + 1),

            clients

        )

        plt.legend()

        self._save_figure(
            "blacklist_timeline"
        )

    # ======================================================
    # Trust Score Distribution
    # ======================================================

    def plot_distribution(
        self,
        trust_history
    ):
        """
        Plot histogram of all trust scores.
        """

        self._setup_style(

            title="Trust Score Distribution",

            xlabel="Trust Score",

            ylabel="Frequency"

        )

        scores = []

        for client in trust_history.get_all_clients():

            history = trust_history.get_client_history(
                client
            )

            for record in history:

                scores.append(
                    record["trust_score"]
                )

        plt.hist(

            scores,

            bins=10,

            edgecolor="black"

        )

        self._save_figure(
            "trust_distribution"
        )
        # ======================================================
    # Trust Score Box Plot
    # ======================================================

    def plot_boxplot(
        self,
        trust_history
    ):
        """
        Plot Trust Score Box Plot.
        """

        self._setup_style(

            title="Trust Score Box Plot",

            xlabel="Clients",

            ylabel="Trust Score"

        )

        data = []

        labels = []

        for client in trust_history.get_all_clients():

            history = trust_history.get_client_history(
                client
            )

            scores = [

                record["trust_score"]

                for record in history

            ]

            data.append(
                scores
            )

            labels.append(
                client
            )

        plt.boxplot(

            data,

            tick_labels=labels

        )

        self._save_figure(
            "trust_boxplot"
        )

    # ======================================================
    # Export All Figures
    # ======================================================

    def export_all(
        self,
        trust_history,
        statistics
    ):
        """
        Generate every figure automatically.
        """

        print("\n")
        print("=" * 65)
        print("Generating Publication Figures")
        print("=" * 65)

        self.plot_trust_scores(
            trust_history
        )

        self.plot_reputation(
            trust_history
        )

        self.plot_average_trust(
            statistics
        )

        self.plot_blacklist_timeline(
            trust_history
        )

        self.plot_distribution(
            trust_history
        )

        self.plot_boxplot(
            trust_history
        )

        print("\n")
        print("=" * 65)
        print("All Figures Generated Successfully")
        print("=" * 65)

    # ======================================================
    # Summary
    # ======================================================

    def summary(self):
        """
        Display information about generated figures.
        """

        print("\n")
        print("=" * 65)
        print("Trust Visualizer Summary")
        print("=" * 65)

        print("Output Directory")
        print(f"  {self.output_directory}")

        print("\nGenerated Figures")

        figures = [

            "trust_score_evolution",

            "reputation_evolution",

            "average_trust",

            "blacklist_timeline",

            "trust_distribution",

            "trust_boxplot"

        ]

        for figure in figures:

            print(f"  ✓ {figure}")

        print("=" * 65)