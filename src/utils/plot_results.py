"""
===========================================================
Results Visualization
===========================================================

Generates graphs from trust_results.csv

Graphs Generated
----------------
1. Accuracy vs Round
2. Precision vs Round
3. Recall vs Round
4. F1 Score vs Round
5. Trust Score vs Round
6. Reputation vs Round

All graphs are saved into:

graphs/

===========================================================
"""

import os

import pandas as pd

import matplotlib.pyplot as plt


RESULT_FILE = "results/trust_results.csv"

GRAPH_DIR = "graphs"

os.makedirs(GRAPH_DIR, exist_ok=True)


def plot_metric(metric_name, ylabel, filename):

    df = pd.read_csv(RESULT_FILE)

    plt.figure(figsize=(8,5))

    for client in sorted(df["Client"].unique()):

        temp = df[df["Client"] == client]

        plt.plot(
            temp["Round"],
            temp[metric_name],
            marker="o",
            linewidth=2,
            label=client
        )

    plt.xlabel("Communication Round")

    plt.ylabel(ylabel)

    plt.title(f"{ylabel} vs Communication Round")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(GRAPH_DIR, filename)
    )

    plt.close()


def main():

    plot_metric(

        "Accuracy",

        "Accuracy",

        "accuracy.png"

    )

    plot_metric(

        "Precision",

        "Precision",

        "precision.png"

    )

    plot_metric(

        "Recall",

        "Recall",

        "recall.png"

    )

    plot_metric(

        "F1",

        "F1 Score",

        "f1_score.png"

    )

    plot_metric(

        "TrustScore",

        "Trust Score",

        "trust_score.png"

    )

    plot_metric(

        "Reputation",

        "Reputation",

        "reputation.png"

    )

    print("\nGraphs generated successfully!")

    print(f"Saved inside: {GRAPH_DIR}")


if __name__ == "__main__":

    main()