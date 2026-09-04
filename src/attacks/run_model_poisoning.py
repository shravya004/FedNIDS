"""
===========================================================
Model Poisoning Experiment
===========================================================

Standalone experiment for testing the Member 2
model-poisoning attack.

This file does NOT modify the existing federated
learning client, server, or trust framework code.
"""

import numpy as np

from src.attacks.model_poisoning import ModelPoisoningAttack
from src.attacks.attack_config import (
    MODEL_POISONING_ENABLED,
    POISONING_SCALE_FACTOR,
)


def run_model_poisoning_experiment():
    """
    Run a small standalone model-poisoning experiment.
    """

    print("\n" + "=" * 70)
    print("MODEL POISONING ATTACK EXPERIMENT")
    print("=" * 70)

    if not MODEL_POISONING_ENABLED:
        print("Model poisoning is disabled.")
        return

    # ------------------------------------------------------
    # Simulated global model
    # ------------------------------------------------------

    global_parameters = [
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, 5.0]),
    ]

    # ------------------------------------------------------
    # Simulated honest client model
    # ------------------------------------------------------

    client_parameters = [
        np.array([1.1, 2.2, 3.3]),
        np.array([4.4, 5.5]),
    ]

    print("\nGlobal Parameters:")
    print(global_parameters)

    print("\nClient Parameters:")
    print(client_parameters)

    # ------------------------------------------------------
    # Create attack
    # ------------------------------------------------------

    attack = ModelPoisoningAttack(
        scale_factor=POISONING_SCALE_FACTOR
    )

    # ------------------------------------------------------
    # Generate poisoned model
    # ------------------------------------------------------

    poisoned_parameters = attack.poison(
        global_parameters=global_parameters,
        client_parameters=client_parameters,
    )

    print("\nPoisoning Configuration:")
    print(attack.get_config())

    print("\nPoisoned Parameters:")
    print(poisoned_parameters)

    # ------------------------------------------------------
    # Compare updates
    # ------------------------------------------------------

    print("\nUpdate Comparison")
    print("-" * 50)

    for i, (global_param, client_param, poisoned_param) in enumerate(
        zip(
            global_parameters,
            client_parameters,
            poisoned_parameters,
        )
    ):

        normal_update = client_param - global_param

        poisoned_update = poisoned_param - global_param

        print(f"\nLayer {i + 1}")

        print(f"Normal Update   : {normal_update}")

        print(f"Poisoned Update : {poisoned_update}")

    print("\n" + "=" * 70)
    print("Model poisoning experiment completed.")
    print("=" * 70)


if __name__ == "__main__":
    run_model_poisoning_experiment()