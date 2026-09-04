"""
===========================================================
Model Poisoning Attack
===========================================================

Creates malicious federated model updates by modifying
the parameters of a trained client model.

Member 2:
Adaptive Trust-Aware Federated Intrusion Detection

Attack:
Model Poisoning
"""

from typing import List

import numpy as np


class ModelPoisoningAttack:
    """
    Model poisoning attack for federated learning.

    The attack scales the client's model update relative
    to the received global model.

    A scale factor > 1 amplifies the update.
    A negative scale factor reverses the update.
    """

    def __init__(
        self,
        scale_factor: float = -3.0,
    ):
        """
        Initialize the model poisoning attack.

        Parameters
        ----------
        scale_factor : float
            Controls the strength and direction of poisoning.

            Example:
                1.0  -> normal update
               -1.0  -> reverse update
               -3.0  -> strong reversed update
        """

        self.scale_factor = scale_factor

    # =======================================================
    # Poison Model Parameters
    # =======================================================

    def poison(
        self,
        global_parameters: List[np.ndarray],
        client_parameters: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Create a poisoned client model update.

        The update is calculated as:

            update = client - global

        The poisoned update becomes:

            poisoned_update = scale_factor * update

        Finally:

            poisoned_model = global + poisoned_update
        """

        if len(global_parameters) != len(client_parameters):
            raise ValueError(
                "Global and client parameter lists must have "
                "the same length."
            )

        poisoned_parameters = []

        for global_param, client_param in zip(
            global_parameters,
            client_parameters,
        ):

            if global_param.shape != client_param.shape:
                raise ValueError(
                    "Global and client parameter shapes do not match."
                )

            global_param = np.asarray(
                global_param,
                dtype=np.float32,
            )

            client_param = np.asarray(
                client_param,
                dtype=np.float32,
            )

            update = client_param - global_param

            poisoned_update = (
                self.scale_factor * update
            )

            poisoned_param = (
                global_param + poisoned_update
            )

            poisoned_parameters.append(
                poisoned_param
            )

        return poisoned_parameters

    # =======================================================
    # Poisoning Strength
    # =======================================================

    def get_scale_factor(self) -> float:
        """
        Return the configured poisoning strength.
        """

        return self.scale_factor

    # =======================================================
    # Configuration
    # =======================================================

    def get_config(self) -> dict:
        """
        Return attack configuration.
        """

        return {
            "attack": "Model Poisoning",
            "scale_factor": self.scale_factor,
        }