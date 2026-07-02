import os

import flwr as fl
import torch

from src.datasets.flnet_dataset import load_client_loaders
from src.models.hybrid_model import HybridGRUTransformer
from src.federated.train_utils import (
    train_one_epoch,
    evaluate,
    extract_embedding,
)


class FLNetClient(fl.client.NumPyClient):

    def __init__(self, client_id):

        self.client_id = client_id

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # =====================================================
        # Dataset
        # =====================================================

        (
            self.train_loader,
            self.val_loader,
            self.seq_len,
            self.input_dim,
        ) = load_client_loaders(
            client_id=client_id,
            batch_size=32,
        )

        # =====================================================
        # Model
        # =====================================================

        self.model = HybridGRUTransformer(
            input_dim=self.input_dim,
            seq_len=self.seq_len,
            num_classes=11,
        ).to(self.device)

        # =====================================================
        # Optimizer
        # =====================================================

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001,
            weight_decay=1e-4,
        )

        self.criterion = torch.nn.CrossEntropyLoss()

    # =====================================================
    # Flower Parameters
    # =====================================================

    def get_parameters(self, config):

        return self.model.get_weights()

    # =====================================================
    # Local Training
    # =====================================================

    def fit(self, parameters, config):

        # -----------------------------------------------------
        # Load Global Model
        # -----------------------------------------------------

        self.model.set_weights(parameters)

        # -----------------------------------------------------
        # Train One Local Epoch
        # -----------------------------------------------------

        train_loss = train_one_epoch(
            self.model,
            self.train_loader,
            self.optimizer,
            self.criterion,
            self.device,
        )

        # -----------------------------------------------------
        # Evaluate Local Model
        # -----------------------------------------------------

        metrics = evaluate(
            self.model,
            self.val_loader,
            self.device,
        )

        # -----------------------------------------------------
        # Extract Client Embedding
        # -----------------------------------------------------

        embedding = extract_embedding(
            self.model,
            self.val_loader,
            self.device,
        )

        # -----------------------------------------------------
        # Save Embedding
        # -----------------------------------------------------

        embedding_dir = os.path.join(
            "results",
            "embeddings",
        )

        os.makedirs(
            embedding_dir,
            exist_ok=True,
        )

        round_num = config.get(
            "server_round",
            0,
        )

        embedding_path = os.path.join(

            embedding_dir,

            f"client_{self.client_id}_round_{round_num}.pt"

        )

        torch.save(
            embedding,
            embedding_path,
        )

        # -----------------------------------------------------
        # Terminal Output
        # -----------------------------------------------------

        print("\n" + "=" * 70)

        print(
            f"CLIENT {self.client_id} TRAINING COMPLETED"
        )

        print("=" * 70)

        print(
            f"Training Loss : {train_loss:.4f}"
        )

        print(
            f"Accuracy      : {metrics['accuracy']:.4f}"
        )

        print(
            f"Precision     : {metrics['precision']:.4f}"
        )

        print(
            f"Recall        : {metrics['recall']:.4f}"
        )

        print(
            f"F1 Score      : {metrics['f1']:.4f}"
        )

        print(
            f"Embedding Size: {tuple(embedding.shape)}"
        )

        print(
            f"Saved To      : {embedding_path}"
        )

        print("=" * 70)

        # -----------------------------------------------------
        # Return Results To Server
        # -----------------------------------------------------

        return (

            self.model.get_weights(),

            len(self.train_loader.dataset),

            {

                "client_id": self.client_id,

                "loss": float(train_loss),

                "accuracy": float(
                    metrics["accuracy"]
                ),

                "precision": float(
                    metrics["precision"]
                ),

                "recall": float(
                    metrics["recall"]
                ),

                "f1": float(
                    metrics["f1"]
                ),

                "embedding_path": embedding_path,

            },

        )

    # =====================================================
    # Local Evaluation
    # =====================================================

    def evaluate(self, parameters, config):

        self.model.set_weights(
            parameters
        )

        metrics = evaluate(
            self.model,
            self.val_loader,
            self.device,
        )

        return (

            float(metrics["loss"]),

            len(self.val_loader.dataset),

            {

                "accuracy": float(
                    metrics["accuracy"]
                ),

                "precision": float(
                    metrics["precision"]
                ),

                "recall": float(
                    metrics["recall"]
                ),

                "f1": float(
                    metrics["f1"]
                ),

            },

        )


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:

        raise ValueError(
            "Usage:\n"
            "python -m src.federated.client <client_id>"
        )

    client_id = int(
        sys.argv[1]
    )

    client = FLNetClient(
        client_id
    )

    fl.client.start_numpy_client(

        server_address="127.0.0.1:8080",

        client=client,

    )

