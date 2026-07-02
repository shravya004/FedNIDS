from typing import List

import numpy as np
import torch
import torch.nn as nn

from src.models.positional_encoding import PositionalEncoding


class HybridGRUTransformer(nn.Module):
    """
    Hybrid GRU + Transformer model for Federated Intrusion Detection.
    """

    def __init__(
        self,
        input_dim: int = 1,
        seq_len: int = 64,
        num_classes: int = 2,
        gru_hidden_size: int = 128,
        gru_num_layers: int = 2,
        gru_bidirectional: bool = True,
        transformer_d_model: int = 128,
        transformer_nhead: int = 4,
        transformer_num_layers: int = 2,
        transformer_dim_feedforward: int = 256,
        dropout: float = 0.2,
        pooling: str = "mean",
    ):

        super().__init__()

        # ==========================================================
        # Validation
        # ==========================================================

        if transformer_d_model % transformer_nhead != 0:

            raise ValueError(

                "Transformer hidden dimension must be divisible "
                "by the number of attention heads."

            )

        if pooling not in ("mean", "last", "cls"):

            raise ValueError(

                "Pooling must be one of "
                "'mean', 'last' or 'cls'."

            )

        # ==========================================================
        # Store Configuration
        # ==========================================================

        self.input_dim = input_dim

        self.seq_len = seq_len

        self.num_classes = num_classes

        self.pooling = pooling

        self.gru_hidden_size = gru_hidden_size

        self.gru_num_layers = gru_num_layers

        self.gru_bidirectional = gru_bidirectional

        self.transformer_d_model = transformer_d_model

        self.transformer_nhead = transformer_nhead

        self.transformer_num_layers = transformer_num_layers

        self.transformer_dim_feedforward = transformer_dim_feedforward

        self.dropout = dropout

        # ==========================================================
        # GRU Encoder
        # ==========================================================

        gru_output_dim = gru_hidden_size * (

            2 if gru_bidirectional else 1

        )

        self.gru = nn.GRU(

            input_size=input_dim,

            hidden_size=gru_hidden_size,

            num_layers=gru_num_layers,

            batch_first=True,

            bidirectional=gru_bidirectional,

            dropout=dropout if gru_num_layers > 1 else 0.0,

        )

        # ==========================================================
        # Projection Layer
        # ==========================================================

        self.input_projection = nn.Sequential(

            nn.Linear(

                gru_output_dim,

                transformer_d_model,

            ),

            nn.Dropout(dropout),

        )

        # ==========================================================
        # Positional Encoding
        # ==========================================================

        self.position_encoding = PositionalEncoding(

            transformer_d_model,

            max_len=seq_len + 8,

            dropout=dropout,

        )

        # ==========================================================
        # CLS Token
        # ==========================================================

        if pooling == "cls":

            self.cls_token = nn.Parameter(

                torch.zeros(

                    1,

                    1,

                    transformer_d_model,

                )

            )

            nn.init.trunc_normal_(

                self.cls_token,

                std=0.02,

            )

        # ==========================================================
        # Transformer Encoder
        # ==========================================================

        encoder_layer = nn.TransformerEncoderLayer(

            d_model=transformer_d_model,

            nhead=transformer_nhead,

            dim_feedforward=transformer_dim_feedforward,

            dropout=dropout,

            activation="gelu",

            batch_first=True,

        )

        self.transformer_encoder = nn.TransformerEncoder(

            encoder_layer,

            num_layers=transformer_num_layers,

        )

        # ==========================================================
        # Classification Head
        # ==========================================================

        self.layer_norm = nn.LayerNorm(

            transformer_d_model

        )

        self.classifier = nn.Sequential(

            nn.Linear(

                transformer_d_model,

                transformer_d_model // 2,

            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(

                transformer_d_model // 2,

                num_classes,

            ),

        )

        # ==========================================================
        # Weight Initialization
        # ==========================================================

        self._initialize_weights()

    # ==============================================================
    # Weight Initialization
    # ==============================================================

    def _initialize_weights(self):
        """
        Xavier initialization for all Linear layers.
        """

        for module in self.modules():

            if isinstance(module, nn.Linear):

                nn.init.xavier_uniform_(

                    module.weight

                )

                if module.bias is not None:

                    nn.init.zeros_(

                        module.bias

                    )
    # ==============================================================
    # Feature Extraction
    # ==============================================================

    def forward_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract latent feature embedding before classification.
        """

        gru_output, _ = self.gru(x)

        projected = self.input_projection(
            gru_output
        )

        if self.pooling == "cls":

            cls_tokens = self.cls_token.expand(
                projected.size(0),
                -1,
                -1,
            )

            projected = torch.cat(
                (
                    cls_tokens,
                    projected,
                ),
                dim=1,
            )

        encoded = self.position_encoding(
            projected
        )

        encoded = self.transformer_encoder(
            encoded
        )

        if self.pooling == "mean":

            pooled = encoded.mean(dim=1)

        elif self.pooling == "last":

            pooled = encoded[:, -1, :]

        else:

            pooled = encoded[:, 0, :]

        pooled = self.layer_norm(
            pooled
        )

        return pooled

    # ==============================================================
    # Forward
    # ==============================================================

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward propagation.
        """

        features = self.forward_features(
            x
        )

        logits = self.classifier(
            features
        )

        return logits

    # ==============================================================
    # Prediction
    # ==============================================================

    @torch.inference_mode()
    def predict_proba(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        was_training = self.training

        self.eval()

        probabilities = torch.softmax(

            self.forward(x),

            dim=-1,

        )

        if was_training:

            self.train()

        return probabilities

    @torch.inference_mode()
    def predict(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.predict_proba(
            x
        ).argmax(dim=-1)

    # ==============================================================
    # Flower Weight Utilities
    # ==============================================================

    def get_weights(
        self,
    ) -> List[np.ndarray]:
        """
        Convert model weights to NumPy arrays.
        """

        return [

            parameter.detach()

            .cpu()

            .numpy()

            for parameter in self.state_dict().values()

        ]

    def set_weights(
        self,
        weights: List[np.ndarray],
    ) -> None:
        """
        Load Flower weights.
        """

        state_dict = {

            key: torch.tensor(weight)

            for key, weight in zip(

                self.state_dict().keys(),

                weights,

            )

        }

        self.load_state_dict(

            state_dict,

            strict=True,

        )

    # ==============================================================
    # Statistics
    # ==============================================================

    def count_parameters(
        self,
    ) -> int:
        """
        Count trainable parameters.
        """

        return sum(

            parameter.numel()

            for parameter in self.parameters()

            if parameter.requires_grad

        )

    def get_config(
        self,
    ) -> dict:
        """
        Return model configuration.
        """

        return {

            "input_dim": self.input_dim,

            "seq_len": self.seq_len,

            "num_classes": self.num_classes,

            "gru_hidden_size": self.gru_hidden_size,

            "gru_num_layers": self.gru_num_layers,

            "gru_bidirectional": self.gru_bidirectional,

            "transformer_d_model": self.transformer_d_model,

            "transformer_heads": self.transformer_nhead,

            "transformer_layers": self.transformer_num_layers,

            "transformer_feedforward":
                self.transformer_dim_feedforward,

            "dropout": self.dropout,

            "pooling": self.pooling,

            "trainable_parameters":
                self.count_parameters(),

        }

    # ==============================================================
    # Save / Load
    # ==============================================================

    def save(
        self,
        path: str,
    ):
        """
        Save model weights.
        """

        torch.save(

            self.state_dict(),

            path,

        )

    def load(
        self,
        path: str,
        map_location="cpu",
    ):
        """
        Load model weights.
        """

        self.load_state_dict(

            torch.load(

                path,

                map_location=map_location,

            )

        )

    # ==============================================================
    # Model Summary
    # ==============================================================

    def summary(
        self,
    ):
        """
        Print model architecture and configuration.
        """

        print("\n" + "=" * 70)

        print(
            "Hybrid GRU + Transformer Model"
        )

        print("=" * 70)

        print(self)

        print("\nConfiguration")

        print("-" * 70)

        for key, value in self.get_config().items():

            print(
                f"{key:<30}: {value}"
            )

        print("=" * 70)


# ==============================================================
# Sanity Check
# ==============================================================

if __name__ == "__main__":

    batch_size = 8

    seq_len = 64

    input_dim = 1

    num_classes = 6

    model = HybridGRUTransformer(

        input_dim=input_dim,

        seq_len=seq_len,

        num_classes=num_classes,

    )

    model.summary()

    dummy_input = torch.randn(

        batch_size,

        seq_len,

        input_dim,

    )

    output = model(
        dummy_input
    )

    print(

        "\nOutput Shape:",

        tuple(output.shape),

    )