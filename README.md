# Federated IDS — FLNET2023 

## Dataset Processing & Hybrid Model (Member 1)

The Dataset Processing and Hybrid Model module provides the complete data preparation pipeline and intrusion detection backbone for the Adaptive Trust-Aware Federated Intrusion Detection System. It processes the FLNET2023 dataset, generates realistic non-IID client partitions, prepares an aligned global test set, and implements the Hybrid GRU-Transformer model used by all federated clients.

The processed datasets and deep learning models are designed to integrate directly with the Adaptive Trust Framework (Member 2) and the Flower Federated Learning Framework (Member 3).

### Components

### FLNET2023 Dataset Processing

The preprocessing pipeline converts the raw FLNET2023 dataset into a clean and standardized format suitable for federated learning.

The pipeline performs:

- Data cleaning
- Missing value handling
- Feature selection
- Feature scaling
- Label encoding
- Dataset standardization

Implemented in:

```
src/data/preprocess.py
```

---

### Client Data Partitioning

The processed dataset is partitioned into client-specific training and validation datasets following the real FLNET2023 router topology (D1–D10).

The partitioning process generates:

- Client-specific training datasets
- Client-specific validation datasets
- Global aligned test dataset
- Client manifest containing dataset statistics

The resulting client datasets preserve the natural non-IID distribution required for realistic federated learning experiments.

Implemented in:

```
src/data/partition.py
```

---

### PyTorch Dataset Loader

A custom dataset loader provides a simple interface for loading client datasets during federated training.

Features include:

- Automatic loading of client datasets
- Feature reshaping for sequential models
- Batch generation using PyTorch DataLoader
- Global test dataset loader

Implemented in:

```
src/datasets/flnet_dataset.py
```

---

### Hybrid GRU-Transformer Model

The Hybrid GRU-Transformer serves as the primary intrusion detection model used throughout the federated learning framework.

The architecture combines:

- Bidirectional GRU layers
- Linear feature projection
- Positional Encoding
- Multi-head Transformer Encoder
- Layer Normalization
- Fully Connected Classification Head

The model supports direct integration with Flower through built-in parameter serialization utilities.

Implemented in:

```
src/models/hybrid_model.py
```

---

### Baseline Models

To evaluate the effectiveness of the proposed Hybrid GRU-Transformer architecture, two baseline models are provided.

Implemented models include:

- GRU Classifier
- Transformer Classifier

These models are used for performance comparison during experimentation.

Implemented in:

```
src/models/baseline_models.py
```

---

### Model Architecture

```
Input (batch, seq_len, input_dim)
        ↓
 Bidirectional GRU (2 layers)
        ↓
 Linear Projection
        ↓
 Positional Encoding
        ↓
 Transformer Encoder
        ↓
 Mean / Last / CLS Pooling
        ↓
 LayerNorm
        ↓
 Dense → GELU → Dropout
        ↓
 Dense
        ↓
 Class Logits
```

---

### Generated Outputs

The preprocessing pipeline automatically generates:

- Cleaned FLNET2023 dataset
- Client-specific training datasets
- Client-specific validation datasets
- Global test dataset
- Client partition manifest
- StandardScaler artifact
- Feature metadata

Generated files include:

```
data/processed/flnet2023_clean.csv

data/processed/clients/client_<id>/train.csv

data/processed/clients/client_<id>/val.csv

data/processed/clients/clients_manifest.json

data/processed/test/global_test.csv

artifacts/scaler.joblib

artifacts/metadata.joblib
```

---

### Integration Support

The generated datasets and Hybrid GRU-Transformer model are designed for seamless integration with the remaining project modules.

For the Federated Learning Framework (Member 3):

- Client DataLoaders
- Global test DataLoader
- Flower-compatible weight serialization
- Model parameter synchronization

For the Adaptive Trust Framework (Member 2):

- Feature embedding extraction
- Intermediate feature representation
- Model weight access for trust computation

---

## Implemented Modules

```
src/data/
│
├── preprocess.py
└── partition.py

src/datasets/
│
└── flnet_dataset.py

src/models/
│
├── hybrid_model.py
├── baseline_models.py
└── positional_encoding.py
```

---

## Testing

The following scripts validate the complete Dataset Processing and Hybrid Model pipeline:

```
test_hybrid_model.py
```

The framework validates:

- Dataset preprocessing
- Client dataset partitioning
- Hybrid GRU-Transformer initialization
- Baseline model implementation
- Data loading pipeline
- Forward propagation
- Model weight serialization

---

## Dataset Processing Workflow

```
          Raw FLNET2023 Dataset
                    │
                    ▼
           Data Cleaning & Encoding
                    │
                    ▼
           Feature Selection & Scaling
                    │
                    ▼
          Non-IID Client Partitioning
                    │
        ┌───────────┴───────────┐
        │                       │
 Client Train/Validation   Global Test Set
        │                       │
        └───────────┬───────────┘
                    │
                    ▼
        Hybrid GRU-Transformer Model
                    │
                    ▼
        Feature Embeddings & Predictions
                    │
        ┌───────────┴───────────┐
        │                       │
 Adaptive Trust Framework   Federated Learning
      (Member 2)               (Member 3)
```

## Adaptive Trust Framework (Member 2)

The Adaptive Trust Framework improves the robustness and security of the federated intrusion detection system by evaluating the reliability of participating clients before global model aggregation. Instead of treating all client updates equally, the framework dynamically assigns trust scores, maintains historical reputation, detects malicious behavior, and performs trust-aware aggregation to mitigate the impact of poisoned or unreliable model updates.

### Components

### Adaptive Trust Scoring

Each participating client receives a trust score computed from multiple complementary indicators:

- Embedding similarity between the client model and the global model
- Local validation accuracy
- Historical reputation maintained across communication rounds

These indicators are combined into a single adaptive trust score used for client evaluation.

---

### Reputation Management

A dedicated reputation manager maintains historical trust information for every federated client throughout the training process.

The stored information includes:

- Current reputation
- Trust history
- Average reputation
- Communication round tracking
- Consecutive low-trust counter

Historical reputation enables long-term behavioral analysis and improves trust stability.

---

### Automatic Blacklisting

Clients whose trust score remains below the predefined threshold for **three consecutive communication rounds** are automatically blacklisted.

The blacklist stores:

- Client ID
- Blacklisting reason
- Communication round
- Trust score at the time of blacklisting

Blacklisted clients are excluded from all subsequent aggregation rounds.

---

### Trust-Aware Aggregation

Instead of conventional FedAvg, aggregation weights are computed by normalizing client trust scores. Only trusted clients contribute to the global model, while blacklisted clients are automatically excluded.

---

## Implemented Modules

```text
src/trust/
│
├── trust_score.py
├── reputation.py
├── blacklist.py
├── aggregation.py
└── integration.py
```

---

## Testing

The following test scripts validate the Adaptive Trust Framework:

- test_trust.py
- test_reputation.py
- test_blacklist.py
- test_aggregation.py
- test_integration.py
- test_member2_complete.py

The complete demonstration (`test_member2_complete.py`) validates:

- Adaptive trust score computation
- Reputation management
- Automatic client classification
- Automatic blacklisting
- Trust-aware aggregation
- End-to-end execution of the Adaptive Trust Framework

## Federated Learning Framework (Member 3)

The Federated Learning Framework implements the complete distributed training pipeline for the Adaptive Trust-Aware Federated Intrusion Detection System. It coordinates multiple FLNET2023 clients using the Flower framework, performs secure model parameter exchange, integrates the Adaptive Trust Framework into the aggregation process, and evaluates the global model after every communication round.

Unlike conventional FedAvg, this framework performs **Trust-Aware Federated Aggregation**, where client updates are evaluated using adaptive trust scores before contributing to the global model. The framework supports secure client communication, embedding-based trust evaluation, reputation management, and automated experiment logging.

### Components

### Flower Federated Server

The Flower server coordinates all participating federated clients throughout the training process.

Responsibilities include:

- Client registration
- Distribution of global model parameters
- Collection of locally trained model updates
- Loading client embeddings
- Global embedding computation
- Adaptive trust evaluation
- Trust-aware model aggregation
- Global model synchronization
- Global model evaluation

The server acts as the central coordinator for the complete federated learning workflow.

---

### Flower Federated Client

Each client performs local intrusion detection training on its own FLNET2023 partition without sharing raw network traffic.

Each communication round performs:

- Loading client-specific datasets
- Receiving global model parameters
- Local Hybrid GRU-Transformer training
- Local validation
- Feature embedding extraction
- Transmission of updated model parameters
- Transmission of evaluation metrics
- Transmission of learned embeddings

Only model parameters, embeddings, and evaluation metrics are exchanged with the server, preserving data privacy.

---

### Local Training Pipeline

Each federated communication round consists of:

- Receiving the latest global model
- Updating local model parameters
- Local model training
- Local validation
- Embedding extraction
- Saving client embeddings
- Returning updated model weights
- Returning evaluation metrics

The complete local training pipeline is implemented in:

```
src/federated/train_utils.py
```

---

### Trust-Aware Flower Strategy

A custom Flower aggregation strategy replaces the standard FedAvg algorithm.

The strategy performs:

- Loading client embeddings
- Computing the global embedding
- Invoking the Adaptive Trust Framework
- Trust score calculation
- Reputation update
- Automatic blacklist verification
- Trust-aware weighted aggregation
- Global model synchronization

Implemented in:

```
src/federated/trust_strategy.py
```

---

### Embedding-Based Client Communication

Each client generates a latent feature embedding after every local training round using the Hybrid GRU-Transformer model.

The embeddings are transmitted to the server and used by the Adaptive Trust Framework for:

- Embedding similarity computation
- Anomaly detection
- Trust score calculation
- Reputation management

This enables secure client evaluation without exposing local training data.

---

### Global Model Evaluation

After every federated communication round, the global model is evaluated using the aligned FLNET2023 global test set.

Evaluation metrics include:

- Loss
- Accuracy
- Precision
- Recall
- F1 Score

These metrics are recorded after every round for performance analysis and comparison.

---

### Experiment Logging

The framework automatically records every federated communication round.

Generated outputs include:

- Client training metrics
- Trust scores
- Reputation history
- Blacklisted clients
- Aggregation statistics
- Performance graphs
- Experiment reports
- Client embeddings

Implemented in:

```
src/utils/project_logger.py
src/utils/plot_results.py
```

---

## Implemented Modules

```
src/federated/
│
├── client.py
├── server.py
├── train_utils.py
└── trust_strategy.py

src/utils/
│
├── project_logger.py
└── plot_results.py
```

---

## Testing

The following scripts validate the complete Federated Learning Framework:

```
test_federated_setup.py
test_train.py
```

The framework validates:

- Flower server initialization
- Flower client communication
- Local model training
- Model parameter synchronization
- Embedding extraction
- Trust-aware aggregation
- Global model updates
- Multi-round federated learning execution

---

## Federated Learning Workflow

```
                 Global Server
                       │
        ┌──────────────┴──────────────┐
        │                             │
 Receive Global Model         Receive Global Model
        │                             │
   Client 1                  Client 2 ... Client N
        │                             │
 Local Training              Local Training
        │                             │
 Local Validation           Local Validation
        │                             │
 Embedding Extraction     Embedding Extraction
        │                             │
 Send Model + Metrics + Embedding
        └──────────────┬──────────────┘
                       │
             Trust Evaluation
                       │
             Reputation Update
                       │
          Trust-Aware Aggregation
                       │
             Updated Global Model
                       │
               Next FL Round
```

---

**Dataset:** FLNET2023

**Framework:** Adaptive Trust-Aware Federated Intrusion Detection System
