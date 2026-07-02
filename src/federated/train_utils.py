import torch

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    confusion_matrix,

)


# ============================================================
# Train One Epoch
# ============================================================

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
):
    """
    Train the local model for one epoch.

    Parameters
    ----------
    model : nn.Module
    train_loader : DataLoader
    optimizer : torch.optim
    criterion : Loss Function
    device : torch.device

    Returns
    -------
    float
        Average training loss.
    """

    model.train()

    running_loss = 0.0

    total_batches = len(train_loader)

    if total_batches == 0:

        return 0.0

    for inputs, labels in train_loader:

        inputs = inputs.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(

            outputs,

            labels,

        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / total_batches

    return float(average_loss)


# ============================================================
# Evaluate Model
# ============================================================

@torch.inference_mode()
def evaluate(
    model,
    dataloader,
    device,
):
    """
    Evaluate a trained model.

    Returns
    -------
    dict
    """

    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0

    predictions = []

    labels = []

    total_batches = len(dataloader)

    if total_batches == 0:

        return {

            "loss": 0.0,

            "accuracy": 0.0,

            "precision": 0.0,

            "recall": 0.0,

            "f1": 0.0,

            "confusion_matrix": [],

        }

    for inputs, targets in dataloader:

        inputs = inputs.to(device)

        targets = targets.to(device)

        outputs = model(inputs)

        loss = criterion(

            outputs,

            targets,

        )

        total_loss += loss.item()

        predicted = torch.argmax(

            outputs,

            dim=1,

        )

        predictions.extend(

            predicted.cpu().tolist()

        )

        labels.extend(

            targets.cpu().tolist()

        )

    validation_loss = total_loss / total_batches

    accuracy = accuracy_score(

        labels,

        predictions,

    )

    precision = precision_score(

        labels,

        predictions,

        average="weighted",

        zero_division=0,

    )

    recall = recall_score(

        labels,

        predictions,

        average="weighted",

        zero_division=0,

    )

    f1 = f1_score(

        labels,

        predictions,

        average="weighted",

        zero_division=0,

    )

    cm = confusion_matrix(

        labels,

        predictions,

    )

    return {

        "loss": round(float(validation_loss), 4),

        "accuracy": round(float(accuracy), 4),

        "precision": round(float(precision), 4),

        "recall": round(float(recall), 4),

        "f1": round(float(f1), 4),

        "confusion_matrix": cm,

    }

# ============================================================
# Extract Client Embedding
# ============================================================

@torch.inference_mode()
def extract_embedding(
    model,
    dataloader,
    device,
):
    """
    Extract a representative embedding for one client.

    The embedding is computed by averaging the feature
    embeddings over the complete validation dataset.

    Returns
    -------
    torch.Tensor
        One-dimensional client embedding.
    """

    model.eval()

    embeddings = []

    for inputs, _ in dataloader:

        inputs = inputs.to(device)

        embedding = model.forward_features(inputs)

        embedding = embedding.mean(dim=0)

        embeddings.append(

            embedding.cpu()

        )

    if len(embeddings) == 0:

        raise ValueError(

            "Unable to extract embedding from an empty dataloader."

        )

    client_embedding = torch.stack(

        embeddings

    ).mean(dim=0)

    return client_embedding


# ============================================================
# Extract Batch Embeddings
# ============================================================

@torch.inference_mode()
def extract_batch_embeddings(
    model,
    dataloader,
    device,
):
    """
    Extract embeddings for every batch.

    Returns
    -------
    list[torch.Tensor]
    """

    model.eval()

    batch_embeddings = []

    for inputs, _ in dataloader:

        inputs = inputs.to(device)

        embedding = model.forward_features(inputs)

        batch_embeddings.append(

            embedding.cpu()

        )

    return batch_embeddings


# ============================================================
# Count Correct Predictions
# ============================================================

@torch.inference_mode()
def count_correct_predictions(
    model,
    dataloader,
    device,
):
    """
    Count correctly classified samples.

    Returns
    -------
    tuple
        (correct_predictions, total_samples)
    """

    model.eval()

    correct = 0

    total = 0

    for inputs, labels in dataloader:

        inputs = inputs.to(device)

        labels = labels.to(device)

        outputs = model(inputs)

        predictions = torch.argmax(

            outputs,

            dim=1,

        )

        correct += (

            predictions == labels

        ).sum().item()

        total += labels.size(0)

    return correct, total


# ============================================================
# Move Batch to Device
# ============================================================

def move_to_device(
    inputs,
    labels,
    device,
):
    """
    Utility function for moving tensors to the
    appropriate device.
    """

    return (

        inputs.to(device),

        labels.to(device),

    )
