import os
import torch
import torch.nn as nn
import pennylane as qml
import torchvision
import torchvision.transforms as transforms

from pennylane import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

class ClassicalBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            padding=0
        )

        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(channels)

        self.conv3 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.bn3 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # Residual connection
        out = out + identity

        out = self.conv3(out)
        out = self.bn3(out)

        return out

class SFEModule(nn.Module):
    def __init__(self):
        super().__init__()

        # Convert RGB image (3 channels)
        # into 64 feature channel

        self.stem = nn.Sequential(

            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.block1 = ClassicalBlock(64)
        self.block2 = ClassicalBlock(64)
        self.block3 = ClassicalBlock(64)
        self.block4 = ClassicalBlock(64)

        self.gap = nn.AdaptiveAvgPool2d((1,1))

        self.flatten = nn.Flatten()

    def forward(self, x):

        x = self.stem(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.gap(x)

        x = self.flatten(x)

        return x


n_qubits = 4
dev = qml.device(
    "default.qubit",
    wires = n_qubits
)

@qml.qnode(dev, interface="torch")

def quantum_circuit(inputs, weights):

    # Angle Encoding
    for i in range(4):
        qml.Hadamard(wires=1)

        qml.RZ(
            inputs[i],
            wires=i
        )

    # Trainable Layer 1
    for i in range(4):
        qml.RZ(
            weights[i],
            wires=i
        )

    # Entanglement
    qml.CNOT(wires=[0,1])
    qml.CNOT(wires=[1,2])
    qml.CNOT(wires=[2,3])

    # Trainable Layer 2
    for i in range(4):
        qml.RZ(
            weights[4+i],
            wires=i
        )

    # Entanglement
    qml.CNOT(wires=[0,1])
    qml.CNOT(wires=[1,2])
    qml.CNOT(wires=[2,3])

    # Trainable Layer 3
    for i in range(4):
        qml.RY(
            weights[8+i],
            wires=i
        )

    return qml.probs(wires=[0,1,2,3])

class QuantumLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(
            torch.randn(16)
        )

    def forward(self, x):
        outputs = []
        current_device = x.device # Store the device of the input tensor
        for sample in x:
            result = quantum_circuit(
                sample.cpu(),  # Move sample to CPU for quantum circuit
                self.weights.cpu() # Move weights to CPU for quantum circuit
            )
            # Move the result back to the original device (GPU if applicable)
            outputs.append(result.to(current_device))

        return torch.stack(outputs).float()

class QuCNet(nn.Module):
    def __init__(self, num_classses=10):
        super().__init__()

        # Feature extractor
        self.sfe = SFEModule()

        # 16 parallel quantum layers
        self.quantum_layers = nn.ModuleList(
            [
                QuantumLayer()
                for _ in range(4)
            ]
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classses)
        )

    def forward(self, x):

        # Extract features
        features = self.sfe(x)

        # shape:
        # (B,64)

        # Split into sixteen chunks
        chunks = torch.chunk(
            features,
            chunks = 16,
            dim=1,
        )

        q_outputs = []

        # Pass each chunk through its quantum circuit
        for i in range(16):

            out = self.quantum_layers[i % 4](
                chunks[i]
            )

            q_outputs.append(out)

        # Concatenate outputs
        quantum_features = torch.cat(
            q_outputs,
            dim=1
        ).float()

        # Shape:
        # (B,64)

        # quantum_features = torch.cat(
        #     q_outputs,
        #     dim=1
        # ).float()

        logits = self.classifier(
            quantum_features
        )

        return logits

#######################
##   Training File   ##
#######################

BATCH_SIZE = 32
EPOCHS = 4
LR = 0.001
NUM_CLASSES = 10

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(device)
# print(torch.__version__)
# print(torch.cuda.is_available())

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5,0.5,0.5),
            (0.5,0.5,0.5)
        )
    ]
)

train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

print(len(train_dataset))
print(len(test_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

images, labels = next(
    iter(train_loader)
)

print(images.shape)
print(labels.shape)

# Model
model = QuCNet(
    num_classses=10
)

model = model.to(device)

# Test Model
images = images.to(device)
outputs = model(images)
print(outputs.shape)

# Loss
criterion = nn.CrossEntropyLoss()

loss = criterion(
    outputs,
    labels.to(device)
)

print(loss)

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr = LR
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

## It is just for testing
# # Test Backpropagation
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()

# # One Batch Training Test
# images, labels = next(
#     iter(train_loader)
# )
# images = images.to(device)
# labels = labels.to(device)
# outputs = model(images)

# loss = criterion(
#     outputs,
#     labels
# )

# optimizer.zero_grad()
# loss.backward()
# optimizer.step()
# print(loss.item())

start_epoch = 0
best_f1 = 0

history = {
    "loss": [],
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1": []
}

if os.path.exists("checkpoint.pth"):
    checkpoint = torch.load(
        "checkpoint.pth"
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    optimizer.load_state_dict(
        checkpoint['optimizer_state_dict']
    )
    scheduler.load_state_dict(
        checkpoint['scheduler_state_dict']
    )
    start_epoch = checkpoint['epoch'] + 1
    best_f1 = checkpoint['f1_score']

    if os.path.exists("history.pth"):
        history =torch.load("history.pth")

    print(
        f"Resuming from epoch {start_epoch}"
    )

# best_loss = float('inf')

for epoch in range(start_epoch, EPOCHS):

    model.train()

    running_loss = 0

    correct = 0

    total = 0

    all_preds = []

    all_labels = []

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # ...
        running_loss += loss.item()

        _, preds = torch.max(
            outputs,
            dim=1
        )

        correct += (
            preds == labels
        ).sum().item()

        total += labels.size(0)

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

    epoch_loss = running_loss / len(train_loader)

    accuracy = 100 * correct/total

    precision = precision_score(
        all_labels,
        all_preds,
        average='weighted'
    )

    recall = recall_score(
        all_labels,
        all_preds,
        average='weighted'
    )

    f1 = f1_score(
        all_labels,
        all_preds,
        average='weighted'
    )

    history["loss"].append(epoch_loss)
    history["accuracy"].append(accuracy)
    history["precision"].append(precision)
    history["recall"].append(recall)
    history["f1"].append(f1)

    # scheduler.step()

    # if epoch_loss < best_loss:

    #     best_loss = epoch_loss

    #     torch.save(
    #         {
    #             'epoch': epoch,
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'loss': epoch_loss
    #         },
    #         # model.state_dict(),
    #         # "best_model.pth"
    #         f"checkpoint_epoch_{epoch+1}.pth"
    #     )

    print(
        f"Epoch {epoch+1}/{EPOCHS} "
        f"Loss: {epoch_loss:.4f}"
        f"Acc: {accuracy:.2f}%"
        f"P: {precision:.4f}"
        f"R: {recall:.4f}"
        f"F1: {f1:.4f}"
    )

    scheduler.step()

    # Save History
    torch.save(
        history,
        "history.pth"
    )

    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': epoch_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,

        },

        "checkpoint.pth"
    )

    if f1 > best_f1:

        best_f1 = f1

        torch.save(
            model.state_dict(),
            "best_model.pth"
        )
        print("Best model saved!")