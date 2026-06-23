import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from model import QuCNet

BATCH_SIZE = 32
EPOCHS = 200
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