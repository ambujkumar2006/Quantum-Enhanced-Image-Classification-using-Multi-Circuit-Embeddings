import torch
import matplotlib.pyplot as plt


history = torch.load("history.pth")

print(history["loss"])

print(history["accuracy"])

print(history["precision"])

print(history["recall"])

print(history["f1"])

## Plott

loss = history["loss"]
accuracy = history["accuracy"]
precision = history["precision"]
recall = history["recall"]
f1 = history["f1"]

epochs = range(
    1,
    len(loss)+1
)

# Loss Curve
plt.figure()
plt.plot(
    epochs,
    loss
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("training Loss")

plt.grid()

plt.show()

# accuracy Curve
plt.figure()
plt.plot(
    epochs,
    accuracy
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title("Training Accuracy")

plt.grid()

plt.show()

# Precision Curve
plt.figure()
plt.plot(
    epochs,
    precision
)

plt.xlabel("Epoch")
plt.ylabel("Precision")

plt.title("Precision")

plt.grid()

plt.show()

# Recall Curve
plt.figure()
plt.plot(
    epochs,
    recall
)

plt.xlabel("Epoch")
plt.ylabel("Recall")

plt.title("Recall")

plt.grid()

plt.show()

# F1 Score
plt.figure

plt.plot(
    epochs,
    f1
)

plt.xlabel("Epcoh")
plt.ylabel("F1 Score")

plt.title("F1 Score")

plt.grid()

plt.show()

# One Figure with all Metrics
plt.figure(
    figsize=(10,6)
)

plt.plot(
    epochs,
    accuracy,
    label="Accuracy"
)

plt.plot(
    epochs,
    precision,
    label="Precision"
)

plt.plot(
    epochs,
    recall,
    label="Recall"
)

plt.plot(
    epochs,
    f1,
    label="F1"
)

plt.xlabel("Epoch")
plt.ylabel("Score")

plt.title("Training Metrics")

plt.legend()

plt.grid()

plt.show()