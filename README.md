# Quantum-Enhanced Image Classification using Multi-Circuit Embeddings

A PyTorch and PennyLane implementation of a hybrid quantum-classical image classification framework inspired by the QuCNet architecture. This project explores the integration of convolutional feature extraction with trainable quantum circuits and multi-circuit quantum embeddings for image classification.

## Overview

This project implements a hybrid quantum-classical neural network based on the architectural ideas introduced in the QuCNet paper. The implementation includes a classical feature extraction backbone, trainable 4-qubit quantum circuits, hybrid cyclic weight sharing, and an end-to-end training pipeline.

To validate the implementation, experiments are performed on the **CIFAR-10** dataset. The focus of this repository is on implementing and studying the architecture rather than reproducing the original remote sensing experiments reported in the paper.

## Key Features
```Complete implementation of the QuCNet architecture from scratch
Hybrid quantum-classical deep learning framework
Scaled Feature Extractor (SFE)
Multi-Circuit Quantum Embeddings
Trainable 4-qubit quantum circuits using PennyLane
Hybrid Cyclic Weight Sharing (HCWS)
End-to-end differentiable training using PyTorch
Checkpoint saving and automatic training resumption
Training history logging
Accuracy, Precision, Recall and F1-score tracking
Metric visualization utilities
```

## Architecture
```Input Image
      │
      ▼
Special Feature Extractor (SFE)
      │
      ▼
64-D Feature Representation
      │
      ▼
Split into 16 Feature Chunks
      │
      ▼
16 Trainable Quantum Circuits
      │
      ▼
256-D Quantum Embedding
      │
      ▼
Fully Connected Classifier
      │
      ▼
Prediction
```

## Repository Structure
```QuCNet/
│
├── model.py               # QuCNet architecture
├── train.py               # Training pipeline
├── plot_history.py        # Training visualization
├── README.md
│
├── checkpoint.pth         # Training checkpoint
├── best_model.pth         # Best model weights
└── history.pth            # Saved training metrics
```

## Model Components
```Classical Feature Extractor
Convolutional Stem
Residual Classical Blocks
Batch Normalization
Global Average Pooling
Quantum Module
4-qubit trainable quantum circuits
Angle encoding
Entanglement using CNOT gates
Quantum probability measurements
Multi-circuit quantum embeddings
Classification Head
Fully connected classifier
Cross-Entropy loss
Adam optimizer
Cosine Annealing learning rate scheduler
Training Pipeline
```

## The repository includes a complete training workflow featuring:

```Automatic checkpoint saving
Resume training from interruption
Training history persistence
Accuracy computation
Precision computation
Recall computation
F1-score computation
Best model checkpointing
Dataset
```

For implementation verification, the architecture is trained and validated on CIFAR-10.

Note: The original paper evaluates QuCNet on remote sensing image datasets. This repository focuses on faithfully implementing the proposed architecture and validating its functionality using CIFAR-10. The implementation can be adapted to remote sensing datasets such as EuroSAT, UC Merced Land Use, or NWPU-RESISC45.

## Training history includes:

```Loss
Accuracy
Precision
Recall
F1-score
Technologies
Python
PyTorch
PennyLane
NumPy
Matplotlib
Scikit-learn
Current Status
Architecture implementation complete
Forward propagation verified
Backward propagation verified
End-to-end training pipeline implemented
Checkpointing and training resumption implemented
Metric logging implemented
Future Work
Training on EuroSAT
Evaluation on remote sensing datasets
Performance optimization using PennyLane Lightning
Benchmark comparison with the original paper
Hyperparameter optimization
Citation
```

If you use this implementation for academic or research purposes, please cite the original QuCNet paper.

Disclaimer

This repository is an independent implementation developed for research and educational purposes. It is not an official implementation by the original authors.

