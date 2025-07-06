# DICE-FER: Decoupling Identity Confounders for Enhanced Facial Expression Recognition

This repository contains the implementation of DICE-FER (Decoupling Identity Confounders for Enhanced Facial Expression Recognition), an information-theoretic approach for identity-invariant facial expression recognition.

## Overview

DICE-FER is a novel method that learns disentangled representations for identity and expression without relying on identity labels or resource-intensive image reconstruction. The method uses mutual information estimation to separate identity and expression features, achieving superior performance on multiple datasets.

## Key Features

- **Mutual Information-based Disentanglement**: Uses statistical networks to estimate and maximize mutual information between images and their representations
- **Two-stage Training**: First learns expression representations, then identity representations
- **Adversarial Mutual Information Minimization**: Ensures identity features don't corrupt expression representations
- **No Identity Labels Required**: Works without auxiliary identity annotations
- **Cross-dataset Generalization**: Demonstrates robust performance across diverse domains

## Architecture

The model consists of:
- **ResNet-18 encoders** pre-trained on CASIA-WebFace
- **Expression encoder** (E_exp): Extracts expression representations
- **Identity encoder** (E_id): Extracts identity representations  
- **Statistical networks**: Estimate mutual information
- **Expression classifier**: Final classification layer

## Datasets Supported

- CK+ (Extended Cohn-Kanade)
- Oulu-CASIA
- RAF-DB (Real-world Affective Faces Database)
- AffectNet

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```python
from dice_fer import DICEFER

# Initialize model
model = DICEFER(
    num_classes=7,
    feature_dim=64,
    lambda_exp=1.0,
    lambda_id=1.0,
    lambda_adv=0.1
)

# Train the model
model.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    learning_rate=1e-4
)
```

### Evaluation

```python
# Evaluate on test set
accuracy = model.evaluate(test_loader)
print(f"Test Accuracy: {accuracy:.4f}")

# Get disentanglement metrics
mig_score = model.compute_mig_score(test_loader)
print(f"MIG Score: {mig_score:.4f}")
```

## Results

DICE-FER achieves state-of-the-art performance:

| Dataset | Accuracy | MIG Score |
|---------|----------|-----------|
| CK+     | 99.50%   | 0.430     |
| Oulu-CASIA | 95.50% | 0.418     |
| RAF-DB  | 85.50%   | 0.448     |
| AffectNet | 85.90% | 0.448     |

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{aquib2025dice,
  title={Decoupling Identity Confounders for Enhanced Facial Expression Recognition: An Information-Theoretic Approach},
  author={Aquib, Mohd and Verma, Nishchal K. and Akhtar, M. Jaleel},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
