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
