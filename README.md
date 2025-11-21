📘 MedFusionNet: A Hybrid CNN–Transformer Architecture with Explainability (Grad-CAM + LIME)

MedFusionNet is a compact yet powerful hybrid deep learning architecture designed for medical image classification and explainable AI (XAI). This model combines the spatial feature-learning power of Convolutional Neural Networks (CNNs) with the global contextual reasoning ability of a lightweight Transformer encoder.

The notebook also integrates Grad-CAM and LIME for model interpretability — essential for clinical and research environments where transparency is critical.

🚀 Project Purpose

Medical imaging tasks suffer from three major challenges:

High variation in imaging conditions

Need for robust generalization on small or imbalanced datasets

Requirement for interpretable and transparent model decisions

MedFusionNet directly targets these challenges through:

A hybrid CNN + Transformer backbone

Feature-rich but lightweight architecture

Comprehensive explainability tools

Easy adaptability across imaging modalities (X-ray, CT, retinal scans, etc.)

🧠 Model Architecture: MedFusionNet

MedFusionNet is designed around three major components:

1. A Strong CNN Feature Extractor

Built using:

Conv2D blocks

BatchNorm

ReLU activations

Depthwise Separable Convolutions for computational efficiency

Role:
Extracts local spatial features, textures, and fine-grained medical patterns (lesions, opacity, edges).

2. Patch Embedding Layer

After CNN extraction:

Image features are reshaped into patch tokens

Each patch becomes a vector — forming a token sequence for the transformer

Role:
Converts CNN feature maps into a transformer-readable format.

3. Tiny Transformer Encoder

A lightweight transformer block consisting of:

Multi-Head Self Attention (MHSA)

Layer Normalization

Feed-Forward Network (FFN)

Residual Connections

Role:
Captures global relationships, long-range dependencies, and structural patterns (e.g., bilateral lung comparison, global morphology).

4. Classification Head

Global Average Pooling

Dense Layer

Sigmoid / Softmax Output (depending on task)

🧩 Why This Hybrid Approach?

CNN = strong at local texture learning
Transformer = strong at global structural understanding

MedFusionNet fuses both worlds — making it highly suitable for medical diagnosis tasks that require multi-scale understanding.

📊 Explainability Tools Integrated

The notebook includes two essential XAI methods:

🔥 Grad-CAM

Highlights the exact regions the model focuses on
Useful for:

Lesion localization

Clinical validation

Trust analysis

🍋 LIME

Generates feature-level interpretability
Useful for:

Understanding pixel contribution

Debugging misclassifications

These tools make the model suitable for real-world medical research.

🆚 Previous Solutions & Their Limitations
1. Pure CNN Models (ResNet, EfficientNet, DenseNet)

Limitations:

Excellent at local features but weak at global context

High parameter count for medical research environments

2. Pure Transformers (ViT, Swin)

Limitations:

Require very large datasets

Computationally heavy

Tend to overfit on small medical datasets

3. Vision Transformers with Large Patch Size

Limitations:

Lose local lesion-level details

✅ Why MedFusionNet Is Better (Advantages)
✔ Combines best of CNN + Transformer

Local + global feature learning in one model.

✔ Lightweight & efficient

Tiny Transformer block

Depthwise convolutions
= Works well on smaller GPUs

✔ Higher generalization on small datasets

Perfect for medical imaging datasets where samples are limited.

✔ Built-in Explainability

Grad-CAM + LIME support helps validate clinical trustworthiness.

✔ Adaptable

Can be applied to:

X-rays

CT scans

Ultrasound

Retinal images

Histopathology

📂 Suggested Repository Description (Copy-Paste Ready)

MedFusionNet is a hybrid CNN–Transformer deep learning model designed for medical image analysis with built-in explainability tools. It combines convolutional feature extraction with a lightweight transformer encoder to achieve strong performance on small and imbalanced datasets. The repository includes full training pipelines, interpretability (Grad-CAM & LIME), evaluation metrics, and modular code suitable for research and publication-ready experiments.
