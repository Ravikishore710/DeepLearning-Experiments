# Deep Learning Experiments: Optimization, Regularization & Transfer Learning on CIFAR-10

A systematic deep learning experimentation study investigating **generalization, overfitting, regularization, optimization algorithms, learning rates, transfer learning, and fine-tuning** using a CNN on the CIFAR-10 image classification benchmark.

The objective was not simply to build a classifier, but to understand **how different training strategies change model behavior** and to empirically compare them under controlled experiments.

---

## Project Overview

Deep learning performance is affected by much more than neural network architecture.

A model can have enough capacity to fit the training data but still generalize poorly. Regularization can reduce overfitting but may also reduce learning capacity. Optimizers can dramatically change convergence speed. Learning rates can determine whether training is stable or ineffective. Finally, pretrained representations can provide a major advantage over training a small CNN entirely from scratch.

This notebook investigates these effects experimentally.

### Main Questions

* How much does a basic CNN overfit CIFAR-10?
* Can data augmentation reduce the generalization gap?
* How do Dropout and L2 regularization affect training?
* How different are SGD, Momentum, and Adam?
* How sensitive is training to the learning rate?
* How much can transfer learning improve performance?
* Does fine-tuning always improve a pretrained model?
* What happens when we evaluate the final model on a completely held-out test set?

---

# Objectives

The project was designed around the following progression:

```text
Baseline CNN
      ↓
Overfitting Diagnosis
      ↓
Data Augmentation
      ↓
Dropout
      ↓
L2 Regularization
      ↓
Dropout + L2
      ↓
Optimizer Comparison
      ↓
Learning-Rate Comparison
      ↓
Transfer Learning
      ↓
Fine-Tuning
      ↓
Final Evaluation
```

The important principle was:

> **Change one major training factor at a time, observe the effect, and use validation performance to understand generalization.**

---

# Dataset — CIFAR-10

The project uses the **CIFAR-10** dataset.

CIFAR-10 contains:

* 60,000 RGB images
* Image resolution: `32 × 32 × 3`
* 10 classes
* 50,000 training images
* 10,000 test images
* 6,000 images per class

### Classes

```text
airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck
```

### Data preprocessing

Pixel values were converted from:

```text
[0, 255]
```

to:

```text
[0.0, 1.0]
```

Labels were flattened from `(N, 1)` to `(N,)`.

A reproducible seed of `42` was used.

---

# Experimental Data Split

The original 50,000 CIFAR-10 training samples were divided into:

| Split      | Samples | Purpose                                      |
| ---------- | ------: | -------------------------------------------- |
| Training   |  45,000 | Model parameter updates                      |
| Validation |   5,000 | Model comparison and generalization analysis |
| Test       |  10,000 | Final evaluation                             |

The test set was kept separate from model selection.

```text
50,000 original training samples
│
├── 45,000 → Training
│
└── 5,000  → Validation

10,000 official test samples
└── Held out for final evaluation
```

---

# Experimental Environment

| Component   | Configuration                    |
| ----------- | -------------------------------- |
| Framework   | TensorFlow / Keras               |
| Dataset     | CIFAR-10                         |
| Hardware    | Google Colab GPU                 |
| Random Seed | 42                               |
| Input       | 32 × 32 × 3 RGB                  |
| Classes     | 10                               |
| Main Metric | Accuracy                         |
| Loss        | Sparse Categorical Cross-Entropy |

The notebook uses TensorFlow/Keras for model construction, training, and evaluation.

---

# 1️⃣ Experiment 1 — Baseline CNN

The first model establishes a reference point.

### Architecture

```text
Input: 32 × 32 × 3
        ↓
Conv2D: 32 filters, 3×3, ReLU
        ↓
MaxPooling2D
        ↓
Conv2D: 64 filters, 3×3, ReLU
        ↓
MaxPooling2D
        ↓
Flatten
        ↓
Dense: 128, ReLU
        ↓
Dense: 10, Softmax
```

The baseline contains approximately **315K trainable parameters**, with the majority concentrated in the dense layer after flattening.

### Training configuration

```text
Optimizer: Adam
Learning Rate: 0.001
Batch Size: 128
Epochs: 15
Loss: Sparse Categorical Cross-Entropy
```

### Final result

| Metric              |                      Result |
| ------------------- | --------------------------: |
| Training Accuracy   |                  **81.35%** |
| Validation Accuracy |                  **69.56%** |
| Generalization Gap  | **11.79 percentage points** |

The baseline clearly demonstrates overfitting.

Training accuracy continued improving while validation performance stopped improving.

---

# 2️⃣ Experiment 2 — Overfitting Analysis

The baseline training trajectory provides direct evidence of overfitting.

Selected observations:

| Epoch | Train Accuracy | Validation Accuracy |
| ----: | -------------: | ------------------: |
|     1 |         42.54% |              53.40% |
|     3 |         62.07% |              61.54% |
|     6 |         69.57% |              66.60% |
|     9 |         74.31% |              69.36% |
|    10 |         75.67% |              69.60% |
|    12 |         78.42% |              70.28% |
|    15 |     **81.55%** |          **69.56%** |

The important pattern is:

```text
Training performance ↑
Validation performance → / ↓
```

Validation loss also began increasing while training loss continued decreasing.

This is a classic signal that the model is increasingly fitting the training distribution rather than improving generalization.

---

# 3️⃣ Experiment 3 — Data Augmentation

Data augmentation was introduced to expose the model to different versions of the same training examples.

The augmentation pipeline included:

```python
RandomFlip("horizontal")
RandomRotation(0.1)
RandomZoom(0.1)
```

The transformations were applied dynamically during training.

### Results

| Metric              |      Result |
| ------------------- | ----------: |
| Training Accuracy   |  **64.62%** |
| Validation Accuracy |  **62.70%** |
| Generalization Gap  | **1.92 pp** |

### Observation

The generalization gap decreased dramatically:

```text
Baseline gap       ≈ 11.79 pp
Augmentation gap   ≈ 1.92 pp
```

However, validation accuracy was lower than the baseline in this particular experiment.

This is an important result:

> **Reducing overfitting does not automatically mean increasing validation accuracy.**

The model became more regularized and therefore harder to fit, but under the chosen architecture and training budget, augmentation also reduced overall performance.

---

# 4️⃣ Experiment 4 — Dropout

Dropout was introduced into the classification network.

The purpose was to prevent neurons from becoming excessively dependent on specific other neurons.

### Configuration

```text
Dropout Rate: 0.5
Optimizer: Adam
Learning Rate: 0.001
Epochs: 12
```

### Results

| Metric              |      Result |
| ------------------- | ----------: |
| Training Accuracy   |  **68.19%** |
| Validation Accuracy |  **65.20%** |
| Generalization Gap  | **2.99 pp** |

Compared with the baseline:

```text
Baseline gap : 11.79 pp
Dropout gap  : 2.99 pp
```

Dropout substantially reduced the generalization gap.

---

# 5️⃣ Experiment 5 — L2 Regularization

L2 regularization was introduced to penalize large model weights.

Conceptually:

```text
Total Loss =
Data Loss + λ × Weight Penalty
```

The experiment used:

```text
λ = 1e-4
```

### Results

| Metric              |      Result |
| ------------------- | ----------: |
| Training Accuracy   |  **70.84%** |
| Validation Accuracy |  **68.00%** |
| Generalization Gap  | **2.84 pp** |

L2 produced a much smaller generalization gap than the unregularized baseline.

---

# 6️⃣ Experiment 6 — Dropout + L2

Dropout and L2 were combined to investigate whether two different regularization mechanisms would complement each other.

### Results

| Metric              |      Result |
| ------------------- | ----------: |
| Training Accuracy   |  **66.86%** |
| Validation Accuracy |  **64.86%** |
| Generalization Gap  | **2.00 pp** |

The combined approach produced the smallest generalization gap among these regularization experiments.

However, it also produced lower validation accuracy than the baseline.

### Key lesson

Regularization introduces a trade-off:

```text
Less memorization
        ↓
Smaller generalization gap
        ↓
But potentially lower training/validation accuracy
```

The objective is not to minimize the gap blindly.

The objective is to obtain **strong validation performance with a reasonable generalization gap**.

---

# 7️⃣ Experiment 7 — Optimizer Comparison

Three optimization strategies were compared using the same general CNN architecture.

### Optimizers

```text
1. SGD
2. SGD + Momentum
3. Adam
```

### Configuration

| Optimizer | Learning Rate |
| --------- | ------------: |
| SGD       |          0.01 |
| Momentum  |          0.01 |
| Adam      |         0.001 |

Each was trained for 8 epochs.

### Results

| Optimizer | Best Validation Accuracy |
| --------- | -----------------------: |
| SGD       |               **31.32%** |
| Momentum  |               **56.44%** |
| Adam      |               **64.24%** |

### Ranking

```text
Adam
 ↓
Momentum
 ↓
SGD
```

Under this experimental configuration, Adam converged substantially faster than the other two methods.

The comparison demonstrates that optimizer choice can strongly influence convergence behavior even when the network architecture remains similar.

---

# 8️⃣ Experiment 8 — Learning Rate Comparison

Adam was evaluated using three different learning rates.

```text
10^-2
10^-3
10^-4
```

Each configuration was trained for 8 epochs.

### Results

| Learning Rate | Best Validation Accuracy |
| ------------: | -----------------------: |
|        0.0100 |               **60.78%** |
|        0.0010 |               **63.10%** |
|        0.0001 |               **52.08%** |

### Best configuration

```text
Adam + learning rate 0.001
```

The experiment demonstrates the effect of optimization step size:

```text
Too large
   ↓
Aggressive updates

Appropriate
   ↓
Stable progress

Too small
   ↓
Slow convergence
```

Importantly, these conclusions are relative to the **8-epoch experimental budget** used here.

---

# 9️⃣ Experiment 9 — Transfer Learning

After experimenting with CNNs trained from scratch, transfer learning was introduced.

The pretrained architecture used was:

**MobileNetV2 pretrained on ImageNet**

Because CIFAR-10 images are only `32 × 32`, the inputs were resized to:

```text
96 × 96
```

### Architecture

```text
CIFAR-10 Image
      ↓
Resize 32×32 → 96×96
      ↓
Normalization
      ↓
MobileNetV2
      ↓
Global Average Pooling
      ↓
Dropout
      ↓
Dense(10)
      ↓
Softmax
```

The pretrained backbone was initially frozen.

### Frozen-backbone training

Five epochs were performed.

Validation accuracy progressed approximately as:

| Epoch | Validation Accuracy |
| ----: | ------------------: |
|     1 |              86.54% |
|     2 |              86.70% |
|     3 |              86.82% |
|     4 |          **87.16%** |
|     5 |              87.08% |

### Best validation accuracy

**87.16%**

This was substantially higher than the custom CNN baseline:

```text
Baseline CNN       : 69.56%
Transfer Learning  : 87.16%

Improvement         : +17.60 percentage points
```

This was the strongest improvement observed in the project.

---

# 🔟 Experiment 10 — Fine-Tuning

After training with a frozen MobileNetV2 backbone, the final 20 layers of the backbone were unfrozen.

The learning rate was reduced to:

```text
1e-5
```

The purpose was to allow high-level pretrained representations to adapt to CIFAR-10 while keeping parameter updates small.

### Results

| Metric              |      Result |
| ------------------- | ----------: |
| Training Accuracy   |  **87.72%** |
| Validation Accuracy |  **85.62%** |
| Generalization Gap  | **2.10 pp** |

### Comparison

| Approach                 | Best Validation Accuracy |
| ------------------------ | -----------------------: |
| Frozen Transfer Learning |               **87.16%** |
| Fine-Tuning              |               **85.62%** |

Fine-tuning therefore **reduced validation accuracy by 1.54 percentage points** in this experiment.

This is an important empirical result:

> **Fine-tuning is not guaranteed to improve a pretrained model.**

The frozen representation already generalized very well, and fine-tuning the selected layers under this particular configuration did not improve validation performance.

---

# 🏆 Overall Experimental Comparison

| Experiment            | Best Validation Accuracy | Generalization Gap |
| --------------------- | -----------------------: | -----------------: |
| Baseline CNN          |               **69.56%** |           11.79 pp |
| Data Augmentation     |               **62.70%** |            1.92 pp |
| Dropout               |               **65.20%** |            2.99 pp |
| L2                    |               **68.00%** |            2.84 pp |
| Dropout + L2          |               **64.86%** |            2.00 pp |
| SGD                   |               **31.32%** |                  — |
| Momentum              |               **56.44%** |                  — |
| Adam                  |               **64.24%** |                  — |
| Adam, LR = 0.01       |               **60.78%** |                  — |
| Adam, LR = 0.001      |               **63.10%** |                  — |
| Adam, LR = 0.0001     |               **52.08%** |                  — |
| **Transfer Learning** |               **87.16%** |                  — |
| Fine-Tuning           |               **85.62%** |            2.10 pp |

---

# Final Test Evaluation

The final test evaluation performed in the notebook produced:

```text
Test Accuracy: 85.19%
Test Loss:     0.4659
```

### Important experimental note

The test evaluation was performed after the MobileNetV2 model had been fine-tuned.

Therefore:

> **85.19% is the test accuracy of the resulting fine-tuned model, not a clean held-out test measurement of the earlier frozen-transfer-learning checkpoint.**

The frozen transfer-learning model achieved the strongest validation result of **87.16%**, while the subsequently fine-tuned model achieved **85.62% validation accuracy** and **85.19% test accuracy**.

This distinction is intentionally documented to preserve experimental integrity.

---

# Main Findings

## 1. The baseline CNN overfits

The baseline reached:

```text
81.35% training accuracy
69.56% validation accuracy
```

The large gap demonstrates that simply increasing the model's ability to fit training data does not guarantee good generalization.

---

## 2. Regularization strongly reduced the generalization gap

Approximate gaps:

```text
Baseline          → 11.79 pp
Augmentation      →  1.92 pp
Dropout           →  2.99 pp
L2                →  2.84 pp
Dropout + L2      →  2.00 pp
```

The experiments demonstrate that different regularization strategies can substantially reduce overfitting.

---

## 3. Smaller generalization gap ≠ automatically better model

The augmentation model had a much smaller gap than the baseline but also lower validation accuracy.

Therefore model selection should consider both:

```text
Validation performance
+
Generalization behavior
```

rather than optimizing only one statistic.

---

## 4. Adam converged faster under the tested configuration

The optimizer experiment produced:

```text
SGD       → 31.32%
Momentum  → 56.44%
Adam      → 64.24%
```

Adam provided the strongest validation performance within the 8-epoch comparison.

---

## 5. Learning rate significantly affects optimization

Among the tested Adam learning rates:

```text
0.001  → 63.10%  ← best
0.01   → 60.78%
0.0001 → 52.08%
```

The result demonstrates the importance of choosing an appropriate optimization step size.

---

## 6. Transfer learning produced the largest improvement

The transition from a custom CNN to a pretrained MobileNetV2 produced:

```text
69.56%
   ↓
87.16%
```

This represents approximately:

```text
+17.60 percentage points
```

of validation improvement.

This experiment demonstrates the practical value of pretrained visual representations when the target dataset is relatively small and low-resolution.

---

## 7. Fine-tuning was not automatically beneficial

Fine-tuning produced:

```text
Frozen backbone : 87.16%
Fine-tuned       : 85.62%
```

The result reinforces an important practical principle:

> Fine-tuning must be treated as an experiment, not an assumption.

The amount of unfreezing, learning rate, training duration, regularization, and target-domain similarity all influence whether fine-tuning helps.

---

# What This Project Demonstrates

This project goes beyond simply training a CNN.

It demonstrates practical understanding of:

* CNN architecture
* Train/validation/test separation
* Overfitting diagnosis
* Generalization gap analysis
* Data augmentation
* Dropout
* L2 regularization
* Regularization trade-offs
* SGD
* Momentum
* Adam
* Learning-rate sensitivity
* Transfer learning
* Frozen pretrained feature extraction
* Fine-tuning
* Validation-based model comparison
* Held-out test evaluation
* Experimental reproducibility
* Empirical deep learning methodology

---

# Recommended Repository Structure

```text
cifar10-deep-learning-experiments/
│
├── notebooks/
│   └── 06_Deep_Learning_Experiments.ipynb
│
├── outputs/
│   ├── sample_visualizations/
│   ├── training_curves/
│   └── comparison_results/
│
├── README.md
├── requirements.txt
└── LICENSE
```

The notebook contains the complete experimental workflow and recorded outputs.

---

# Running the Project

## 1. Clone the repository

```bash
git clone <your-repository-url>
cd cifar10-deep-learning-experiments
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Launch Jupyter

```bash
jupyter notebook
```

Open:

```text
notebooks/06_Deep_Learning_Experiments.ipynb
```

Alternatively, the notebook can be executed directly in **Google Colab with GPU acceleration**.

---

# Core Dependencies

```text
Python 3.x
TensorFlow
NumPy
Pandas
Matplotlib
```

---

# Reproducibility

A fixed random seed was used:

```python
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)
```

The CIFAR-10 dataset is downloaded through Keras.

Because GPU execution and neural-network optimization can contain nondeterministic operations depending on the environment, exact numerical reproduction may vary slightly.

---

# Experimental Limitations

This project is intentionally an **experimental study**, not a state-of-the-art CIFAR-10 benchmark.

Important limitations include:

* The custom CNN architecture is relatively small.
* Experiments use limited epoch budgets for rapid iteration.
* Hyperparameters were not exhaustively optimized.
* Data augmentation was not extensively tuned.
* The transfer-learning input resolution was increased from 32×32 to 96×96.
* Fine-tuning configuration was limited to a selected set of backbone layers.
* The frozen transfer-learning checkpoint was not separately preserved before fine-tuning, so the final test evaluation corresponds to the subsequently fine-tuned model.
* Results can vary slightly across hardware and TensorFlow versions.

These limitations are deliberately stated because the purpose of the project is **experimental learning and empirical analysis**, rather than claiming state-of-the-art performance.

---

# Conclusion

The experiments demonstrate that deep learning performance is determined by the interaction between:

```text
Architecture
      +
Data
      +
Regularization
      +
Optimization
      +
Learning Rate
      +
Pretraining
      +
Fine-Tuning
```

The most significant result was the transition from a custom CNN trained from scratch to a pretrained MobileNetV2:

```text
Custom CNN
69.56% validation accuracy
        ↓
Transfer Learning
87.16% validation accuracy
```

At the same time, the experiments showed that techniques should not be applied blindly:

* Regularization reduced overfitting but sometimes reduced accuracy.
* Smaller generalization gaps did not necessarily produce better validation scores.
* Adam outperformed SGD and Momentum under the tested configuration.
* Learning rate strongly affected convergence.
* Transfer learning produced a major improvement.
* Fine-tuning unexpectedly reduced validation performance in this setup.

The central lesson is:

> **Deep learning is not only about choosing a model. It is about experimentally understanding how optimization, regularization, representation learning, and generalization interact.**

---

# 👤 Author

**Venkata Ravi Kishore**

GitHub: `@Ravikishore710`

---

# 📄 License

This project is released under the MIT License.
