# Attention-Based k-Nearest Neighbors (Attn-KNN)

## Abstract
The k-Nearest Neighbors (KNN) algorithm is a simple yet powerful non-parametric method widely used for classification and regression tasks. However, its core limitation lies in treating all neighbors equally or relying on fixed distance-based weighting schemes.  
This project introduces **Attention-Based KNN (Attn-KNN)**, a novel extension that integrates a learnable attention mechanism to dynamically weight neighbors based on their relevance to the query sample.  
The proposed method bridges classical instance-based learning and modern neural attention mechanisms, enabling adaptive and data-driven neighbor selection.

---

## 1. Introduction

KNN operates under a fundamental assumption: proximity implies similarity. While effective, this assumption is often too rigid in high-dimensional or noisy data settings.

We propose replacing static weighting with a **learned attention function**, allowing the model to infer which neighbors are most informative for prediction.

---

## 2. Methodology

### 2.1 Standard KNN

Given a query sample \( x \), KNN selects the \( K \) closest samples and predicts:

\[
y = \frac{1}{K} \sum_{i=1}^{K} y_i
\]

This formulation assumes uniform contribution from all neighbors.

---

### 2.2 Attention-Based KNN

We redefine prediction as:

\[
y = \sum_{i=1}^{K} \alpha_i \cdot y_i
\]

where \( \alpha_i \) are attention weights such that:

\[
\sum_{i=1}^{K} \alpha_i = 1
\]

---

### 2.3 Attention Mechanism

For each neighbor \( x_i \), a learnable scoring function computes:

\[
\alpha_i = \text{softmax}(f(x, x_i))
\]

where \( f \) is a neural network operating on:

\[
[x, x_i, |x - x_i|]
\]

This allows the model to capture:
- Feature-level interactions  
- Nonlinear similarity  
- Context-dependent importance  

---

## 3. Architecture

The pipeline consists of:

1. Neighbor retrieval using Euclidean distance  
2. Feature construction for each neighbor pair  
3. Attention scoring via a Multi-Layer Perceptron (MLP)  
4. Softmax normalization  
5. Weighted aggregation of neighbor labels  

---

## 4. Training Procedure

The model is trained in a supervised setting:

- Neighbors are retrieved using standard KNN (non-learned)
- The attention module is optimized using cross-entropy loss
- Only the attention weights are learned

This preserves the simplicity of KNN while enhancing its expressiveness.

---

## 5. Experimental Setup

- Dataset: `sklearn.datasets.load_digits`
- Task: Multi-class classification
- Metric: Accuracy
- Baseline: Standard KNN

---

## 6. Results

The Attn-KNN model is expected to:

- Improve classification accuracy over standard KNN  
- Be more robust to noisy neighbors  
- Adaptively focus on informative samples  

Results are logged in:
...


---

## 7. Project Structure
attention_knn/
│
├── models/ # Attention-based model
├── core/ # KNN + training logic
├── data/ # Dataset loading
│
├── config.py # Hyperparameters
├── main.py # Entry point
│
├── outputs/ # Logs, models, plots
└── README.md


---

## 8. Key Contributions

- Introduction of attention mechanism into KNN  
- Learnable neighbor weighting instead of fixed distance metrics  
- Hybridization of non-parametric and neural methods  
- Minimal yet extensible implementation  

---

## 9. Future Work

- Multi-head attention for diverse similarity patterns  
- Integration with learned embeddings (metric learning)  
- Dynamic selection of K  
- Application to larger and more complex datasets  

---

## 10. Conclusion

This work demonstrates that even simple algorithms like KNN can be significantly enhanced by incorporating modern deep learning concepts.  
Attention-based weighting transforms KNN from a purely geometric method into a **context-aware predictive model**, opening new directions for hybrid machine learning approaches.

---

## Installation

```bash
pip install -r requirements.txt
```

## Author

**Abderrahmane CHIBOUT**

Department of Computer Science  
Specialization: Artificial Intelligence and Data Science.
Contribution: Designed the Attention-KNN architecture, implemented the pipeline.

**Aida HANAD**

Department of Computer Science  
Specialization: Artificial Intelligence and Data Science.
Contribution: Trained and evaluated the model.

**Rania MEDLES**

Department of Computer Science  
Specialization: Artificial Intelligence and Data Science.
Contribution: Conducted experiments, Analyzed results.
