## 1. Comparison Matrix – Models, Datasets & Evaluation Metrics  

| **Model** | **Paper 1 – “Comprehensive Evaluation of Foundation Models and Neural Networks for Tabular Classification”** | **Paper 2 – “Analysis of the automl challenge series 2015‑2018”** |
|-----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| **TabPFN** | • Datasets: OpenMLCC18 (18 classification datasets)  <br>• Metric: ROC‑AUC (small‑data regime) | *Not evaluated* |
| **XGBoost** | • Datasets: OpenMLCC18  <br>• Metric: ROC‑AUC | *Not evaluated* |
| **CatBoost** | • Datasets: OpenMLCC18  <br>• Metric: ROC‑AUC | • Datasets: Adult, Helena, Jannis, Higgs, ALOI, Epsilon, Covertype, Yahoo, Microsoft (classification)  <br>• Metric: Accuracy |
| **MLP** | • Datasets: OpenMLCC18  <br>• Metric: ROC‑AUC | • Datasets: Adult, Helena, Jannis, Higgs, ALOI, Epsilon, Covertype, Yahoo, Microsoft  <br>• Metric: Accuracy |
| **RealMLP** | • Datasets: OpenMLCC18  <br>• Metric: ROC‑AUC | *Not evaluated* |
| **ResNet** | • Datasets: OpenMLCC18  <br>• Metric: ROC‑AUC | • Datasets: Adult, Helena, Jannis, Higgs, ALOI, Epsilon, Covertype, Yahoo, Microsoft  <br>• Metric: Accuracy |
| **FT‑Transformer** | • Datasets: OpenMLCC18  <br>• Metric: ROC‑AUC | • Datasets: Adult, Helena, Jannis, Higgs, ALOI, Epsilon, Covertype, Yahoo, Microsoft  <br>• Metric: Accuracy |
| **SAINT** | • Datasets: OpenMLCC18  <br>• Metric: ROC‑AUC | *Not evaluated* |
| **TabNet** | • Datasets: OpenMLCC18  <br>• Metric: ROC‑AUC | *Not evaluated* |
| **CARTE** | • Datasets: OpenMLCC18  <br>• Metric: ROC‑AUC | *Not evaluated* |
| **TP‑Berta** | • Datasets: OpenMLCC18 (10/18 datasets failed due to memory limits)  <br>• Metric: ROC‑AUC | *Not evaluated* |
| **AutoGluon** | • Datasets: OpenMLCC18  <br>• Metric: ROC‑AUC | *Not evaluated* |
| **LightGBM** | *Not evaluated* | • Datasets: Adult, Helena, Jannis, Higgs, ALOI, Epsilon, Covertype, Yahoo, Microsoft  <br>• Metric: Accuracy |
| **TabTransformer** | *Not evaluated* | • Datasets: Adult, Helena, Jannis, Higgs, ALOI, Epsilon, Covertype, Yahoo, Microsoft  <br>• Metric: Accuracy |
| **Regression‑specific (California Housing)** | *Not evaluated* | • Dataset: California Housing  <br>• Metric: RMSE |

> **Note:** “*Not evaluated*” means the model was not part of the empirical study in that paper.

---
...
- **If the goal is a thorough, up‑to‑date evaluation of deep‑learning and foundation‑model approaches for tabular **classification**, especially under data‑scarce conditions, **Paper 1** provides the more robust and detailed handling.**  
- **If the goal is to compare a smaller set of models on a heterogeneous mix of classification and regression tasks (including a regression benchmark), Paper 2 offers a useful but narrower perspective.**  

In practice, the two studies are **complementary**: Paper 1 advances the state‑of‑the‑art for classification‑focused tabular DL, while Paper 2 supplies an early‑era baseline for both classification and regression. Future work that unifies their strengths (e.g., a modern benchmark that includes foundation models, fine‑tuning, and both classification and regression metrics) would best address the remaining research gaps.
