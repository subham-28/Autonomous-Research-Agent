## 1. Comparison Matrix (Models | Datasets | Metrics)

| Paper | Models Evaluated (excerpt) | Datasets Used | Evaluation Metrics |
|-------|----------------------------|---------------|--------------------|
| **Foundation Models for Tabular Classification** (Paper 1) | TabPFN, XTab, CARTE, TP‑BERTa, GBDT, AutoML, ICL, FT, FNN, Transformer, SAINT, MLP, ResNet, **FT‑Transformer**, TabNet, CatBoost, XGBoost, RealMLP | OpenMLCC18 (18 classification datasets) | ROC‑AUC |
| **Deep Learning Models for Tabular Data** (Paper 2) | ResNet, **FT‑Transformer**, MLP, GBDT | California Housing (regression), Adult, Helena, Jannis, Higgs, ALOI, Epsilon, Year, Covertype, Yahoo, Microsoft (mixed regression/classification) | RMSE (regression), Accuracy (classification) |

*Notes*  

- **FT‑Transformer** appears both as a model evaluated in Paper 1 and as the novel architecture introduced in Paper 2.  
- Paper 1 focuses **exclusively on classification** (ROC‑AUC), while Paper 2 covers **both regression (RMSE) and classification (Accuracy)**.  
- The “AutoML” entry in Paper 1 aggregates a suite of automated tree‑based pipelines (including GBDT, XGBoost, CatBoost, etc.).  

---

## 2. Key Points of Agreement / Conflict in Their Conclusions  

| Aspect | Paper 1 (Foundation Models) | Paper 2 (DL Comparative) | Agreement / Conflict |
|--------|----------------------------|--------------------------|----------------------|
| **Overall ranking of DL vs. tree‑based models** | *AutoML* (tree‑based pipelines) **outperforms** deep‑learning models on a broad range of datasets. TabPFN (a foundation‑model DL approach) achieves the **highest rank on small datasets**. | Only a **limited DL set** (ResNet, FT‑Transformer, MLP) is compared to GBDT. FT‑Transformer **outperforms ResNet** on most tasks, but **no clear statement** that DL beats GBDT overall. | **Conflict** – Paper 1 claims tree‑based AutoML is generally superior, while Paper 2 does not provide enough evidence to conclude DL dominates tree‑based models. |
...
| **Extension to Regression & Mixed‑Modal Data** | Paper 1 is classification‑only; Paper 2 includes regression but does not compare DL to tree‑based regression models comprehensively. | Expand foundation‑model studies to **regression tasks** and **mixed‑modal tabular data** (e.g., with text or image features), evaluating both **fine‑tuning** and **in‑context** approaches. |
| **Theoretical Insights** | Empirical results dominate; underlying reasons for performance differences (e.g., why TabPFN shines on small data) are not explained. | Provide **theoretical analyses** (e.g., bias‑variance decomposition, inductive bias studies) to explain **when and why** DL or tree‑based models excel. |

Addressing these gaps will lead to **more reliable, reproducible, and actionable conclusions** about the current state and future directions of deep learning versus tree‑based modeling for tabular data.
