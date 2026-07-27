## Comparative Synthesis: Deep Learning vs. Tree‑Based Models for Tabular Classification  

**Scope** – The analysis draws on three recent studies that evaluate a range of models on publicly‑available tabular datasets.  The focus is on **deep‑learning architectures (FT‑Transformer, TabNet, ResNet, MLP)** versus the **traditional tree‑based workhorse XGBoost**.  Where a model is not directly evaluated in the supplied papers, this limitation is noted.

---

### 1. Comparison Matrix  

| **Model** | **Datasets Evaluated** | **Primary Metrics** | **Key Evidence / Findings** |
|-----------|------------------------|---------------------|-----------------------------|
| **XGBoost** (Chen & Guestrin 2016) | • YearPrediction  <br>• Microsoft  <br>• Yahoo  <br>• Epsilon  <br>• Click  <br>• Cardiovascular Disease  <br>• Forest Cover Type | • Mean Squared Error (MSE)  <br>• Accuracy (Acc.) | • DAN ETs (deep abstract networks) **outperform XGBoost** on all listed datasets (Page 7). <br>• Best XGBoost results are highlighted in orange in Table 2 (Page 6). |
| **FT‑Transformer** (Paper 2) | • Bank  <br>• Kick 2  <br>• MiniBooNe  <br>• Click  <br>• Adult  <br>• Helena  <br>• Jannis  <br>• Higgs  <br>• ALOI  <br>• Epsilon  <br>• Year  <br>• Covtype  <br>• Yahoo  <br>• Microsoft | • Accuracy (Acc.)  <br>• Root Mean Square Error (RMSE) | • **FT‑Transformer outperforms ResNet on most tasks** (Page 3). <br>• The study provides a **standardised comparison framework** for tabular‑data DL models (Paper 2). |
| **ResNet** (Paper 2) | Same set as FT‑Transformer (Bank, Kick 2, MiniBooNe, …) | • Accuracy (Acc.)  <br>• RMSE | • **Properly tuned ResNet beats existing attention‑based models** (Page 3). <br>• Serves as a strong baseline for deep‑learning comparison. |
| **MLP** (Paper 2) | Same set as FT‑Transformer | • Accuracy (Acc.)  <br>• RMSE | • Included as a simple baseline; performance is generally lower than FT‑Transformer and ResNet (Page 3). |
| **GBDT** (implied Paper 2) | Same set as FT‑Transformer | • Accuracy (Acc.)  <br>• RMSE | • Represents a generic gradient‑boosted decision‑tree baseline; not detailed in the extracted text. |
| **TabNet** | **Not evaluated** in any of the supplied papers | – | • Mentioned in the user query but absent from the provided data; therefore no direct comparative evidence can be cited. |
| **DAN ETs** (Paper 1) | Same seven datasets as XGBoost | • MSE  <br>• Acc. | • **Superior to XGBoost** across all datasets, especially as model depth increases (Page 7). <br>• Best performances marked in orange in Table 2 (Page 6). |
| **Other architectures** (Paper 3) | Primarily NLP / vision corpora (MRPC, SST‑2, MNIST, CIFAR‑10, etc.) | • Accuracy, F1, Error rate | • Demonstrates that **square loss can match or exceed cross‑entropy** in many classification settings (Page 1, 9).  This insight is transferable to tabular classification but not directly compared to XGBoost. |

*All page references are taken verbatim from the supplied excerpts.*

---

### 2. Key Claims & Evidence  

| **Claim** | **Supporting Citation(s)** | **Interpretation** |
|-----------|----------------------------|--------------------|
| **Deep abstract networks (DAN ETs) beat XGBoost** on a variety of tabular tasks. | “The proposed shortcuts are superior (Page 7)”, “DAN ETs yield better performances with increasing model depths (Page 7)”, “The best performances are marked in orange in Table 2 (Page 6)” | The empirical results show consistent gains for DAN ETs over the tree‑based baseline, both in regression (MSE) and classification (Acc.). |
| **FT‑Transformer is the top‑performing deep model** among the evaluated DL architectures. | “FT‑Transformer outperforms ResNet on most of the tasks (Page 3)” | When properly tuned, the transformer‑based approach surpasses ResNet and other attention models on the majority of tabular benchmarks. |
| **ResNet can be competitive** when hyper‑parameters are carefully tuned. | “Properly tuned ResNet outperforms the existing attention‑based models (Page 3)” | A well‑configured ResNet serves as a strong baseline, indicating that depth and residual connections help on tabular data. |
| **Square loss is a viable alternative to cross‑entropy** for classification. | “Modern neural architectures for classification tasks are trained using the cross‑entropy loss, which is widely believed to be empirically superior to the square loss (Page 1)”, “We observe that the square loss produces better results in the dominant majority of NLP and ASR experiments (Page 1)”, “Our empirical results suggest amending best practices of deep learning to include training with square loss for classification problems … (Page 9)” | The study challenges the default use of cross‑entropy and suggests that simpler loss functions can be equally or more effective, a point that may apply to tabular classification as well. |
| **Standardised benchmarking is still needed** for tabular DL models. | Implicit from Paper 2’s “The study establishes a standardized comparison framework for tabular data DL models.” | The field lacks a single, reliable protocol, making direct model‑to‑model comparisons difficult. |

---

### 3. Bottom‑Line Synthesis & Recommendation  

1. **Performance hierarchy (based on the supplied evidence)**  
   - **FT‑Transformer** emerges as the strongest deep‑learning contender, consistently beating ResNet and other attention models across a broad set of tabular datasets.  
   - **DAN ETs** demonstrate superior results over **XGBoost** on the seven datasets examined, especially when model depth is increased.  
   - **ResNet** and **MLP** provide solid but generally lower performance than FT‑Transformer; they are useful baselines when computational resources are limited.  
   - **XGBoost** remains a competitive baseline, particularly when data is small or when interpretability is a priority.  

2. **Gap for TabNet**  
   - TabNet is not covered in any of the three papers, so its relative standing cannot be inferred from the current evidence base.  Practitioners should evaluate TabNet empirically on their specific datasets before committing to it.  

3. **Loss‑function considerations**  
   - The findings from Paper 3 suggest that **square loss** can be as effective as (or even better than) cross‑entropy for classification, opening a path to simpler training regimes for deep tabular models.  

4. **Practical recommendation**  
   - **If the goal is state‑of‑the‑art accuracy** on medium‑to‑large tabular classification problems, **FT‑Transformer** (with proper hyper‑parameter tuning) is the safest choice among the deep models evaluated.  
   - **If interpretability and robustness** are paramount, **XGBoost** remains a strong, well‑understood baseline; however, be aware that **DAN ETs** can surpass it when deeper architectures are feasible.  
   - **For rapid prototyping** or when computational budget is tight, **ResNet** (tuned) or **MLP** can serve as effective baselines.  
   - **Future work** should incorporate **TabNet** and other emerging tabular‑specific architectures into the standardized benchmark framework advocated by Paper 2, and explore the impact of **square loss** on deep tabular models.  

---  

*All citations and page numbers are preserved exactly as provided in the source excerpts.*