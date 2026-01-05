# Research Questions and Objectives Alignment

This document demonstrates the direct alignment between research questions (Section 1.3) and research objectives (Section 1.4) as per academic best practices.

## Principle

**Research objectives are the answers to research questions.**

Each objective is designed to directly address one specific research question, ensuring the study has a clear logical structure from problem formulation through methodology to results.

---

## Alignment Table

| Research Question | Research Objective | How Objective Answers Question |
|-------------------|-------------------|-------------------------------|
| **RQ1:** How accurately can machine learning predict the fundamental natural frequency of fixed reinforced concrete beams? | **Objective 1:** To develop and validate machine learning models for predicting the fundamental natural frequency of fixed-fixed reinforced concrete beams, achieving prediction accuracy of R² ≥ 0.95 on independent test data. | Objective 1 establishes a specific, measurable target (R² ≥ 0.95) to answer the "how accurately" question. The result (R² = 0.989) provides empirical answer to RQ1. |
| **RQ2:** Which algorithm performs best for this specific application? | **Objective 2:** To perform comprehensive comparative analysis of five regression algorithms—Linear Regression, Random Forest, XGBoost, CatBoost, and Support Vector Regression—using multiple performance metrics (R², MAE, RMSE, training time, and inference speed) to identify the optimal model. | Objective 2 directly answers "which algorithm" by systematically evaluating five candidates using multiple metrics. The comparative analysis identifies CatBoost as optimal. |
| **RQ3:** What are the most important parameters influencing natural frequency? | **Objective 3:** To quantify the influence of beam parameters on natural frequency predictions using SHAP analysis and permutation importance methods, identifying which factors among beam length, cross-sectional dimensions, concrete strength, reinforcement ratio, and corrosion damage most significantly affect frequency. | Objective 3 directly answers "what are the most important parameters" by using explainability methods (SHAP, permutation importance) to rank parameter influence. Results show Length (0.45) > Corrosion (0.20) > Depth (0.15) > Concrete Strength (0.10) > Width (0.03). |

---

## Implementation in Thesis

### Section 1.3: Research Questions
Lists three questions that guide the investigation.

### Section 1.4: Research Objectives
Explicitly states: *"The research objectives directly address the research questions posed in Section 1.3"*

Each objective includes:
- Clear label: **Objective N (Addresses RQN)**
- Specific, measurable targets
- Methods/approach to answer the question
- Expected outcome

### Section 5.2.1: Achievement of Research Objectives
Reports outcomes for each of the three objectives, showing:
- Whether targets were met
- Actual performance achieved
- How results answer the corresponding research question

### Section 5.2.2: Answers to Research Questions
Opening statement: *"Achievement of the three research objectives (Section 5.2.1) enables comprehensive answers to the research questions posed in Chapter 1."*

Each answer includes:
- Question restatement with label: **Question N (Answered through Objective N)**
- Empirical evidence from results
- Closing statement linking back to objective achievement

---

## Example: RQ1 and Objective 1

**Research Question 1 (Section 1.3):**
> How accurately can machine learning predict the fundamental natural frequency of fixed reinforced concrete beams?

**Objective 1 (Section 1.4):**
> To develop and validate machine learning models for predicting the fundamental natural frequency of fixed-fixed reinforced concrete beams, achieving prediction accuracy of R² ≥ 0.95 on independent test data. This objective answers the first research question by establishing whether ML can achieve performance comparable to existing work on metallic beams (Das, 2023).

**Achievement (Section 5.2.1):**
> This objective was achieved and exceeded. The best-performing model (CatBoost) achieved R² = 0.989 with MAE of 3.00 Hz—significantly exceeding the initial goal.

**Answer (Section 5.2.2):**
> Question 1: How accurately can machine learning predict the fundamental natural frequency of fixed reinforced concrete beams? *(Answered through Objective 1)*
>
> CatBoost achieved R² = 0.989 on an independent test set, with MAE of 3.00 Hz and RMSE of 5.61 Hz...
>
> Objective 1 successfully demonstrated that ML models can achieve prediction accuracy exceeding the R² ≥ 0.95 target.

---

## Verification Checklist

- [x] Three research questions defined (Section 1.3)
- [x] Three research objectives defined (Section 1.4)
- [x] Each objective explicitly labeled as addressing specific RQ
- [x] Section 1.4 opening states objectives answer questions
- [x] Section 5.2.1 assesses achievement of three objectives
- [x] Section 5.2.2 links answers to objectives
- [x] Each answer labeled with corresponding objective
- [x] Consistent terminology throughout document

---

*Document created: January 5, 2026*
*Purpose: Demonstrate alignment between research questions and objectives*
