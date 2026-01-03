# MS Thesis Document Update Instructions

This document provides step-by-step instructions to update MS_Thesis_Document.docx with content from documentation_humanized.md while preserving equations.

## Overview

The updates involve:
1. **Numbered lists** in sections 1.3 and 1.4
2. **Figure/table captions** with correct chapter numbering
3. **Three new images** to add
4. **Tone changes** in Chapters 2-3 (personal → impersonal)

## Part 1: Numbered Lists (Sections 1.3 & 1.4)

### Section 1.3 Research Questions

**Current format:** "First...", "Second...", "Third..."
**New format:** Numbered list (1., 2., 3.)

**Steps in Word:**
1. Select the three question paragraphs (starting with "First, how accurately...", "Second, which algorithm...", "Third, what are...")
2. Remove "First, ", "Second, ", "Third, " from the beginning of each
3. Apply numbering (Home → Numbering button)
4. Text should now read:
   - "How accurately can machine learning..."
   - "Which algorithm performs best..."
   - "What are the most important parameters..."

### Section 1.4 Research Objectives

**Current format:** "The first was...", "The second objective...", "Third...", "Finally..."
**New format:** Numbered list (1., 2., 3., 4.)

**Steps in Word:**
1. Select the four objective paragraphs
2. Change text:
   - "The first was generating..." → "Generating..."
   - "The second objective involved developing..." → "Developing..."
   - "Third, I wanted to quantify..." → "Quantifying..."
   - "Finally, I planned to validate..." → "Validating..."
3. Apply numbering to all four paragraphs

## Part 2: Add Three New Images

### Image 1: Figure 3.2 - Parameter Distributions
- **Location:** After "The dataset composition breaks down as follows: 1,500 pristine..." in Section 3.6.1
- **Image:** `/Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/simulation/outputs/ml_figures/parameter_distributions.png`
- **Caption:** "Figure 3.2: Distribution of input parameters across the 3,000-sample dataset generated using Latin Hypercube Sampling. The uniform coverage across all parameter ranges demonstrates the effectiveness of LHS in ensuring comprehensive exploration of the design space, covering beam lengths (3-8 m), cross-sectional dimensions (width: 0.2-0.5 m, depth: 0.3-0.7 m), concrete strengths (25-50 MPa), and damage severities (0-20%)."

### Image 2: Figure 4.2 - Correlation Matrix
- **Location:** After Table 4.4 and Equation 13 in Section 4.3.2
- **Image:** `/Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/simulation/outputs/ml_figures/correlation_matrix.png`
- **Caption:** "Figure 4.2: Pearson correlation matrix heatmap showing relationships between all input parameters and output frequencies (Mode 1 and Mode 2). Warm colors (red) indicate strong positive correlations, while cool colors (blue) indicate strong negative correlations. The strong negative correlation between length and frequency (-0.87) and positive correlation between depth and frequency (+0.64) are clearly visible, confirming the theoretical relationships embedded in the Euler-Bernoulli beam equation."

### Image 3: Figure 4.3 - Damage vs Frequency
- **Location:** At the beginning of Section 4.4, before subsection 4.4.1
- **Image:** `/Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/simulation/outputs/ml_figures/damage_vs_frequency.png`
- **Caption:** "Figure 4.3: Comprehensive visualization of the relationship between damage severity and natural frequency reduction across all damage types in the dataset. The plot demonstrates the nonlinear decay in both Mode 1 and Mode 2 frequencies as damage severity increases from 0% to 20%. Different damage types (pristine, uniform corrosion, localized cracks, and random damage) show distinct frequency response patterns, with uniform corrosion generally producing the most significant frequency reductions for equivalent damage levels."

## Part 3: Update Figure Numbers (CRITICAL!)

After adding the three new images, **ALL subsequent figure numbers must be updated**:

### Chapter 3 Figures:
- Figure 3.1 (Research Workflow) - KEEP as 3.1
- Figure 3.2 (Parameter Distributions) - NEW IMAGE
- All other Chapter 3 figures shift accordingly

### Chapter 4 Figures:
- Figure 4.1 (Dataset Distribution) - KEEP as 4.1
- Figure 4.2 (Correlation Matrix) - NEW IMAGE
- Figure 4.3 (Damage vs Frequency) - NEW IMAGE
- **OLD Figure 4.2** (freq_vs_corrosion.png) → NOW Figure 4.4
- **OLD Figure 4.3** (mode_shape_comparison.png) → NOW Figure 4.5
- **OLD Figure 4.4** (severity_impact.png) → NOW Figure 4.6
- **OLD Figure 4.5** (model_comparison.png) → NOW Figure 4.7
- **OLD Figure 4.6** (prediction_vs_actual.png) → NOW Figure 4.8
- **OLD Figure 4.7** (residual_plots.png) → NOW Figure 4.9
- **OLD Figure 4.8** (feature_importance.png) → NOW Figure 4.10
- **OLD Figure 4.9** (shap_summary.png) → NOW Figure 4.11
- **OLD Figure 4.10** (uncertainty_quantification.png) → NOW Figure 4.12
- **OLD Figure 4.11** (coverage_analysis.png) → NOW Figure 4.13
- **OLD Figure 4.12** (hyperparameter_importance.png) → NOW Figure 4.14

**In Word:**
- Use Find & Replace to update figure references:
  - Find: "Figure 4.2:" → Replace: "Figure 4.4:" (for freq_vs_corrosion)
  - Find: "Figure 4.3:" → Replace: "Figure 4.5:" (for mode_shape)
  - And so on for all subsequent figures...

## Part 4: Tone Changes (Chapters 2 & 3)

Use Word's Find & Replace (Ctrl+H / Cmd+H) to change from personal to impersonal tone:

### Chapter 2 Replacements:

| Find | Replace With |
|------|--------------|
| I needed to understand | Understanding... is essential |
| I chose | was chosen |
| I selected | was selected |
| my study | the study |
| I consider | (delete "I consider" phrase) |
| I studied | was studied |
| I was studying | were studied |

**Note:** Review each replacement individually to ensure it makes grammatical sense.

### Chapter 3 Replacements:

| Find | Replace With |
|------|--------------|
| I developed | was developed |
| My approach | The approach |
| I created | was created |
| I applied | was applied |
| I used | was used |
| I generated | was generated |
| I implemented | was implemented / were implemented |
| I formulated | was formulated / were formulated |
| I simulated | was simulated |
| I introduced | was introduced |
| I determined | was determined |
| I solved | was solved |
| I calculated | was calculated |
| I performed | was performed |
| I employed | was employed / were employed |
| I planned | was planned |

**Important Notes:**
- Always use "Find Next" and replace individually - don't use "Replace All"
- Some instances may need "were" instead of "was" depending on plural context
- Check grammar after each replacement
- Skip any instances inside equations or figure captions from Chapter 1 (which should remain personal)

## Part 5: Table Caption Verification

Verify all tables have correct chapter-specific numbering:

**Chapter 1:**
- Table 1.1: Parametric Boundaries for FEM Simulations

**Chapter 2:**
- Table 2.1: ML Algorithm Performance for Beam Frequency Prediction (Das 2023)
- Table 2.2: Experimental Corrosion Effects on Natural Frequency (Zhang et al. 2020)
- Table 2.3: Research Gaps Addressed by This Thesis

**Chapter 3:**
- Table 3.1: FEM Simulation Parameter Ranges

**Chapter 4:**
- Table 4.1: Three-Way Validation Comparison
- Table 4.2: Validation of Corrosion-Frequency Relationship
- Table 4.3: Statistical Summary of FEM-Generated Natural Frequency Dataset
- Table 4.4: Parameter Sensitivity - Pearson Correlation with Mode 1 Natural Frequency
- Table 4.5: Damage Type Comparison
- Table 4.6: Model Performance Metrics
- Table 4.7: Comparison with Literature Benchmarks
- Table 4.8: Bootstrap Confidence Interval Statistics
- Table 4.9: Hyperparameter Search Space
- Table 4.10: Optimized Parameters vs. Default Configuration
- Table 4.11: ML Model Performance Comparison
- Table 4.12: Time Comparison Analysis
- Table 4.13: Literature Comparison

## Completion Checklist

- [ ] Updated sections 1.3 and 1.4 to numbered lists
- [ ] Added Figure 3.2 (Parameter Distributions)
- [ ] Added Figure 4.2 (Correlation Matrix)
- [ ] Added Figure 4.3 (Damage vs Frequency)
- [ ] Updated all subsequent figure numbers in Chapter 4
- [ ] Updated all figure references in text to match new numbering
- [ ] Applied tone changes to Chapter 2 (personal → impersonal)
- [ ] Applied tone changes to Chapter 3 (personal → impersonal)
- [ ] Verified all table captions have correct chapter numbers
- [ ] Verified equations were NOT modified
- [ ] Saved final document

## Notes

- **DO NOT modify equations** - they are in correct LaTeX format for Word
- **Test print/PDF** a few pages to ensure formatting is preserved
- **Track changes** if you want to review modifications before finalizing
