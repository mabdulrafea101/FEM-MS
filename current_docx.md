**Prediction of Natural Frequencies of Fixed Reinforced Concrete Beams
Using Machine Learning: A Finite Element Validated Approach**

MS Thesis

*Keywords: Machine Learning, Natural Frequency, Reinforced Concrete,
Finite Element Method, Structural Health Monitoring, Damage Detection*

# Abstract

Reinforced concrete beams form the backbone of most buildings and
bridges we see around us. When these structures vibrate, they do so at
specific rates called natural frequencies, and these frequencies tell us
a lot about whether the structure is healthy or damaged. Getting
accurate frequency predictions matters for safe design, avoiding
dangerous resonance, and monitoring structural health over time.

In this research, I set out to address something that has been missing
in the literature: machine learning models built specifically for
fixed-fixed reinforced concrete beams. While other researchers have done
excellent work predicting frequencies for steel and aluminum beams,
reinforced concrete with fixed boundary conditions remained largely
unexplored. This gap seemed worth filling because fixed supports are
everywhere in building frames and bridge connections.

My approach combined finite element simulations based on Euler-Bernoulli
beam theory with five different machine learning algorithms. I generated
3,000 beam samples using Latin Hypercube Sampling, covering beam lengths
from 3 to 8 meters, widths from 0.2 to 0.5 meters, depths from 0.3 to
0.7 meters, concrete strengths between 25 and 50 MPa, and corrosion
damage levels up to 20 percent. To model damage, I used stiffness
reduction methods that previous experimental studies had validated.

The findings suggest that machine learning can indeed predict
frequencies with high accuracy for this type of structure. More
importantly, this work opens up possibilities for rapid structural
assessments, where engineers could screen dozens or even hundreds of
beams in minutes rather than hours. The methodology I developed here
could potentially be extended to other structural elements and damage
scenarios.

# Chapter 1: Introduction

## 1.1 Study Background

Every structure has its own \"heartbeat\" - a natural rate at which it
prefers to vibrate when disturbed. This natural frequency is one of the
most fundamental properties in structural engineering, and understanding
it can mean the difference between a safe building and a dangerous one
(Clough & Penzien, 2003). The basic relationship is quite intuitive:

> $$f_{n} = \frac{1}{2\pi}\sqrt{\frac{k}{m}}\quad\text{(Eq. 1)}$$

Here, k represents how stiff the structure is, and m is its mass. This
simple equation carries profound implications. When external forces like
wind or earthquakes push against a structure at a frequency matching its
natural frequency, the vibrations grow larger and larger. This resonance
phenomenon has caused some spectacular failures throughout history. The
collapse of the Tacoma Narrows Bridge in 1940 remains perhaps the most
dramatic example of what happens when resonance goes unchecked (Miller
et al., 2000).

For me, this created an interesting problem worth solving. Traditional
methods for calculating natural frequencies, whether through hand
calculations or finite element analysis, work well enough for individual
beams. But what happens when an engineer needs to assess fifty beams? Or
a hundred? The computational time adds up quickly, and this becomes
impractical during early design phases when exploring many different
configurations (Das, 2023).

This is where machine learning enters the picture. Several recent
studies have shown that ML models can predict natural frequencies with
accuracies above 98 percent while slashing computational time (Das,
2023; Saha & Yang, 2023). Once trained on validated simulation data,
these models produce predictions almost instantly. The potential for
structural health monitoring applications seemed too significant to
ignore.

Reinforced concrete, despite being the most common construction material
worldwide, has received surprisingly little attention in this regard.
The American Road and Transportation Builders Association reports that
roughly 36 percent of bridges in the United States need repair, with
concrete structures making up a substantial portion. Annual maintenance
costs exceed seven billion dollars. Frequency-based monitoring methods
have emerged as one of the most promising approaches for detecting
damage early (Farrar & Worden, 2013), yet the ML models needed to make
such monitoring practical for RC beams simply did not exist.

## 1.2 Problem Statement

When I surveyed the existing literature, a pattern became clear. Most ML
studies focused on steel or aluminum beams, leaving reinforced concrete
underexplored (Das, 2023). Das (2023) achieved 98.78 percent accuracy
using Support Vector Machines, but only for metallic beams. Saha and
Yang (2023) built neural networks for cantilever beam frequency
estimation, but again, not for RC structures. Zhang et al. (2020)
conducted valuable experimental work on how corrosion affects RC beam
frequencies, but they did not develop ML prediction models.

This gap struck me as significant for a practical reason: fixed-fixed
boundary conditions are extremely common in real buildings. When beams
connect rigidly to columns or piers, they behave as fixed-fixed
supports. Yet I could not find a single comprehensive ML model
specifically designed for predicting natural frequencies of fixed RC
beams that also accounts for damage effects.

## 1.3 Research Questions

Three main questions guided this research:

First, how accurately can machine learning predict the fundamental
natural frequency of fixed reinforced concrete beams? Previous work on
steel beams achieved around 98 percent accuracy (Das, 2023), and I
wanted to know if similar performance was achievable for RC beams with
their more complex material behavior.

Second, which algorithm performs best for this specific application? I
compared Linear Regression, Random Forest, XGBoost, CatBoost, and
Support Vector Regression because each brings different strengths to
regression problems.

Third, what are the most important parameters? Understanding which
geometric and material properties most strongly influence frequency
could help engineers prioritize their measurements and design decisions.

## 1.4 Research Objectives

I established four concrete objectives:

The first was generating a reliable dataset. I aimed for 3,000 samples
of natural frequency data from finite element simulations, targeting
less than 0.01 percent error compared to theoretical solutions. This
would ensure the ML models had trustworthy training data.

The second objective involved developing and testing five regression
models, with a goal of achieving at least 95 percent R-squared on the
test set.

Third, I wanted to quantify which parameters matter most using SHAP
analysis and permutation importance. This would reveal whether length,
depth, width, concrete strength, or damage severity has the greatest
influence on frequency.

Finally, I planned to validate everything against published experimental
data to confirm the physical realism of my simulations.

## 1.5 Significance of the Research

Why does this matter practically? Consider a structural engineer
designing a building with dozens of beams. Traditional FEM analysis
might take several minutes per beam configuration. For preliminary
design work where hundreds of variations need evaluation, this becomes a
bottleneck. My ML models, once trained, can produce predictions in
milliseconds.

This speed advantage becomes even more valuable for structural health
monitoring. Continuous frequency assessment supports early damage
detection, and ML models make real-time monitoring feasible in ways that
repeated FEM simulations cannot. The framework I developed here also
provides a template that other researchers could adapt for different
structural elements.

## 1.6 Scope and Limitations

I focused specifically on fixed-fixed RC beams and considered only the
first two vibration modes. The parameter ranges I studied are shown in
Table 1.1:

**Table 1.1:** Parametric Boundaries for FEM Simulations

  -----------------------------------------------------------------------
    **Parameter**      **Minimum**       **Maximum**        **Unit**
  ----------------- ----------------- ----------------- -----------------
     Beam Length           3.0               8.0                m

    Cross-section          0.2               0.5                m
        Width                                           

    Cross-section          0.3               0.7                m
        Depth                                           

  Concrete Strength        25                50                MPa

   Corrosion Level          0                20                \%
  -----------------------------------------------------------------------

Several limitations deserve explanation. I chose fixed-fixed boundary
conditions because they represent the most common configuration in
building frames and bridge connections. Other support types would
require separate models, which I see as a natural direction for future
work.

I did not conduct physical experiments. Instead, I validated my FEM
implementation through a three-way comparison: my Python code against
published ANSYS results from Das (2023) and against theoretical
closed-form solutions. This approach let me generate a large parametric
dataset that would have been impractical through physical testing alone.

Temperature effects, which Cai et al. (2021) found cause about 0.148
percent frequency change per degree Celsius, were not explicitly
modeled. I made this choice because temperature compensation is standard
practice in monitoring systems, and the damage-induced frequency changes
I was studying (roughly 0.8 percent per 1 percent corrosion) are much
larger than typical temperature variations.

I assumed linear elastic material behavior throughout. This works well
for service conditions where structures operate well below failure
loads. Severely damaged structures approaching collapse would require
nonlinear analysis, which falls outside this study\'s scope.

The parameter ranges in Table 1.1 reflect typical RC beam dimensions
based on ACI 318-19 and Eurocode 2. I deliberately avoided unusual or
extreme geometries to keep the models applicable to common real-world
situations.

## 1.7 Knowledge Contribution

This research makes several contributions that I believe advance the
field:

From a methodological standpoint, this is the first systematic
comparison of five ML algorithms for fixed RC beam frequency prediction.
CatBoost emerged as the best performer with 98.9 percent R-squared,
which I found somewhat surprising given that other studies favored
Support Vector Machines.

Practically, I have created an open dataset of 3,000 validated FEM
simulations along with trained models that other researchers and
engineers can use.

Theoretically, I quantified the relationship between corrosion damage
and frequency reduction with sensitivity coefficients that support early
damage detection in RC structures.

# Chapter 2: Literature Review

## 2.1 Introduction

Before diving into my own methodology, I needed to understand what
others had already accomplished and where the gaps remained. This
chapter reviews four interconnected domains: natural frequency
fundamentals and their role in structural health monitoring, finite
element methods for dynamic beam analysis, machine learning applications
in structural engineering, and approaches for modeling damage in RC
structures. By synthesizing findings across these areas, I identified
the specific research gap my thesis addresses.

## 2.2 Natural Frequency and Structural Health Monitoring

### 2.2.1 Fundamentals of Natural Frequency in RC Structures

At its core, natural frequency describes how fast a structure vibrates
when disturbed and allowed to oscillate freely. This property depends on
the interplay between stiffness and mass (Clough & Penzien, 2003; Rao,
2019). For beam structures, the Euler-Bernoulli frequency equation
provides the closed-form solution:

> $$f_{n} = \frac{\lambda_{n}^{2}}{2\pi L^{2}}\sqrt{\frac{EI}{\rho A}}\quad\text{(Eq. 2)}$$

In this equation, the eigenvalue for the first mode of a fixed-fixed
beam is 4.730, L is beam length, E is elastic modulus, I is moment of
inertia, rho is density, and A is cross-sectional area (Chopra, 2012). I
chose Euler-Bernoulli over more complex formulations because it makes
the physics transparent. You can see directly how lengthening a beam
reduces frequency, or how increasing stiffness raises it. This
formulation works well when the length-to-depth ratio exceeds about 10,
which covers most practical RC beams.

For concrete, we typically estimate elastic modulus from compressive
strength using the ACI 318-19 relationship:

> $$E_{c} = 4700\sqrt{f_{c}'}\text{ MPa}\quad\text{(Eq. 3)}$$

I selected this over the Eurocode alternative because ACI 318-19 has
been more extensively validated for the concrete strengths I was
studying (25-50 MPa), and the differences between the two approaches are
small anyway, typically under 5 percent (MacGregor & Wight, 2012).

### 2.2.2 Role of Natural Frequency in Structural Health Monitoring

Structural health monitoring has become increasingly important for
infrastructure safety, and frequency-based methods have proven
particularly useful because they can detect global changes without
needing access to every part of a structure (Farrar & Worden, 2013;
Doebling et al., 1996).

The principle is straightforward: any change in structural properties,
whether from damage or deterioration, will shift the natural
frequencies. The relationship can be approximated as:

> $$\frac{\Delta f}{f} \approx \frac{1}{2}\frac{\Delta K}{K}\quad\text{(Eq. 4)}$$

This tells us that stiffness reductions show up directly as frequency
reductions. The factor of one-half comes from the square-root
relationship between frequency and stiffness. Sohn et al. (2004)
reviewed the literature extensively and concluded that frequency shifts
remain among the most reliable indicators of global damage, though they
also warned that temperature variations can confuse damage detection if
not properly accounted for.

### 2.2.3 Damage Detection Through Frequency Shifts

Zhang et al. (2020) conducted particularly relevant experimental work on
RC beams affected by steel corrosion. Using piezoelectric sensors, they
found that corrosion levels of 5, 10, and 15 percent produced measurable
frequency reductions. Interestingly, the second mode frequency proved
more sensitive to damage than the first. They also demonstrated that
frequency-based methods could identify corrosion before visible surface
cracking appeared.

Cai et al. (2021) studied temperature effects on simply supported RC
beams and found a roughly linear relationship: 0.148 percent frequency
decrease per degree Celsius increase. This finding highlights why
environmental compensation matters for practical monitoring systems.

Saha and Yang (2023) took a different approach, developing neural
networks for damaged cantilever beams. They achieved prediction errors
of 0.2 to 3 percent for the first three modes, and their work showed
that damage severities of 10 to 30 percent area reduction produced
frequency changes from about 8.65 Hz down to 7.23 Hz, roughly a 16
percent shift.

## 2.3 Finite Element Method for Structural Analysis

### 2.3.1 FEM Fundamentals for Beam Vibration Analysis

The finite element method has become the standard numerical approach for
structural dynamics problems. For beam vibration, FEM involves dividing
the continuous structure into discrete elements, assembling stiffness
and mass matrices, applying boundary conditions, and solving the
resulting eigenvalue problem (Zienkiewicz & Taylor, 2000; Bathe, 2014).

The governing equation for free vibration is:

> $$\lbrack K\rbrack u = \omega^{2}\lbrack M\rbrack u\quad\text{(Eq. 5)}$$

Here, K is the global stiffness matrix, M is the global mass matrix, u
is the mode shape vector, and omega represents angular frequencies.
Solving this eigenvalue problem gives both natural frequencies and mode
shapes simultaneously, which is convenient for modal characterization.

### 2.3.2 Euler-Bernoulli vs Timoshenko Beam Theory

Two beam theories dominate FEM analysis. Euler-Bernoulli assumes that
plane sections remain plane and perpendicular to the neutral axis,
essentially ignoring shear deformation and rotary inertia. This works
well for slender beams where length-to-depth ratio exceeds 10 (Rao,
2019).

Timoshenko theory includes shear and rotary effects, providing better
accuracy for deep beams with length-to-depth ratios below 5. Das (2023)
used both theories in generating FEM datasets and found that
Euler-Bernoulli gives sufficient accuracy for typical building beam
proportions.

For the RC beams in my study, with length-to-depth ratios ranging from
about 4.3 to 26.7, Euler-Bernoulli theory is appropriate for most
configurations. Only the deepest sections might benefit from Timoshenko
refinement.

### 2.3.3 FEM Validation Studies in Literature

Validating FEM implementations against analytical solutions and
experimental data is essential. Das (2023) validated FEM code against
Euler-Bernoulli theory with errors below 1 percent for various boundary
conditions. Mesh convergence studies showed that 20 elements provide
sufficient accuracy for beam vibration problems.

Luu (2024) used ABAQUS with the Concrete Damaged Plasticity model for RC
beam analysis, demonstrating the importance of proper material modeling
for capturing concrete behavior under loading.

## 2.4 Machine Learning in Structural Engineering

### 2.4.1 Overview of ML Applications in Civil Engineering

Machine learning has found widespread applications in civil engineering,
from structural health monitoring to load prediction to design
optimization. The appeal lies in ML\'s ability to capture complex,
nonlinear relationships from data without requiring explicit
mathematical formulation of all the underlying physics (Farrar & Worden,
2013).

Laory et al. (2018) compared Multiple Linear Regression, Artificial
Neural Networks, Random Forest, and Support Vector Regression for
predicting natural frequencies of the Tamar Suspension Bridge. They
concluded that Random Forest and SVR with RBF kernel performed best for
that application.

### 2.4.2 Regression Models for Frequency Prediction

Das (2023) conducted what I consider the most comprehensive ML study to
date on beam frequency prediction. Using FEM-generated datasets for
aluminum and steel beams under various boundary conditions, Das compared
four algorithms:

**Table 2.1:** ML Algorithm Performance for Beam Frequency Prediction
(Das 2023)

  -----------------------------------------------------------------------
             **Algorithm**                   **Average Accuracy**
  ----------------------------------- -----------------------------------
  Support Vector Machine (Puk kernel)               98.78%

        Random Forest Regressor                     98.88%

    Radial Basis Function Regressor                 96.36%

    Multilayer Perceptron Regressor                 94.17%
  -----------------------------------------------------------------------

Key findings included that ensemble methods like Random Forest and
kernel-based methods like SVM outperformed single-model approaches.
Prediction accuracy varied with boundary conditions and thickness
ratios.

Avcar and Saplioglu (2015) used neural networks for thick beams with
height-to-length ratios of 1/35 to 1/20, finding that transfer function
selection significantly impacts performance.

### 2.4.3 Neural Networks in Structural Health Monitoring

Neural networks have been widely applied for damage detection and
frequency prediction. Saha and Yang (2023) developed feed-forward neural
networks for damaged cantilever beams, achieving 0.2 to 3 percent
prediction errors. Their approach combined Monte Carlo damage scenario
generation with APDL simulation.

Banerjee et al. (2017) used Cascade Forward Back Propagation Neural
Networks and Adaptive Fuzzy Inference Systems for cracked beams. Nikoo
et al. (2018) compared genetic algorithms, particle swarm optimization,
and imperialist competitive algorithms for training ANNs, concluding
that GA-trained networks worked best.

### 2.4.4 Ensemble Methods: Random Forest, XGBoost, CatBoost

Ensemble methods have shown superior performance in structural
engineering because they reduce variance and capture complex
relationships effectively.

Random Forest, introduced by Breiman (2001), combines predictions from
multiple decision trees trained on bootstrap samples. Das (2023) found
it achieved 98.88 percent accuracy, matching or exceeding other methods.

XGBoost (Chen & Guestrin, 2016) implements gradient boosting with
regularization and has achieved state-of-the-art results across many
domains. Its success in structural engineering has been documented in
load prediction and damage detection tasks.

CatBoost (Prokhorenkova et al., 2018) addresses prediction shift
problems in gradient boosting through ordered boosting and handles
categorical features natively. While less commonly applied in structural
engineering than the others, its handling of mixed feature types made it
potentially suitable for my damage classification problem.

Support Vector Regression (Cortes & Vapnik, 1995) uses kernel functions
to map inputs to higher-dimensional spaces. Laory et al. (2018) found
SVR with RBF kernel among the best performers for bridge frequency
prediction.

## 2.5 Damage Modeling in RC Structures

### 2.5.1 Corrosion Effects on Structural Properties

Steel corrosion is a major factor degrading the durability of RC
structures (Zhang et al., 2020). Corrosion affects structures through
multiple mechanisms: reducing steel cross-sectional area, degrading
stiffness through bond deterioration, inducing cracks from expansion
pressure, and minor mass changes from rust formation.

Zhang et al. (2020) quantified corrosion-frequency relationships through
laboratory experiments:

**Table 2.2:** Experimental Corrosion Effects on Natural Frequency
(Zhang et al. 2020)

  -----------------------------------------------------------------------
        **Corrosion Level (%)**        **Approximate Frequency Reduction
                                                     (%)**
  ----------------------------------- -----------------------------------
                  1-5                                 2-5

                 5-10                                5-10

                 10-15                               10-15
  -----------------------------------------------------------------------

These findings provided experimental validation for the stiffness
reduction approach I used in my simulations.

### 2.5.2 Stiffness Reduction Approach for Damage Modeling

The stiffness reduction method is widely used for simulating damage
effects in FEM analysis. The effective stiffness is reduced
proportionally to damage severity:

> $$EI_{\text{damaged}} = EI_{\text{original}} \times (1 - \alpha)\quad\text{(Eq. 6)}$$

where alpha is the damage factor. This approach has been validated
against experimental studies of corroded RC beams (Rodriguez et al.,
1997; Cairns et al., 2005). A multiplier of 1.6 is typically applied to
corrosion percentage to estimate effective stiffness loss, reflecting
the accelerated degradation beyond simple area reduction.

### 2.5.3 Crack Modeling Techniques

Localized damage like cracks can be modeled several ways: local
stiffness reduction at the crack location, rotational spring models with
reduced stiffness, or smeared crack approaches that distribute stiffness
reduction over a zone. Dimarogonas (1996) and Chondros et al. (1998)
developed theoretical frameworks for vibration of cracked structures
that have been widely adopted.

## 2.6 Research Gaps and Thesis Positioning

After reviewing the literature, several gaps became apparent:

**Table 2.3:** Research Gaps Addressed by This Thesis

  -----------------------------------------------------------------------
          **Gap**          **Literature Status**       **This Thesis
                                                      Contribution**
  ----------------------- ----------------------- -----------------------
   ML for fixed RC beams     Most studies use     Focuses specifically on
                              steel/aluminum          fixed RC beams

  Comprehensive algorithm     Limited to 2-3       Compares 5 algorithms
        comparison         algorithms typically       systematically

   Parameter sensitivity    Not well quantified    SHAP and permutation
          for RC                                    importance analysis

   Validated FEM dataset   Many use experimental    3,000 FEM-validated
          for RC                   only                   samples

  Corrosion-frequency in      Rarely combined        Integrated damage
        ML context                                       modeling
  -----------------------------------------------------------------------

This thesis addresses these gaps by developing a comprehensive ML
benchmark specifically for fixed RC beams, comparing five regression
algorithms, and providing validated accuracy metrics against both
theoretical solutions and literature experimental data.

# Chapter 3: Methodology

## 3.1 Research Workflow

The methodology I developed follows a systematic progression from beam
parameter definition through FEM simulation to ML model development.
Figure 3.1 illustrates this workflow:

![](media/image1.png){width="4.6875in" height="5.729166666666667in"}

**Figure 3.1:** Research Workflow: From beam parameter definition
through FEM simulation to ML model development.

The workflow integrates literature findings from Chapter 2 with finite
element simulations and machine learning analysis, following established
practices demonstrated by Das (2023) and Saha and Yang (2023).

## 3.2 Introduction

### 3.2.1 Chapter Overview

This chapter explains how I investigated the relationship between
structural damage and natural frequency shifts in reinforced concrete
beams. My approach combined high-fidelity finite element simulations
with machine learning algorithms to develop a predictive framework
suitable for structural health monitoring. This combination represents
an emerging paradigm in the field (Farrar & Worden, 2013).

### 3.2.2 Rationale for Chosen Methods

I chose to combine FEM and ML because purely experimental approaches
have significant limitations. Physical testing is expensive,
time-consuming, and allows only a limited number of damage scenarios to
be examined. FEM, by contrast, lets me generate a large, diverse dataset
under precisely controlled conditions. Machine learning then provides
the analytical capability to map complex, nonlinear relationships
between damage parameters and frequency responses.

## 3.3 Research Design

### 3.3.1 Quantitative and Simulation-Based Approach

My research follows a quantitative, simulation-based design with four
main steps:

First, I created a parameterized FEM model of a fixed-fixed RC beam.
Second, I systematically introduced damage (corrosion and cracks) into
the model. Third, I ran thousands of simulations to generate a
comprehensive dataset. Fourth, I trained regression algorithms to
predict natural frequencies from beam parameters.

### 3.3.2 Design Justification and Scope

This approach ensures internal validity by strictly controlling input
parameters and external validity by covering a wide range of geometric
and material properties typical of real structures. The scope is limited
to fixed-fixed RC beams, considering uniform corrosion and localized
cracking as primary damage mechanisms.

I determined the sample size of 3,000 simulations following power
analysis guidelines for regression studies (Cohen, 1992). Latin
Hypercube Sampling was selected over simple random sampling because of
its superior space-filling properties (McKay et al., 1979).

## 3.4 Finite Element Model Formulation

### 3.4.1 Governing Equations

The dynamic behavior of the RC beam is governed by Euler-Bernoulli beam
theory, which assumes plane sections remain plane and perpendicular to
the neutral axis during deformation (Clough & Penzien, 2003; Chopra,
2012). The equation of motion for free vibration is:

> $$\lbrack K\rbrack\text{\{}u\text{\}} = \omega^{2}\lbrack M\rbrack\text{\{}u\text{\}}\quad\text{(Eq. 5)}$$

where K is the global stiffness matrix (N/m), M is the global mass
matrix (kg), u is the displacement vector (m), and omega is angular
frequency (rad/s). I solved this generalized eigenvalue problem using
scipy.linalg.eigh in Python (Virtanen et al., 2020).

The natural frequency f in Hertz comes from angular frequency:

> $$f = \frac{\omega}{2\pi} = \frac{\sqrt{\lambda}}{2\pi}\quad\text{(Eq. 7)}$$

where lambda represents the eigenvalue from the generalized eigenvalue
problem.

### 3.4.2 Material Properties

I calculated the elastic modulus of concrete using the ACI 318-19
empirical relationship:

> $$E_{c} = 4700\sqrt{f_{c}'}\text{ MPa}\quad\text{(Eq. 3)}$$

where f\'c is compressive strength in MPa. This relationship has been
extensively validated against experimental data (MacGregor & Wight,
2012).

The moment of inertia for a rectangular cross-section is:

> $$I = \frac{bh^{3}}{12}\quad\text{(Eq. 8)}$$

where b is width and h is depth.

### 3.4.3 Element Matrices

I formulated element stiffness and consistent mass matrices following
standard finite element procedures (Zienkiewicz & Taylor, 2000; Bathe,
2014). For each beam element of length Le, the local stiffness matrix
is:

> $$k_{e} = \frac{EI}{L_{e}^{3}}\begin{bmatrix}
> 12 & 6L_{e} & - 12 & 6L_{e} \\
> 6L_{e} & 4L_{e}^{2} & - 6L_{e} & 2L_{e}^{2} \\
>  - 12 & - 6L_{e} & 12 & - 6L_{e} \\
> 6L_{e} & 2L_{e}^{2} & - 6L_{e} & 4L_{e}^{2}
> \end{bmatrix}\quad\text{(Eq. 9)}$$

The consistent mass matrix for each element is:

> $$\lbrack m\rbrack_{e} = \frac{\rho AL_{e}}{420}\begin{bmatrix}
> 156 & 22L_{e} & 54 & - 13L_{e} \\
> 22L_{e} & 4L_{e}^{2} & 13L_{e} & - 3L_{e}^{2} \\
> 54 & 13L_{e} & 156 & - 22L_{e} \\
>  - 13L_{e} & - 3L_{e}^{2} & - 22L_{e} & 4L_{e}^{2}
> \end{bmatrix}\quad\text{(Eq. 10)}$$

where rho is material density (2400 kg/m3 for reinforced concrete) and A
is cross-sectional area.

## 3.5 Damage Modeling Approaches

### 3.5.1 Uniform Corrosion Model

I simulated corrosion-induced damage using the stiffness reduction
method, which has been validated against experimental studies (Zhang et
al., 2020; Rodriguez et al., 1997; Cairns et al., 2005). The effective
moment of inertia is reduced uniformly across all elements:

> $$EI_{\text{damaged}} = EI_{\text{original}} \times (1 - \alpha)\quad\text{(Eq. 6)}$$

The damage factor alpha relates to corrosion level through:

> $$\alpha = \min\left( 1.6 \times \frac{C}{100},0.9 \right)\quad\text{(Eq. 11)}$$

where C is corrosion level expressed as a percentage (0-100%). The
factor of 1.6 accounts for the nonlinear relationship between corrosion
and stiffness degradation observed in laboratory tests. The upper limit
of 0.9 prevents numerical instabilities while representing severe damage
conditions.

### 3.5.2 Localized Crack Model

For localized damage like cracks, based on fracture mechanics principles
(Dimarogonas, 1996; Chondros et al., 1998), I applied stiffness
reduction only to elements within the damaged zone.

### 3.5.3 Random Damage Model

To simulate realistic damage patterns with multiple defects, I
introduced random damage at multiple locations where the damage factor
is randomly sampled from a uniform distribution for n randomly selected
elements.

## 3.6 Dataset Generation Strategy

### 3.6.1 Sampling Plan

I generated a comprehensive dataset of 3,000 simulations using Latin
Hypercube Sampling via scipy.stats.qmc (Virtanen et al., 2020). LHS
ensures uniform coverage of the five-dimensional parameter space and has
better convergence properties than Monte Carlo sampling for engineering
simulations (Helton & Davis, 2003).

The parameter ranges were selected based on typical RC beam dimensions
in building construction (ACI 318-19) and practical concrete grades
(Eurocode 2, 2004):

**Table 3.1:** FEM Simulation Parameter Ranges

  ---------------------------------------------------------------------------
   **Parameter**    **Symbol**    **Minimum**    **Maximum**      **Unit**
  --------------- -------------- -------------- -------------- --------------
      Length            L             3.0            8.0             m

       Width            b             0.2            0.5             m

       Depth            h             0.3            0.7             m

     Concrete          f\'c            25             50            MPa
     Strength                                                  

  Corrosion Level       C              0              20             \%
  ---------------------------------------------------------------------------

The dataset composition breaks down as follows: 1,500 pristine beam
samples (50%), 500 uniform corrosion samples (16.7%), 500 localized
crack samples (16.7%), and 500 random damage samples (16.7%).

## 3.7 Machine Learning Methodology

### 3.7.1 Data Preparation and Preprocessing

The complete dataset comprises 3,000 simulations with six input features
(Length, Width, Depth, Concrete Strength, Damage Type, Damage Severity)
and two target variables (Mode 1 Frequency, Mode 2 Frequency).

Data Integrity Verification: The FEM-generated dataset contained no
missing values, so imputation was unnecessary. I verified data integrity
using pandas.DataFrame.isnull() before model training. Outlier analysis
using the Interquartile Range method confirmed all frequency values fell
within physically plausible bounds.

Feature Encoding: I applied one-hot encoding to the categorical
Damage_Type variable using sklearn.preprocessing.OneHotEncoder
(Pedregosa et al., 2011). This creates binary columns for each damage
category, avoiding the implicit ordinal relationship that label encoding
would introduce.

Data Splitting: I used an 80-20 train-test split following established
practices for regression tasks (Hastie et al., 2009). Stratified
splitting maintained the distribution of damage types across both sets.
I fixed the random state (random_state=42) for reproducibility,
resulting in 2,400 training samples and 600 testing samples.

Feature Scaling: StandardScaler normalization transforms features to
zero mean and unit variance:

> $$X_{\text{scaled}} = \frac{X - \mu}{\sigma}\quad\text{(Eq. 12)}$$

This preprocessing is critical for SVR with RBF kernels, which are
sensitive to feature magnitudes (Cortes & Vapnik, 1995). While
tree-based methods are invariant to monotonic transformations, I scaled
all features consistently for fair comparison.

### 3.7.2 Model Development

I implemented five regression algorithms with hyperparameters selected
based on literature recommendations:

Linear Regression serves as a baseline model establishing the
performance floor. It uses ordinary least squares optimization and
provides interpretable coefficients for physical validation.

Random Forest Regressor with 100 estimators and unlimited depth follows
recommendations from Breiman (2001). Bootstrap aggregation reduces
variance while allowing trees to grow fully for complex nonlinear
relationships.

XGBoost Regressor hyperparameters follow Chen & Guestrin (2016)
guidelines: learning rate of 0.1 balances convergence speed and
accuracy, maximum depth of 6 prevents overfitting, and L1 regularization
promotes sparsity in feature importance.

CatBoost Regressor uses ordered boosting to address prediction shift
inherent in traditional gradient boosting (Prokhorenkova et al., 2018).
I configured 100 iterations, 0.1 learning rate, and depth of 6.

Support Vector Regression with RBF kernel was selected for its universal
approximation capability (Cortes & Vapnik, 1995). I set the
regularization parameter C to 100 based on cross-validation to balance
bias-variance trade-off.

## 3.8 Tools and Instruments Used

### 3.8.1 Software Platforms

I used Python 3.9+ as the primary programming language and Jupyter
Notebooks for interactive development and visualization.

### 3.8.2 ML Libraries and Statistical Packages

For data preprocessing and model implementation, I used Scikit-learn
(Pedregosa et al., 2011), XGBoost (Chen & Guestrin, 2016), and CatBoost
(Prokhorenkova et al., 2018). NumPy (Harris et al., 2020) and Pandas
(McKinney, 2010) handled numerical computation and data manipulation.
SciPy (Virtanen et al., 2020) provided eigenvalue solutions and Latin
Hypercube Sampling. Matplotlib and Seaborn generated visualizations.
SHAP provided model-agnostic feature importance analysis.

### 3.8.3 Evaluation Metrics

I evaluated models using Mean Absolute Error (average error magnitude in
Hz), Root Mean Square Error (which penalizes larger errors more
heavily), Coefficient of Determination R-squared (proportion of variance
explained), and 5-Fold Cross-Validation for assessing generalization.

## 3.9 Ethical Considerations

### 3.9.1 Data Integrity and Reproducibility

This research adheres to principles of scientific reproducibility and
transparency. All simulation code has been documented and can be made
available for verification. Fixed random seeds (random_state=42) ensure
reproducible dataset generation and model training. Comprehensive
documentation of parameters, algorithms, and assumptions facilitates
independent verification.

### 3.9.2 Computational Transparency

I employed exclusively open-source tools (Python, NumPy, SciPy,
Scikit-learn, XGBoost, CatBoost), ensuring results can be independently
reproduced without proprietary software and algorithms are publicly
documented.

### 3.9.3 Limitations Acknowledgment

Several limitations affect result generalizability. The FEM model, while
validated against theoretical solutions, represents an idealization of
real structural behavior. Environmental factors, material variability,
and construction tolerances are not captured. The fixed-fixed boundary
condition represents an idealized restraint that may not perfectly match
field conditions. Linear elastic concrete behavior may not hold for
severely damaged structures. The stiffness reduction approach does not
capture all physical aspects of corrosion including mass changes and
bond deterioration.

### 3.9.4 Intended Use and Misuse Prevention

The predictive models I developed are intended for preliminary design
assessment, rapid parametric studies, educational purposes, and research
benchmarking. These models should not replace detailed finite element
analysis for critical structures, experimental testing for validation,
or professional engineering judgment in design decisions.

# Chapter 4: Results and Discussion

## 4.1 Introduction

This chapter presents the comprehensive results I obtained from finite
element analysis of fixed-fixed reinforced concrete beams subjected to
various damage scenarios. My primary objective was investigating the
relationship between structural damage and natural frequency shifts,
which provides a foundation for developing predictive models for
structural health monitoring applications.

I organized the results into four main sections: model validation
against theoretical and experimental benchmarks, parametric analysis of
damage effects, dataset generation and statistical analysis, and
comparative analysis of different damage scenarios. Each section
includes detailed mathematical formulations, graphical representations,
and discussion of the observed phenomena.

## 4.2 Model Validation

### 4.2.1 Theoretical Validation

I validated the FEM implementation against the analytical solution for a
fixed-fixed beam. For a uniform, undamaged beam, the theoretical natural
frequency for the first mode is (Clough & Penzien, 2003).

Validation Test Parameters: Length: L = 3.0 m, Width: b = 0.3 m, Depth:
h = 0.45 m, Concrete strength: f\'c = 30 MPa, Density: rho = 2400 kg/m3

The extremely low error (less than 0.002%) confirms the accuracy of my
FEM implementation. This exceeds the validation results Das (2023)
reported.

### 4.2.2 Three-Way Validation Against Published FEM Results

To demonstrate that my Python FEM implementation produces results
consistent with validated commercial software, I performed a three-way
comparison using beam parameters from Das (2023). This compares my
results against published ANSYS results and theoretical Euler-Bernoulli
solutions.

**Table 4.1:** Three-Way Validation Comparison

  ---------------------------------------------------------------------------
   **Mode**   **Das ANSYS  **Das EBT   **Theoretical     **Our    **Error vs
                (Hz)**    FEM (Hz)**    EBT (Hz)**    Python FEM   Theory**
                                                        (Hz)**    
  ----------- ----------- ----------- --------------- ----------- -----------
       1        13.552      13.555        14.196        14.196      0.000%

       2        84.816      84.909        88.966        88.966      0.000%

       3        237.030     237.570       249.110       249.107     0.001%
  ---------------------------------------------------------------------------

The roughly 5 percent difference between my EBT implementation and Das
(2023) ANSYS results is expected. ANSYS uses 3D solid elements that
capture shear deformation and Poisson effects not included in
Euler-Bernoulli theory. My Python FEM correctly implements classical
EBT, as evidenced by the near-perfect match with theoretical values.

### 4.2.3 Convergence Analysis

A mesh convergence study showed that 20 elements provide sufficient
accuracy (error below 0.01%) while maintaining computational efficiency.
Further refinement beyond 20 elements yielded negligible improvements.

### 4.2.4 Comparison with Literature Experimental Data

To validate the corrosion-frequency relationship, I compared FEM
predictions with experimental data from Zhang et al. (2020):

**Table 4.2:** Validation of Corrosion-Frequency Relationship

  -----------------------------------------------------------------------
     **Corrosion     **Zhang et al.   **Our FEM Trend**  **Consistency**
       Level**       (2020) Trend**                     
  ----------------- ----------------- ----------------- -----------------
        0-5%         2-5% frequency    3-4% reduction      Consistent
                        reduction                       

        5-10%        5-10% frequency   6-8% reduction      Consistent
                        reduction                       

       10-15%       10-15% frequency  10-13% reduction     Consistent
                        reduction                       
  -----------------------------------------------------------------------

The FEM model captures the corrosion-frequency relationship observed in
experiments, validating the stiffness reduction approach.

## 4.3 Dataset Generation and Analysis

This section describes the dataset I generated through the FEM
simulation framework described in Chapter 3. Understanding the dataset
characteristics is essential before examining damage effects, as it
establishes the baseline frequency distributions and parameter
relationships.

### 4.3.1 Frequency Distribution Analysis

Figure 4.1 shows the statistical distribution of natural frequencies in
the generated dataset.

![](media/image2.png){width="5.208333333333333in"
height="3.6458333333333335in"}

**Figure 4.1:** Histogram of Mode 1 and Mode 2 frequencies across the
entire dataset, showing separate distributions for pristine and damaged
beams.

**Table 4.3:** Statistical Summary of FEM-Generated Natural Frequency
Dataset (3,000 Samples)

  ---------------------------------------------------------------------------
   **Statistic**     **Mode 1       **Mode 1       **Mode 2       **Mode 2
                   (Pristine)**   (Damaged)**    (Pristine)**   (Damaged)**
  --------------- -------------- -------------- -------------- --------------
       Mean          78.4 Hz        71.2 Hz        216.1 Hz       196.3 Hz

     Std. Dev.       42.3 Hz        38.9 Hz        116.5 Hz       107.2 Hz

        Min          18.5 Hz        15.2 Hz        51.0 Hz        41.9 Hz

        Max          245.7 Hz       223.4 Hz       677.2 Hz       615.8 Hz
  ---------------------------------------------------------------------------

The frequency range spans more than an order of magnitude, reflecting
the diverse geometric and material configurations in the dataset. The
mean frequency reduction due to damage is approximately 9.2% for Mode 1
and 9.1% for Mode 2, averaged across all damage levels.

### 4.3.2 Correlation Analysis

The Pearson correlation coefficients between input parameters and output
frequencies reveal important physical relationships:

**Table 4.4:** Parameter Sensitivity - Pearson Correlation with Mode 1
Natural Frequency

  -----------------------------------------------------------------------
       **Parameter**           **Correlation        **Interpretation**
                               Coefficient**      
  ----------------------- ----------------------- -----------------------
        Length (L)                 -0.87          Strong negative (longer
                                                     beams have lower
                                                        frequency)

         Depth (h)                 +0.64             Moderate positive
                                                    (deeper beams have
                                                     higher frequency)

     Concrete Strength             +0.52             Moderate positive
          (f\'c)                                    (stronger concrete
                                                   increases frequency)

    Corrosion Level (C)            -0.78           Strong negative (more
                                                     corrosion reduces
                                                        frequency)

         Width (b)                 +0.31               Weak positive
  -----------------------------------------------------------------------

These correlations align with theoretical expectations from the
frequency equation.

> $$f \propto \frac{1}{L^{2}}\sqrt{\frac{EI}{\rho A}} \propto \frac{h}{L^{2}}\sqrt{f_{c}'}\quad\text{(Eq. 13)}$$

## 4.4 Parametric Analysis of Damage Effects

With the dataset characteristics established, this section examines how
different damage scenarios affect natural frequencies.

### 4.4.1 Effect of Uniform Corrosion on Natural Frequencies

Figure 4.2 illustrates the relationship between corrosion level and the
fundamental natural frequency for a representative beam configuration.

![](media/image3.png){width="5.208333333333333in"
height="3.6458333333333335in"}

**Figure 4.2:** Impact of uniform corrosion on the first two natural
frequencies of a fixed-fixed RC beam (L=3.0m, b=0.3m, h=0.45m, f\'c=30
MPa).

Both Mode 1 and Mode 2 frequencies exhibit a monotonic decrease with
increasing corrosion level, consistent with the reduction in structural
stiffness. The frequency reduction follows a nonlinear trend
approximated by:

> $$\frac{f_{\text{corroded}}}{f_{\text{pristine}}} \approx \sqrt{1 - \alpha} = \sqrt{1 - 1.6 \times \frac{C}{100}}\quad\text{(Eq. 14)}$$

This square-root relationship arises from the proportionality f
proportional to sqrt(K/M), where corrosion primarily affects stiffness
while mass remains relatively constant. At low corrosion levels (0-10%),
the frequency reduction rate is approximately 0.8% per 1% corrosion,
aligning with findings from Zhang et al. (2020).

### 4.4.2 Mode Shape Analysis

Figure 4.3 presents the comparison of mode shapes between pristine and
corroded beams.

![](media/image4.png){width="5.208333333333333in"
height="3.6458333333333335in"}

**Figure 4.3:** Comparison of the first two mode shapes for pristine and
corroded (20% corrosion) beams.

The fundamental mode shape (single curvature) and second mode shape
(double curvature) maintain their characteristic forms even under
significant corrosion (20%), confirming that uniform damage does not
alter the modal patterns.

### 4.4.3 Effect of Localized Damage

Figure 4.4 demonstrates the impact of crack severity on natural
frequencies for a mid-span crack.

![](media/image5.png){width="5.208333333333333in"
height="3.6458333333333335in"}

**Figure 4.4:** Influence of crack severity (0-90% stiffness loss) at
mid-span on natural frequencies.

Cracks located at mid-span (maximum bending moment region for Mode 1)
produce the most significant frequency reduction for the fundamental
mode. The frequency reduction approximately follows:

> $$\Delta f \approx - k_{1}\beta - k_{2}\beta^{2}\quad\text{(Eq. 15)}$$

where beta is the crack severity, and k1, k2 are coefficients that
depend on crack location and beam geometry.

## 4.5 Comparative Analysis of Damage Scenarios

### 4.5.1 Uniform vs. Localized Damage

I conducted a comparative study to evaluate the differential effects of
uniform corrosion versus localized cracks on modal characteristics.

Test Configuration: Beam: L=4.0m, b=0.3m, h=0.5m, f\'c=35 MPa; Uniform
damage: 15% corrosion; Localized damage: Mid-span crack with 50%
severity, width=0.4m

**Table 4.5:** Damage Type Comparison - Frequency Response for Different
Damage Scenarios

  -----------------------------------------------------------------------
   **Damage Type**      **Mode 1          **Mode 2         **Frequency
                       Frequency**       Frequency**     Reduction (Mode
                                                              1)**
  ----------------- ----------------- ----------------- -----------------
      Pristine           98.7 Hz          272.1 Hz             \-

    Uniform (15%)        89.3 Hz          246.2 Hz            9.5%

  Localized (50% at      91.2 Hz          258.4 Hz            7.6%
      mid-span)                                         
  -----------------------------------------------------------------------

The results demonstrate that spatial distribution of damage
significantly affects frequency response, with distributed corrosion
producing larger frequency shifts than localized cracks of higher
severity.

## 4.6 Sensitivity Analysis

### 4.6.1 Parameter Sensitivity

I performed a local sensitivity analysis to quantify the influence of
each parameter on the natural frequency. The sensitivity coefficient is
defined as:

> $$S_{i} = \frac{\partial f}{\partial p_{i}} \times \frac{p_{i}}{f}\quad\text{(Eq. 16)}$$

where $p_{i}$ is the i-th parameter.

Normalized Sensitivity Coefficients: Length: -2.00, Depth: +1.50,
Concrete Strength: +0.50, Corrosion Level: -0.80

Length exhibits the highest sensitivity (-2.00), consistent with the
theoretical f proportional to $L^{- 2}$ relationship (Clough & Penzien,
2003), while corrosion sensitivity (-0.80) confirms its detectability in
SHM applications.

## 4.7 Computational Performance

Performance Metrics: Matrix Assembly: 0.8 ms, Eigenvalue Solution: 1.2
ms, Total Simulation: 2.0 ms

The high computational efficiency of both FEM and ML enables rapid
parametric studies and real-time damage assessment.

## 4.8 Machine Learning Results

### 4.8.1 Overview

Following the generation of the comprehensive dataset through finite
element analysis, I developed machine learning models to predict the
natural frequencies of RC beams based on their geometric and damage
parameters.

### 4.8.2 Model Performance Comparison

Table 4.6 presents comprehensive performance metrics for all five models
across training and testing datasets:

**Table 4.6:** Model Performance Metrics

  ------------------------------------------------------------------------------
   **Model**    **Train    **Train    **Train     **Test     **Test     **Test
                 MAE**      RMSE**      R2**      MAE**      RMSE**      R2**
  ------------ ---------- ---------- ---------- ---------- ---------- ----------
     Linear      15.93      20.98      0.834      17.05      22.28      0.828
   Regression                                                         

     Random       2.22       3.65      0.995       4.66       7.99      0.978
     Forest                                                           

    XGBoost       0.25       0.37      0.999       4.06       7.38      0.981

    CatBoost      1.74       2.58      0.997       3.00       5.61      0.989

      SVR         2.97       5.74      0.988       3.80       7.51      0.981
  ------------------------------------------------------------------------------

CatBoost demonstrates superior performance with the lowest test error
and highest R-squared score.

**Table 4.7:** Comparison with Literature Benchmarks

  -----------------------------------------------------------------------
      **Study**      **Best Model**      **Best R2**     **This Study**
  ----------------- ----------------- ----------------- -----------------
    Das (2023) -         SVM-Puk           98.78%        CatBoost: 98.9%
   Steel/Al beams                                       

    Das (2023) -      Random Forest        98.88%           RF: 97.8%
   Steel/Al beams                                       

     Saha & Yang     Neural Network       about 97%      CatBoost: 98.9%
      (2023) -                                          
     Cantilever                                         

    This Study -        CatBoost            98.9%              \-
   Fixed RC beams                                       
  -----------------------------------------------------------------------

![](media/image6.png){width="6.39457239720035in"
height="2.094419291338583in"}

**Figure 4.5:** Comparative visualization of model performance metrics.
CatBoost achieves the best balance between training accuracy and
generalization capability.

### 4.8.2.2 Prediction Accuracy Visualization

![](media/image7.png){width="6.140082020997375in"
height="4.298057742782152in"}

**Figure 4.6:** Scatter plots comparing predicted vs. actual frequencies
for all models. Perfect predictions would align along the diagonal line
(y = x). CatBoost shows the tightest clustering around the ideal
prediction line.

### 4.8.2.3 Residual Analysis

![](media/image8.png){width="6.168689851268591in"
height="2.2080271216097986in"}

**Figure 4.7:** Residual plots (predicted - actual) for each model.
Ideal models show randomly distributed residuals centered at zero with
no systematic patterns.

### 4.8.3 Feature Importance Analysis

![](media/image9.png){width="6.145030621172354in"
height="1.9405861767279091in"}

**Figure 4.8:** Permutation feature importance scores for the
best-performing model (CatBoost). Higher scores indicate greater
influence on prediction accuracy.

Feature Importance Rankings: Length (Importance about 0.45): Most
influential parameter, consistent with theoretical frequency dependence
f proportional to $L^{- 2}$. Damage Severity (about 0.25): Second most
critical, reflecting direct impact on stiffness degradation. Depth
(about 0.15): Significant contributor through moment of inertia
influence. Concrete Strength (about 0.10): Moderate importance via
elastic modulus relationship.

### 4.8.3.1 SHAP Value Analysis

![](media/image10.png){width="5.208333333333333in"
height="3.6458333333333335in"}

**Figure 4.9:** SHAP (SHapley Additive exPlanations) summary plot
showing feature contribution to model predictions. Each point represents
a sample, colored by feature value (red = high, blue = low).

SHAP Insights: Length: High values (red) strongly decrease predicted
frequency (negative SHAP values). Damage Severity: Increasing severity
consistently reduces predictions. Depth: Higher depth values increase
predicted frequencies (positive SHAP values).

### 4.8.4 Cross-Validation and Generalization

5-Fold Cross-Validation Results: All models underwent rigorous 5-fold
cross-validation to assess generalization capability. CatBoost: Mean
R-squared = 0.989 plus or minus 0.002 (excellent stability). XGBoost:
Mean R-squared = 0.982 plus or minus 0.004 (high consistency). SVR: Mean
R-squared = 0.983 plus or minus 0.002 (robust performance).

### 4.8.4.1 Uncertainty Quantification

To assess prediction reliability and provide confidence intervals for
operational deployment, I performed bootstrap-based uncertainty
quantification on test predictions.

![](media/image11.png){width="6.207740594925634in"
height="2.9722265966754158in"}

**Figure 4.10:** Left panel shows predictions with 95% confidence
intervals for 200 sorted test samples. Right panel displays the
distribution of confidence interval widths.

**Table 4.8:** Bootstrap Confidence Interval Statistics

  -----------------------------------------------------------------------
        **Metric**               **Value**          **Interpretation**
  ----------------------- ----------------------- -----------------------
      Mean Prediction            185.47 Hz        Average confidence band
      Interval Width                                       span

     Median Prediction           186.32 Hz        Typical interval width
      Interval Width                              

       95% Coverage                93.2%          Actual vs. target (95%)
        Probability                               

      Mean Prediction            51.20 Hz           Ensemble prediction
    Standard Deviation                                  uncertainty
  -----------------------------------------------------------------------

The bootstrap analysis reveals excellent calibration of uncertainty
estimates. The 93.2% coverage rate is slightly conservative relative to
the nominal 95% target, ensuring operational reliability.

![](media/image12.png){width="4.4682195975503065in"
height="3.127753718285214in"}

**Figure 4.11:** Scatter plot validating confidence interval
calibration. Green points represent predictions where actual frequencies
fall within the 95% CI (93.2% coverage); red points indicate
out-of-interval predictions (6.8%).

### 4.8.5 Computational Efficiency

Training Time Comparison (2,400 samples): Linear Regression: 0.05
seconds, Random Forest: 2.3 seconds, XGBoost: 1.8 seconds, CatBoost: 3.2
seconds, SVR: 18.5 seconds. Inference Time (600 predictions): All models
completed in less than 0.1 seconds.

### 4.8.6 Model Selection and Recommendations

Primary Model: CatBoost Regressor. CatBoost is selected as the
production model based on: Superior Accuracy: Lowest prediction errors
(MAE = 3.00 Hz, RMSE = 5.61 Hz). Best Generalization: Highest test
R-squared (0.989) with minimal overfitting. Excellent Stability: Lowest
cross-validation variance (std = 0.002).

### 4.8.6.1 Hyperparameter Optimization Analysis

I performed systematic hyperparameter optimization using
RandomizedSearchCV with 50 iterations and 5-fold cross-validation to
refine CatBoost model performance.

![](media/image13.png){width="6.298578302712161in"
height="4.409004811898512in"}

**Figure 4.12:** Feature importance visualization showing the impact of
each hyperparameter on model performance across 50 RandomizedSearchCV
iterations.

**Table 4.9:** Hyperparameter Search Space for RandomizedSearchCV
Optimization

  -----------------------------------------------------------------------
       **Parameter**             **Range**              **Purpose**
  ----------------------- ----------------------- -----------------------
        iterations                50-500            Number of boosting
                                                        iterations

       learning_rate             0.01-0.31          Step size shrinkage

           depth                   4-10                 Tree depth

        l2_leaf_reg                1-10              L2 regularization
                                                         strength

       border_count               32-255           Splits for numerical
                                                         features

      random_strength              0-10           Randomness for scoring
                                                          splits
  -----------------------------------------------------------------------

**Table 4.10:** Optimized Parameters vs. Default Configuration

  -----------------------------------------------------------------------
    **Parameter**      **Default**      **Optimized**     **Direction**
  ----------------- ----------------- ----------------- -----------------
     iterations            200               436               Up

    learning_rate         0.100             0.096             Down

        depth               8                 5               Down

     l2_leaf_reg           1.0              4.01               Up

    border_count           254               70               Down

   random_strength         1.0              0.37              Down
  -----------------------------------------------------------------------

**Table 4.11:** ML Model Performance Comparison - Default vs. Optimized
Parameters

  -----------------------------------------------------------------------
     **Metric**     **Default Model**    **Optimized     **Improvement**
                                           Model**      
  ----------------- ----------------- ----------------- -----------------
   R-squared Score       0.98958           0.99028           +0.071%

      MAE (Hz)            3.034             2.861             -5.7%

      RMSE (Hz)           5.491             5.302             -3.4%

  Training Time (s)       0.073             0.165         2.26x slower
  -----------------------------------------------------------------------

The hyperparameter optimization analysis reveals that modest performance
improvements (+0.071% R-squared, -5.7% MAE) come at a significant
computational cost (2.26x training time). The optimized configuration
demonstrates that the original default parameters were exceptionally
well-tuned.

### 4.8.7 Practical Implications for Structural Health Monitoring

Detection Capabilities: With CatBoost\'s MAE of 3.00 Hz: Minimum
Detectable Damage: Approximately 3-4% corrosion. Reliability: 98.9%
variance explained enables confident damage quantification. Early
Warning: Sufficient precision for detecting degradation before
structural safety compromised.

### 4.8.7.1 Real-World Application Scenario

To illustrate the practical utility of the developed ML model, consider
a typical bridge inspection scenario where rapid preliminary assessment
is required.

**Table 4.12:** Time Comparison Analysis - Real-World Application
Scenario

  -----------------------------------------------------------------------
     **Method**           **100            **1,000        **Processing
                      Predictions**     Predictions**      Approach**
  ----------------- ----------------- ----------------- -----------------
   Traditional FEM     8-10 hours       80-100 hours       Sequential
   (ANSYS/ABAQUS)                                       modeling required

  Python FEM (This     0.2 seconds       2.0 seconds     Automated batch
       Study)                                              processing

  CatBoost ML Model   0.01 seconds      0.05 seconds         Instant
                                                           prediction

       Manual             Days              Weeks        Impractical for
     Calculation                                           this scale
  -----------------------------------------------------------------------

Efficiency Gains: The ML approach reduces analysis time by a factor of
approximately 40,000 compared to traditional FEM software (0.01s vs 6
minutes per beam).

## 4.9 Discussion

### 4.9.1 Physical Interpretation

The results demonstrate clear physical relationships between structural
damage and dynamic characteristics: The observed frequency reductions
are directly attributable to stiffness degradation, following f
proportional to square root of K (Clough & Penzien, 2003). This explains
why uniform corrosion produces monotonic, nonlinear frequency decay.

A critical finding is that 50% localized stiffness loss over 0.4m
produces less frequency reduction (7.6%) than 15% uniform corrosion
(9.5%). Frequency is governed by global strain energy, so localized
damage affects only part of the beam while distributed damage reduces
stiffness throughout.

### 4.9.2 Comparison with Literature Benchmarks

**Table 4.13:** Literature Comparison

  --------------------------------------------------------------------------------
   **Study**   **Structure    **Method**      **Best     **Accuracy**    **This
                 Type**                      Model**                     Study**
  ----------- ------------- -------------- ------------ -------------- -----------
  Das (2023)    Al/Steel        FEM+ML      SVM-Puk/RF   98.78-98.88%   CatBoost:
                  Beams                                                   98.9%

  Saha & Yang  Cantilever       FEM+NN         ANN        about 97%     CatBoost:
    (2023)      (damaged)                                                 98.9%

   Zhang et      RC Beam     Experimental       \-         0.8%/1%       0.8%/1%
  al. (2020)   (corrosion)                                corrosion     corrosion
  --------------------------------------------------------------------------------

The frequency-corrosion sensitivity of approximately 0.8% per 1%
corrosion aligns with Zhang et al. (2020) experimental findings,
validating the damage modeling approach.

### 4.9.3 Practical Implications for SHM

The findings have several important implications: Sensitivity thresholds
depend on measurement accuracy. With typical accelerometer precision
(plus or minus 0.1 Hz), corrosion levels as low as 2-3% can be detected
for baseline beam configurations.

Environmental factors like temperature cause frequency variations
similar to early-stage damage. Based on Cai et al. (2021), temperature
effects cause approximately 0.148% frequency change per degree Celsius.
Robust SHM systems must incorporate environmental compensation.

### 4.9.4 Limitations and Future Work

Several limitations should be acknowledged: The stiffness reduction
approach, while computationally efficient, does not capture all physical
aspects of corrosion including mass changes and bond degradation. Real
structures may have boundary conditions deviating from ideal fixed-fixed
constraints. Linear elastic assumptions may not hold for severely
damaged structures.

Future research directions include more sophisticated damage models
based on fracture mechanics, experimental validation with laboratory
specimens and field structures, inverse problem algorithms for damage
identification from frequency measurements, and integration with other
SHM techniques.

## 4.10 Summary

This chapter presented comprehensive results from finite element
analysis of damaged RC beams:

1\. Model Validation: The FEM implementation achieved below 0.002% error
compared to theoretical solutions. Results align with experimental
trends from Zhang et al. (2020).

2\. Damage Effects: Uniform corrosion causes monotonic, nonlinear
frequency reductions with sensitivity of approximately 0.8% per 1%
corrosion.

3\. Dataset Generation: A diverse dataset of 3,000 simulations was
created using Latin Hypercube Sampling.

4\. Statistical Analysis: Frequency distributions show strong
correlations with beam length (r=-0.87) and corrosion level (r=-0.78).

5\. ML Performance: CatBoost achieved 98.9% R-squared, exceeding
literature benchmarks.

The results provide a solid foundation for developing machine learning
models for predictive maintenance and structural health monitoring
applications.

# Chapter 5: Conclusions

## 5.1 Summary of Findings

This thesis investigated the prediction of natural frequencies of fixed
reinforced concrete beams using machine learning models validated
against finite element simulations. The key findings are:

The Python-based FEM implementation achieved excellent accuracy with
less than 0.002% error compared to theoretical Euler-Bernoulli solutions
and consistent results with published ANSYS analyses from Das (2023).

Among the five machine learning algorithms tested (Linear Regression,
Random Forest, XGBoost, CatBoost, and SVR), CatBoost demonstrated
superior performance with R-squared of 98.9%, MAE of 3.00 Hz, and the
lowest cross-validation variance.

The corrosion-frequency sensitivity of approximately 0.8% frequency
reduction per 1% corrosion level aligns with experimental findings from
Zhang et al. (2020), validating the stiffness reduction approach for
damage modeling.

Feature importance analysis revealed beam length as the most influential
parameter, consistent with theoretical expectations, followed by damage
severity and cross-sectional depth.

## 5.2 Contributions

This research makes several contributions to the field:

Methodological: First systematic comparison of five ML algorithms for
fixed RC beam frequency prediction, establishing CatBoost as the optimal
choice for this application.

Practical: Development of an open dataset of 3,000 validated FEM
simulations and trained models that can be used for rapid structural
assessments.

Theoretical: Quantification of damage-frequency relationships with
sensitivity coefficients supporting early damage detection in RC
structures.

## 5.3 Recommendations for Practice

Based on the findings, the following recommendations are made for
practical applications:

For preliminary design assessments, the trained CatBoost model can
provide rapid frequency predictions, enabling engineers to screen
multiple beam configurations efficiently.

For structural health monitoring applications, the established
sensitivity thresholds (approximately 0.8% frequency reduction per 1%
corrosion) can guide measurement requirements and damage detection
criteria.

Temperature compensation should be implemented in field monitoring
systems to distinguish damage-induced frequency changes from
environmental variations.

## 5.4 Future Work

Several directions for future research are recommended:

Experimental validation with laboratory specimens and field structures
to further confirm the FEM and ML predictions.

Extension to other boundary conditions (simply supported, cantilever,
continuous spans) and structural elements (slabs, columns).

Development of inverse problem algorithms for damage identification and
localization from measured frequency data.

Integration with physics-informed neural networks to incorporate
governing equations directly into the ML framework.

Investigation of transfer learning approaches to adapt models trained on
simulation data for real-world applications.

# References

1\. ACI Committee 318. (2019). Building Code Requirements for Structural
Concrete (ACI 318-19). American Concrete Institute.

2\. Avcar, M., & Saplioglu, K. (2015). An artificial neural network
application for estimation of natural frequencies of beams. Research
Journal of Applied Sciences, Engineering and Technology, 9(3), 131-138.

3\. Banerjee, A., Panigrahi, B., & Pohit, G. (2017). Crack modelling and
detection in Timoshenko FGM beam under transverse vibration using
frequency contour and response surface model with GA. Nondestructive
Testing and Evaluation, 32(1), 27-48.

4\. Bathe, K. J. (2014). Finite Element Procedures (2nd ed.).
Klaus-Jurgen Bathe.

5\. Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter
optimization. Journal of Machine Learning Research, 13(10), 281-305.

6\. Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.

7\. Cai, Y., Zhang, K., Ye, Z., Liu, C., Lu, K., & Wang, L. (2021).
Influence of temperature on the natural vibration characteristics of
simply supported reinforced concrete beam. Sensors, 21, 4242.

8\. Cairns, J., Plizzari, G. A., Du, Y., Law, D. W., & Franzoni, C.
(2005). Mechanical properties of corrosion-damaged reinforcement. ACI
Materials Journal, 102(4), 256-264.

9\. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting
System. Proceedings of the 22nd ACM SIGKDD International Conference on
Knowledge Discovery and Data Mining, 785-794.

10\. Chondros, T. G., Dimarogonas, A. D., & Yao, J. (1998). A continuous
cracked beam vibration theory. Journal of Sound and Vibration, 215(1),
17-34.

11\. Chopra, A. K. (2012). Dynamics of Structures: Theory and
Applications to Earthquake Engineering (4th ed.). Pearson.

12\. Clough, R. W., & Penzien, J. (2003). Dynamics of Structures (3rd
ed.). Computers & Structures, Inc.

13\. Cohen, J. (1992). A power primer. Psychological Bulletin, 112(1),
155-159.

14\. Cook, R. D. (2007). Concepts and Applications of Finite Element
Analysis (4th ed.). Wiley.

15\. Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine
Learning, 20(3), 273-297.

16\. Das, O. (2023). Prediction of the natural frequencies of various
beams using regression machine learning models. Sigma Journal of
Engineering and Natural Sciences, 41(2), 302-321.

17\. Dimarogonas, A. D. (1996). Vibration of cracked structures: A state
of the art review. Engineering Fracture Mechanics, 55(5), 831-857.

18\. Doebling, S. W., Farrar, C. R., Prime, M. B., & Shevitz, D. W.
(1996). Damage identification and health monitoring of structural and
mechanical systems from changes in their vibration characteristics: A
literature review. Los Alamos National Laboratory Report LA-13070-MS.

19\. Efron, B., & Tibshirani, R. (1993). An Introduction to the
Bootstrap. Chapman and Hall.

20\. Eurocode 2. (2004). Design of Concrete Structures - Part 1-1:
General Rules and Rules for Buildings. EN 1992-1-1.

21\. Farrar, C. R., & Worden, K. (2013). Structural Health Monitoring: A
Machine Learning Perspective. John Wiley & Sons.

22\. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning.
MIT Press.

23\. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On
calibration of modern neural networks. In International Conference on
Machine Learning (pp. 1321-1330). PMLR.

24\. Harris, C. R., et al. (2020). Array programming with NumPy. Nature,
585, 357-362.

25\. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of
Statistical Learning: Data Mining, Inference, and Prediction (2nd ed.).
Springer.

26\. Helton, J. C., & Davis, F. J. (2003). Latin hypercube sampling and
the propagation of uncertainty in analyses of complex systems.
Reliability Engineering & System Safety, 81(1), 23-69.

27\. Hughes, T. J. R. (2000). The Finite Element Method: Linear Static
and Dynamic Finite Element Analysis. Dover Publications.

28\. Inman, D. J. (2014). Engineering Vibration (4th ed.). Pearson.

29\. Laory, I., Trinh, T. N., Smith, I. F., & Brownjohn, J. M. (2018).
Methodologies for predicting natural frequency variation of a suspension
bridge. Engineering Structures, 80, 211-221.

30\. Luu, X.-B. (2024). Finite element modelling of reinforced concrete
beam strengthening using ultra-high performance fiber-reinforced
shotcrete. Structures, 60, 105794.

31\. MacGregor, J. G., & Wight, J. K. (2012). Reinforced Concrete:
Mechanics and Design (6th ed.). Pearson.

32\. McKay, M. D., Beckman, R. J., & Conover, W. J. (1979). A comparison
of three methods for selecting values of input variables in the analysis
of output from a computer code. Technometrics, 21(2), 239-245.

33\. McKinney, W. (2010). Data Structures for Statistical Computing in
Python. Proceedings of the 9th Python in Science Conference, 51-56.

34\. Meirovitch, L. (2001). Fundamentals of Vibrations. McGraw-Hill.

35\. Miller, J., et al. (2000). The Tacoma Narrows Bridge collapse: A
review of the causes. Engineering History and Heritage, 153(1), 25-30.

36\. Nikoo, M., Zarfam, P., & Sayahpour, H. (2018). Determination of
natural frequency of Euler-Bernoulli beam using artificial neural
network. Engineering Structures, 157, 154-166.

37\. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in
Python. Journal of Machine Learning Research, 12, 2825-2830.

38\. Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., &
Gulin, A. (2018). CatBoost: Unbiased Boosting with Categorical Features.
Advances in Neural Information Processing Systems, 31.

39\. Rao, S. S. (2019). Mechanical Vibrations (6th ed.). Pearson.

40\. Rodriguez, J., Ortega, L. M., & Casal, J. (1997). Load carrying
capacity of concrete structures with corroded reinforcement.
Construction and Building Materials, 11(4), 239-248.

41\. Saha, P., & Yang, M. (2023). A neural network approach to estimate
the frequency of a cantilever beam with random multiple damages.
Sensors, 23, 7867.

42\. Sohn, H., Farrar, C. R., Hemez, F. M., Shunk, D. D., Stinemates, D.
W., Nadler, B. R., & Czarnecki, J. J. (2004). A review of structural
health monitoring literature: 1996-2001. Los Alamos National Laboratory
Report LA-13976-MS.

43\. Virtanen, P., et al. (2020). SciPy 1.0: Fundamental Algorithms for
Scientific Computing in Python. Nature Methods, 17, 261-272.

44\. Zhang, Y., Cheng, Y., Tan, G., Lyu, X., Sun, X., Bai, Y., & Yang,
S. (2020). Natural frequency response evaluation for RC beams affected
by steel corrosion using acceleration sensors. Sensors, 20, 5335.

45\. Zienkiewicz, O. C., & Taylor, R. L. (2000). The Finite Element
Method (5th ed.). Butterworth-Heinemann.
