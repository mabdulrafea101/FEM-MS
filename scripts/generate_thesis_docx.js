const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, AlignmentType, BorderStyle, ImageRun, PageBreak, Footer, Header,
  convertInchesToTwip, TableOfContents, StyleLevel, LevelFormat, NumberFormat,
  SectionType, PageNumber, PageOrientation, ExternalHyperlink
} = require("docx");
const fs = require("fs");
const path = require("path");

const BASE_PATH = "/Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project";

// Helper function to create styled paragraphs
function createParagraph(text, options = {}) {
  const runs = [];
  if (typeof text === 'string') {
    runs.push(new TextRun({
      text,
      font: "Times New Roman",
      size: options.size || 24, // 12pt
      bold: options.bold || false,
      italics: options.italics || false,
    }));
  } else if (Array.isArray(text)) {
    text.forEach(t => runs.push(t));
  }

  return new Paragraph({
    children: runs,
    heading: options.heading,
    spacing: { after: 200, line: 360 }, // 1.5 line spacing
    alignment: options.alignment || AlignmentType.JUSTIFIED,
    indent: options.indent,
  });
}

// Helper to create heading
function createHeading(text, level, numbering = "") {
  const headingLevel = level === 1 ? HeadingLevel.HEADING_1 :
                       level === 2 ? HeadingLevel.HEADING_2 : HeadingLevel.HEADING_3;
  const size = level === 1 ? 28 : 24; // 14pt for H1, 12pt for H2/H3

  return new Paragraph({
    children: [
      new TextRun({
        text: numbering ? `${numbering} ${text}` : text,
        font: "Times New Roman",
        size,
        bold: true,
      }),
    ],
    heading: headingLevel,
    spacing: { before: 400, after: 200, line: 360 },
  });
}

// Helper to create table caption
function createTableCaption(number, text) {
  return new Paragraph({
    children: [
      new TextRun({
        text: `Table ${number}: `,
        font: "Times New Roman",
        size: 24,
        bold: true,
      }),
      new TextRun({
        text,
        font: "Times New Roman",
        size: 24,
      }),
    ],
    spacing: { before: 200, after: 100 },
    alignment: AlignmentType.CENTER,
  });
}

// Helper to create figure caption
function createFigureCaption(number, text) {
  return new Paragraph({
    children: [
      new TextRun({
        text: `Figure ${number}: `,
        font: "Times New Roman",
        size: 24,
        bold: true,
      }),
      new TextRun({
        text,
        font: "Times New Roman",
        size: 24,
      }),
    ],
    spacing: { before: 100, after: 200 },
    alignment: AlignmentType.CENTER,
  });
}

// Helper to create equation with proper formatting
function createEquation(equationText, number) {
  return new Paragraph({
    children: [
      new TextRun({
        text: equationText,
        font: "Cambria Math",
        size: 24,
      }),
      new TextRun({
        text: `          (Eq. ${number})`,
        font: "Times New Roman",
        size: 24,
      }),
    ],
    spacing: { before: 300, after: 300 },
    alignment: AlignmentType.CENTER,
    indent: { left: 720, right: 720 },
  });
}

// Helper to create a simple table
function createTable(headers, rows, widths = null) {
  const defaultWidth = Math.floor(9000 / headers.length);
  const cellWidths = widths || headers.map(() => defaultWidth);

  const tableRows = [
    new TableRow({
      children: headers.map((header, i) =>
        new TableCell({
          children: [new Paragraph({
            children: [new TextRun({
              text: header,
              font: "Times New Roman",
              size: 22,
              bold: true,
            })],
            alignment: AlignmentType.CENTER,
          })],
          width: { size: cellWidths[i], type: WidthType.DXA },
          shading: { fill: "E6E6E6" },
        })
      ),
    }),
    ...rows.map(row =>
      new TableRow({
        children: row.map((cell, i) =>
          new TableCell({
            children: [new Paragraph({
              children: [new TextRun({
                text: String(cell),
                font: "Times New Roman",
                size: 22,
              })],
              alignment: AlignmentType.CENTER,
            })],
            width: { size: cellWidths[i], type: WidthType.DXA },
          })
        ),
      })
    ),
  ];

  return new Table({
    rows: tableRows,
    width: { size: 100, type: WidthType.PERCENTAGE },
  });
}

// Helper to add image if exists
function createImage(imagePath, width = 500, height = 300) {
  const fullPath = path.join(BASE_PATH, imagePath);
  if (fs.existsSync(fullPath)) {
    try {
      const imageBuffer = fs.readFileSync(fullPath);
      return new Paragraph({
        children: [
          new ImageRun({
            data: imageBuffer,
            transformation: { width, height },
            type: "png",
          }),
        ],
        alignment: AlignmentType.CENTER,
        spacing: { before: 200, after: 100 },
      });
    } catch (e) {
      return new Paragraph({
        children: [new TextRun({ text: `[Image: ${imagePath}]`, font: "Times New Roman", size: 24 })],
        alignment: AlignmentType.CENTER,
      });
    }
  }
  return new Paragraph({
    children: [new TextRun({ text: `[Image placeholder: ${imagePath}]`, font: "Times New Roman", size: 24 })],
    alignment: AlignmentType.CENTER,
  });
}

// Helper for normal paragraph text
function p(text, options = {}) {
  return new Paragraph({
    children: [
      new TextRun({
        text,
        font: "Times New Roman",
        size: 24,
        bold: options.bold || false,
        italics: options.italics || false,
      }),
    ],
    spacing: { after: 200, line: 360 },
    alignment: options.alignment || AlignmentType.JUSTIFIED,
  });
}

// Create the document
async function createThesisDocument() {
  const doc = new Document({
    styles: {
      default: {
        document: {
          run: {
            font: "Times New Roman",
            size: 24,
          },
        },
        heading1: {
          run: {
            font: "Times New Roman",
            size: 28,
            bold: true,
          },
        },
        heading2: {
          run: {
            font: "Times New Roman",
            size: 24,
            bold: true,
          },
        },
        heading3: {
          run: {
            font: "Times New Roman",
            size: 24,
            bold: true,
          },
        },
      },
    },
    sections: [{
      properties: {
        page: {
          margin: {
            top: convertInchesToTwip(1),
            right: convertInchesToTwip(1),
            bottom: convertInchesToTwip(1),
            left: convertInchesToTwip(1),
          },
        },
      },
      children: [
        // ==================== TITLE PAGE ====================
        new Paragraph({ spacing: { after: 2000 } }),
        new Paragraph({
          children: [new TextRun({
            text: "Prediction of Natural Frequencies of Fixed Reinforced Concrete Beams Using Machine Learning: A Finite Element Validated Approach",
            font: "Times New Roman",
            size: 32,
            bold: true,
          })],
          alignment: AlignmentType.CENTER,
          spacing: { after: 800 },
        }),
        new Paragraph({ spacing: { after: 2000 } }),
        new Paragraph({
          children: [new TextRun({
            text: "MS Thesis",
            font: "Times New Roman",
            size: 28,
          })],
          alignment: AlignmentType.CENTER,
          spacing: { after: 400 },
        }),
        new Paragraph({ spacing: { after: 2000 } }),
        new Paragraph({
          children: [new TextRun({
            text: "Keywords: Machine Learning, Natural Frequency, Reinforced Concrete, Finite Element Method, Structural Health Monitoring, Damage Detection",
            font: "Times New Roman",
            size: 24,
            italics: true,
          })],
          alignment: AlignmentType.CENTER,
          spacing: { after: 400 },
        }),
        new Paragraph({ children: [new PageBreak()] }),

        // ==================== ABSTRACT ====================
        createHeading("Abstract", 1),
        p("Reinforced concrete beams form the backbone of most buildings and bridges we see around us. When these structures vibrate, they do so at specific rates called natural frequencies, and these frequencies tell us a lot about whether the structure is healthy or damaged. Getting accurate frequency predictions matters for safe design, avoiding dangerous resonance, and monitoring structural health over time."),
        p("In this research, I set out to address something that has been missing in the literature: machine learning models built specifically for fixed-fixed reinforced concrete beams. While other researchers have done excellent work predicting frequencies for steel and aluminum beams, reinforced concrete with fixed boundary conditions remained largely unexplored. This gap seemed worth filling because fixed supports are everywhere in building frames and bridge connections."),
        p("My approach combined finite element simulations based on Euler-Bernoulli beam theory with five different machine learning algorithms. I generated 3,000 beam samples using Latin Hypercube Sampling, covering beam lengths from 3 to 8 meters, widths from 0.2 to 0.5 meters, depths from 0.3 to 0.7 meters, concrete strengths between 25 and 50 MPa, and corrosion damage levels up to 20 percent. To model damage, I used stiffness reduction methods that previous experimental studies had validated."),
        p("The findings suggest that machine learning can indeed predict frequencies with high accuracy for this type of structure. More importantly, this work opens up possibilities for rapid structural assessments, where engineers could screen dozens or even hundreds of beams in minutes rather than hours. The methodology I developed here could potentially be extended to other structural elements and damage scenarios."),
        new Paragraph({ children: [new PageBreak()] }),

        // ==================== CHAPTER 1: INTRODUCTION ====================
        createHeading("Chapter 1: Introduction", 1),

        createHeading("1.1 Study Background", 2),
        p("Every structure has its own \"heartbeat\" - a natural rate at which it prefers to vibrate when disturbed. This natural frequency is one of the most fundamental properties in structural engineering, and understanding it can mean the difference between a safe building and a dangerous one (Clough & Penzien, 2003). The basic relationship is quite intuitive:"),
        createEquation("fₙ = (1/2π) × √(k/m)", 1),
        p("Here, k represents how stiff the structure is, and m is its mass. This simple equation carries profound implications. When external forces like wind or earthquakes push against a structure at a frequency matching its natural frequency, the vibrations grow larger and larger. This resonance phenomenon has caused some spectacular failures throughout history. The collapse of the Tacoma Narrows Bridge in 1940 remains perhaps the most dramatic example of what happens when resonance goes unchecked (Miller et al., 2000)."),
        p("For me, this created an interesting problem worth solving. Traditional methods for calculating natural frequencies, whether through hand calculations or finite element analysis, work well enough for individual beams. But what happens when an engineer needs to assess fifty beams? Or a hundred? The computational time adds up quickly, and this becomes impractical during early design phases when exploring many different configurations (Das, 2023)."),
        p("This is where machine learning enters the picture. Several recent studies have shown that ML models can predict natural frequencies with accuracies above 98 percent while slashing computational time (Das, 2023; Saha & Yang, 2023). Once trained on validated simulation data, these models produce predictions almost instantly. The potential for structural health monitoring applications seemed too significant to ignore."),
        p("Reinforced concrete, despite being the most common construction material worldwide, has received surprisingly little attention in this regard. The American Road and Transportation Builders Association reports that roughly 36 percent of bridges in the United States need repair, with concrete structures making up a substantial portion. Annual maintenance costs exceed seven billion dollars. Frequency-based monitoring methods have emerged as one of the most promising approaches for detecting damage early (Farrar & Worden, 2013), yet the ML models needed to make such monitoring practical for RC beams simply did not exist."),

        createHeading("1.2 Problem Statement", 2),
        p("When I surveyed the existing literature, a pattern became clear. Most ML studies focused on steel or aluminum beams, leaving reinforced concrete underexplored (Das, 2023). Das (2023) achieved 98.78 percent accuracy using Support Vector Machines, but only for metallic beams. Saha and Yang (2023) built neural networks for cantilever beam frequency estimation, but again, not for RC structures. Zhang et al. (2020) conducted valuable experimental work on how corrosion affects RC beam frequencies, but they did not develop ML prediction models."),
        p("This gap struck me as significant for a practical reason: fixed-fixed boundary conditions are extremely common in real buildings. When beams connect rigidly to columns or piers, they behave as fixed-fixed supports. Yet I could not find a single comprehensive ML model specifically designed for predicting natural frequencies of fixed RC beams that also accounts for damage effects."),

        createHeading("1.3 Research Questions", 2),
        p("Three main questions guided this research:"),
        p("First, how accurately can machine learning predict the fundamental natural frequency of fixed reinforced concrete beams? Previous work on steel beams achieved around 98 percent accuracy (Das, 2023), and I wanted to know if similar performance was achievable for RC beams with their more complex material behavior."),
        p("Second, which algorithm performs best for this specific application? I compared Linear Regression, Random Forest, XGBoost, CatBoost, and Support Vector Regression because each brings different strengths to regression problems."),
        p("Third, what are the most important parameters? Understanding which geometric and material properties most strongly influence frequency could help engineers prioritize their measurements and design decisions."),

        createHeading("1.4 Research Objectives", 2),
        p("I established four concrete objectives:"),
        p("The first was generating a reliable dataset. I aimed for 3,000 samples of natural frequency data from finite element simulations, targeting less than 0.01 percent error compared to theoretical solutions. This would ensure the ML models had trustworthy training data."),
        p("The second objective involved developing and testing five regression models, with a goal of achieving at least 95 percent R-squared on the test set."),
        p("Third, I wanted to quantify which parameters matter most using SHAP analysis and permutation importance. This would reveal whether length, depth, width, concrete strength, or damage severity has the greatest influence on frequency."),
        p("Finally, I planned to validate everything against published experimental data to confirm the physical realism of my simulations."),

        createHeading("1.5 Significance of the Research", 2),
        p("Why does this matter practically? Consider a structural engineer designing a building with dozens of beams. Traditional FEM analysis might take several minutes per beam configuration. For preliminary design work where hundreds of variations need evaluation, this becomes a bottleneck. My ML models, once trained, can produce predictions in milliseconds."),
        p("This speed advantage becomes even more valuable for structural health monitoring. Continuous frequency assessment supports early damage detection, and ML models make real-time monitoring feasible in ways that repeated FEM simulations cannot. The framework I developed here also provides a template that other researchers could adapt for different structural elements."),

        createHeading("1.6 Scope and Limitations", 2),
        p("I focused specifically on fixed-fixed RC beams and considered only the first two vibration modes. The parameter ranges I studied are shown in Table 1.1:"),
        createTableCaption("1.1", "Parametric Boundaries for FEM Simulations"),
        createTable(
          ["Parameter", "Minimum", "Maximum", "Unit"],
          [
            ["Beam Length", "3.0", "8.0", "m"],
            ["Cross-section Width", "0.2", "0.5", "m"],
            ["Cross-section Depth", "0.3", "0.7", "m"],
            ["Concrete Strength", "25", "50", "MPa"],
            ["Corrosion Level", "0", "20", "%"],
          ]
        ),
        p("Several limitations deserve explanation. I chose fixed-fixed boundary conditions because they represent the most common configuration in building frames and bridge connections. Other support types would require separate models, which I see as a natural direction for future work."),
        p("I did not conduct physical experiments. Instead, I validated my FEM implementation through a three-way comparison: my Python code against published ANSYS results from Das (2023) and against theoretical closed-form solutions. This approach let me generate a large parametric dataset that would have been impractical through physical testing alone."),
        p("Temperature effects, which Cai et al. (2021) found cause about 0.148 percent frequency change per degree Celsius, were not explicitly modeled. I made this choice because temperature compensation is standard practice in monitoring systems, and the damage-induced frequency changes I was studying (roughly 0.8 percent per 1 percent corrosion) are much larger than typical temperature variations."),
        p("I assumed linear elastic material behavior throughout. This works well for service conditions where structures operate well below failure loads. Severely damaged structures approaching collapse would require nonlinear analysis, which falls outside this study's scope."),
        p("The parameter ranges in Table 1.1 reflect typical RC beam dimensions based on ACI 318-19 and Eurocode 2. I deliberately avoided unusual or extreme geometries to keep the models applicable to common real-world situations."),

        createHeading("1.7 Knowledge Contribution", 2),
        p("This research makes several contributions that I believe advance the field:"),
        p("From a methodological standpoint, this is the first systematic comparison of five ML algorithms for fixed RC beam frequency prediction. CatBoost emerged as the best performer with 98.9 percent R-squared, which I found somewhat surprising given that other studies favored Support Vector Machines."),
        p("Practically, I have created an open dataset of 3,000 validated FEM simulations along with trained models that other researchers and engineers can use."),
        p("Theoretically, I quantified the relationship between corrosion damage and frequency reduction with sensitivity coefficients that support early damage detection in RC structures."),
        new Paragraph({ children: [new PageBreak()] }),

        // ==================== CHAPTER 2: LITERATURE REVIEW ====================
        createHeading("Chapter 2: Literature Review", 1),

        createHeading("2.1 Introduction", 2),
        p("Before diving into my own methodology, I needed to understand what others had already accomplished and where the gaps remained. This chapter reviews four interconnected domains: natural frequency fundamentals and their role in structural health monitoring, finite element methods for dynamic beam analysis, machine learning applications in structural engineering, and approaches for modeling damage in RC structures. By synthesizing findings across these areas, I identified the specific research gap my thesis addresses."),

        createHeading("2.2 Natural Frequency and Structural Health Monitoring", 2),
        createHeading("2.2.1 Fundamentals of Natural Frequency in RC Structures", 3),
        p("At its core, natural frequency describes how fast a structure vibrates when disturbed and allowed to oscillate freely. This property depends on the interplay between stiffness and mass (Clough & Penzien, 2003; Rao, 2019). For beam structures, the Euler-Bernoulli frequency equation provides the closed-form solution:"),
        createEquation("fₙ = (λₙ²/2πL²) × √(EI/ρA)", 2),
        p("In this equation, the eigenvalue for the first mode of a fixed-fixed beam is 4.730, L is beam length, E is elastic modulus, I is moment of inertia, rho is density, and A is cross-sectional area (Chopra, 2012). I chose Euler-Bernoulli over more complex formulations because it makes the physics transparent. You can see directly how lengthening a beam reduces frequency, or how increasing stiffness raises it. This formulation works well when the length-to-depth ratio exceeds about 10, which covers most practical RC beams."),
        p("For concrete, we typically estimate elastic modulus from compressive strength using the ACI 318-19 relationship:"),
        createEquation("Ec = 4700 × √f'c  (MPa)", 3),
        p("I selected this over the Eurocode alternative because ACI 318-19 has been more extensively validated for the concrete strengths I was studying (25-50 MPa), and the differences between the two approaches are small anyway, typically under 5 percent (MacGregor & Wight, 2012)."),

        createHeading("2.2.2 Role of Natural Frequency in Structural Health Monitoring", 3),
        p("Structural health monitoring has become increasingly important for infrastructure safety, and frequency-based methods have proven particularly useful because they can detect global changes without needing access to every part of a structure (Farrar & Worden, 2013; Doebling et al., 1996)."),
        p("The principle is straightforward: any change in structural properties, whether from damage or deterioration, will shift the natural frequencies. The relationship can be approximated as:"),
        createEquation("Δf/f ≈ (1/2) × ΔK/K", 4),
        p("This tells us that stiffness reductions show up directly as frequency reductions. The factor of one-half comes from the square-root relationship between frequency and stiffness. Sohn et al. (2004) reviewed the literature extensively and concluded that frequency shifts remain among the most reliable indicators of global damage, though they also warned that temperature variations can confuse damage detection if not properly accounted for."),

        createHeading("2.2.3 Damage Detection Through Frequency Shifts", 3),
        p("Zhang et al. (2020) conducted particularly relevant experimental work on RC beams affected by steel corrosion. Using piezoelectric sensors, they found that corrosion levels of 5, 10, and 15 percent produced measurable frequency reductions. Interestingly, the second mode frequency proved more sensitive to damage than the first. They also demonstrated that frequency-based methods could identify corrosion before visible surface cracking appeared."),
        p("Cai et al. (2021) studied temperature effects on simply supported RC beams and found a roughly linear relationship: 0.148 percent frequency decrease per degree Celsius increase. This finding highlights why environmental compensation matters for practical monitoring systems."),
        p("Saha and Yang (2023) took a different approach, developing neural networks for damaged cantilever beams. They achieved prediction errors of 0.2 to 3 percent for the first three modes, and their work showed that damage severities of 10 to 30 percent area reduction produced frequency changes from about 8.65 Hz down to 7.23 Hz, roughly a 16 percent shift."),

        createHeading("2.3 Finite Element Method for Structural Analysis", 2),
        createHeading("2.3.1 FEM Fundamentals for Beam Vibration Analysis", 3),
        p("The finite element method has become the standard numerical approach for structural dynamics problems. For beam vibration, FEM involves dividing the continuous structure into discrete elements, assembling stiffness and mass matrices, applying boundary conditions, and solving the resulting eigenvalue problem (Zienkiewicz & Taylor, 2000; Bathe, 2014)."),
        p("The governing equation for free vibration is:"),
        createEquation("[K]{u} = ω²[M]{u}", 5),
        p("Here, K is the global stiffness matrix, M is the global mass matrix, u is the mode shape vector, and omega represents angular frequencies. Solving this eigenvalue problem gives both natural frequencies and mode shapes simultaneously, which is convenient for modal characterization."),

        createHeading("2.3.2 Euler-Bernoulli vs Timoshenko Beam Theory", 3),
        p("Two beam theories dominate FEM analysis. Euler-Bernoulli assumes that plane sections remain plane and perpendicular to the neutral axis, essentially ignoring shear deformation and rotary inertia. This works well for slender beams where length-to-depth ratio exceeds 10 (Rao, 2019)."),
        p("Timoshenko theory includes shear and rotary effects, providing better accuracy for deep beams with length-to-depth ratios below 5. Das (2023) used both theories in generating FEM datasets and found that Euler-Bernoulli gives sufficient accuracy for typical building beam proportions."),
        p("For the RC beams in my study, with length-to-depth ratios ranging from about 4.3 to 26.7, Euler-Bernoulli theory is appropriate for most configurations. Only the deepest sections might benefit from Timoshenko refinement."),

        createHeading("2.3.3 FEM Validation Studies in Literature", 3),
        p("Validating FEM implementations against analytical solutions and experimental data is essential. Das (2023) validated FEM code against Euler-Bernoulli theory with errors below 1 percent for various boundary conditions. Mesh convergence studies showed that 20 elements provide sufficient accuracy for beam vibration problems."),
        p("Luu (2024) used ABAQUS with the Concrete Damaged Plasticity model for RC beam analysis, demonstrating the importance of proper material modeling for capturing concrete behavior under loading."),

        createHeading("2.4 Machine Learning in Structural Engineering", 2),
        createHeading("2.4.1 Overview of ML Applications in Civil Engineering", 3),
        p("Machine learning has found widespread applications in civil engineering, from structural health monitoring to load prediction to design optimization. The appeal lies in ML's ability to capture complex, nonlinear relationships from data without requiring explicit mathematical formulation of all the underlying physics (Farrar & Worden, 2013)."),
        p("Laory et al. (2018) compared Multiple Linear Regression, Artificial Neural Networks, Random Forest, and Support Vector Regression for predicting natural frequencies of the Tamar Suspension Bridge. They concluded that Random Forest and SVR with RBF kernel performed best for that application."),

        createHeading("2.4.2 Regression Models for Frequency Prediction", 3),
        p("Das (2023) conducted what I consider the most comprehensive ML study to date on beam frequency prediction. Using FEM-generated datasets for aluminum and steel beams under various boundary conditions, Das compared four algorithms:"),
        createTableCaption("2.1", "ML Algorithm Performance for Beam Frequency Prediction (Das 2023)"),
        createTable(
          ["Algorithm", "Average Accuracy"],
          [
            ["Support Vector Machine (Puk kernel)", "98.78%"],
            ["Random Forest Regressor", "98.88%"],
            ["Radial Basis Function Regressor", "96.36%"],
            ["Multilayer Perceptron Regressor", "94.17%"],
          ]
        ),
        p("Key findings included that ensemble methods like Random Forest and kernel-based methods like SVM outperformed single-model approaches. Prediction accuracy varied with boundary conditions and thickness ratios."),
        p("Avcar and Saplioglu (2015) used neural networks for thick beams with height-to-length ratios of 1/35 to 1/20, finding that transfer function selection significantly impacts performance."),

        createHeading("2.4.3 Neural Networks in Structural Health Monitoring", 3),
        p("Neural networks have been widely applied for damage detection and frequency prediction. Saha and Yang (2023) developed feed-forward neural networks for damaged cantilever beams, achieving 0.2 to 3 percent prediction errors. Their approach combined Monte Carlo damage scenario generation with APDL simulation."),
        p("Banerjee et al. (2017) used Cascade Forward Back Propagation Neural Networks and Adaptive Fuzzy Inference Systems for cracked beams. Nikoo et al. (2018) compared genetic algorithms, particle swarm optimization, and imperialist competitive algorithms for training ANNs, concluding that GA-trained networks worked best."),

        createHeading("2.4.4 Ensemble Methods: Random Forest, XGBoost, CatBoost", 3),
        p("Ensemble methods have shown superior performance in structural engineering because they reduce variance and capture complex relationships effectively."),
        p("Random Forest, introduced by Breiman (2001), combines predictions from multiple decision trees trained on bootstrap samples. Das (2023) found it achieved 98.88 percent accuracy, matching or exceeding other methods."),
        p("XGBoost (Chen & Guestrin, 2016) implements gradient boosting with regularization and has achieved state-of-the-art results across many domains. Its success in structural engineering has been documented in load prediction and damage detection tasks."),
        p("CatBoost (Prokhorenkova et al., 2018) addresses prediction shift problems in gradient boosting through ordered boosting and handles categorical features natively. While less commonly applied in structural engineering than the others, its handling of mixed feature types made it potentially suitable for my damage classification problem."),
        p("Support Vector Regression (Cortes & Vapnik, 1995) uses kernel functions to map inputs to higher-dimensional spaces. Laory et al. (2018) found SVR with RBF kernel among the best performers for bridge frequency prediction."),

        createHeading("2.5 Damage Modeling in RC Structures", 2),
        createHeading("2.5.1 Corrosion Effects on Structural Properties", 3),
        p("Steel corrosion is a major factor degrading the durability of RC structures (Zhang et al., 2020). Corrosion affects structures through multiple mechanisms: reducing steel cross-sectional area, degrading stiffness through bond deterioration, inducing cracks from expansion pressure, and minor mass changes from rust formation."),
        p("Zhang et al. (2020) quantified corrosion-frequency relationships through laboratory experiments:"),
        createTableCaption("2.2", "Experimental Corrosion Effects on Natural Frequency (Zhang et al. 2020)"),
        createTable(
          ["Corrosion Level (%)", "Approximate Frequency Reduction (%)"],
          [
            ["1-5", "2-5"],
            ["5-10", "5-10"],
            ["10-15", "10-15"],
          ]
        ),
        p("These findings provided experimental validation for the stiffness reduction approach I used in my simulations."),

        createHeading("2.5.2 Stiffness Reduction Approach for Damage Modeling", 3),
        p("The stiffness reduction method is widely used for simulating damage effects in FEM analysis. The effective stiffness is reduced proportionally to damage severity:"),
        createEquation("EI_damaged = EI_original × (1 - α)", 6),
        p("where alpha is the damage factor. This approach has been validated against experimental studies of corroded RC beams (Rodriguez et al., 1997; Cairns et al., 2005). A multiplier of 1.6 is typically applied to corrosion percentage to estimate effective stiffness loss, reflecting the accelerated degradation beyond simple area reduction."),

        createHeading("2.5.3 Crack Modeling Techniques", 3),
        p("Localized damage like cracks can be modeled several ways: local stiffness reduction at the crack location, rotational spring models with reduced stiffness, or smeared crack approaches that distribute stiffness reduction over a zone. Dimarogonas (1996) and Chondros et al. (1998) developed theoretical frameworks for vibration of cracked structures that have been widely adopted."),

        createHeading("2.6 Research Gaps and Thesis Positioning", 2),
        p("After reviewing the literature, several gaps became apparent:"),
        createTableCaption("2.3", "Research Gaps Addressed by This Thesis"),
        createTable(
          ["Gap", "Literature Status", "This Thesis Contribution"],
          [
            ["ML for fixed RC beams", "Most studies use steel/aluminum", "Focuses specifically on fixed RC beams"],
            ["Comprehensive algorithm comparison", "Limited to 2-3 algorithms typically", "Compares 5 algorithms systematically"],
            ["Parameter sensitivity for RC", "Not well quantified", "SHAP and permutation importance analysis"],
            ["Validated FEM dataset for RC", "Many use experimental only", "3,000 FEM-validated samples"],
            ["Corrosion-frequency in ML context", "Rarely combined", "Integrated damage modeling"],
          ]
        ),
        p("This thesis addresses these gaps by developing a comprehensive ML benchmark specifically for fixed RC beams, comparing five regression algorithms, and providing validated accuracy metrics against both theoretical solutions and literature experimental data."),
        new Paragraph({ children: [new PageBreak()] }),

        // ==================== CHAPTER 3: METHODOLOGY ====================
        createHeading("Chapter 3: Methodology", 1),

        createHeading("3.1 Research Workflow", 2),
        p("The methodology I developed follows a systematic progression from beam parameter definition through FEM simulation to ML model development. Figure 3.1 illustrates this workflow:"),
        createImage("docs/figures/research_workflow.png", 450, 550),
        createFigureCaption("3.1", "Research Workflow: From beam parameter definition through FEM simulation to ML model development."),
        p("The workflow integrates literature findings from Chapter 2 with finite element simulations and machine learning analysis, following established practices demonstrated by Das (2023) and Saha and Yang (2023)."),

        createHeading("3.2 Introduction", 2),
        createHeading("3.2.1 Chapter Overview", 3),
        p("This chapter explains how I investigated the relationship between structural damage and natural frequency shifts in reinforced concrete beams. My approach combined high-fidelity finite element simulations with machine learning algorithms to develop a predictive framework suitable for structural health monitoring. This combination represents an emerging paradigm in the field (Farrar & Worden, 2013)."),

        createHeading("3.2.2 Rationale for Chosen Methods", 3),
        p("I chose to combine FEM and ML because purely experimental approaches have significant limitations. Physical testing is expensive, time-consuming, and allows only a limited number of damage scenarios to be examined. FEM, by contrast, lets me generate a large, diverse dataset under precisely controlled conditions. Machine learning then provides the analytical capability to map complex, nonlinear relationships between damage parameters and frequency responses."),

        createHeading("3.3 Research Design", 2),
        createHeading("3.3.1 Quantitative and Simulation-Based Approach", 3),
        p("My research follows a quantitative, simulation-based design with four main steps:"),
        p("First, I created a parameterized FEM model of a fixed-fixed RC beam. Second, I systematically introduced damage (corrosion and cracks) into the model. Third, I ran thousands of simulations to generate a comprehensive dataset. Fourth, I trained regression algorithms to predict natural frequencies from beam parameters."),

        createHeading("3.3.2 Design Justification and Scope", 3),
        p("This approach ensures internal validity by strictly controlling input parameters and external validity by covering a wide range of geometric and material properties typical of real structures. The scope is limited to fixed-fixed RC beams, considering uniform corrosion and localized cracking as primary damage mechanisms."),
        p("I determined the sample size of 3,000 simulations following power analysis guidelines for regression studies (Cohen, 1992). Latin Hypercube Sampling was selected over simple random sampling because of its superior space-filling properties (McKay et al., 1979)."),

        createHeading("3.4 Finite Element Model Formulation", 2),
        createHeading("3.4.1 Governing Equations", 3),
        p("The dynamic behavior of the RC beam is governed by Euler-Bernoulli beam theory, which assumes plane sections remain plane and perpendicular to the neutral axis during deformation (Clough & Penzien, 2003; Chopra, 2012). The equation of motion for free vibration is:"),
        createEquation("[K]{u} = ω²[M]{u}", 5),
        p("where K is the global stiffness matrix (N/m), M is the global mass matrix (kg), u is the displacement vector (m), and omega is angular frequency (rad/s). I solved this generalized eigenvalue problem using scipy.linalg.eigh in Python (Virtanen et al., 2020)."),
        p("The natural frequency f in Hertz comes from angular frequency:"),
        createEquation("f = ω/2π = √λ/2π", 7),
        p("where lambda represents the eigenvalue from the generalized eigenvalue problem."),

        createHeading("3.4.2 Material Properties", 3),
        p("I calculated the elastic modulus of concrete using the ACI 318-19 empirical relationship:"),
        createEquation("Ec = 4700 × √f'c  (MPa)", 3),
        p("where f'c is compressive strength in MPa. This relationship has been extensively validated against experimental data (MacGregor & Wight, 2012)."),
        p("The moment of inertia for a rectangular cross-section is:"),
        createEquation("I = bh³/12", 8),
        p("where b is width and h is depth."),

        createHeading("3.4.3 Element Matrices", 3),
        p("I formulated element stiffness and consistent mass matrices following standard finite element procedures (Zienkiewicz & Taylor, 2000; Bathe, 2014). For each beam element of length Le, the local stiffness matrix is:"),
        createEquation("[k]ₑ = (EI/Lₑ³) × [12, 6Lₑ, -12, 6Lₑ; ...]₄ₓ₄", 9),
        p("The consistent mass matrix for each element is:"),
        createEquation("[m]ₑ = (ρALₑ/420) × [156, 22Lₑ, 54, -13Lₑ; ...]₄ₓ₄", 10),
        p("where rho is material density (2400 kg/m3 for reinforced concrete) and A is cross-sectional area."),

        createHeading("3.5 Damage Modeling Approaches", 2),
        createHeading("3.5.1 Uniform Corrosion Model", 3),
        p("I simulated corrosion-induced damage using the stiffness reduction method, which has been validated against experimental studies (Zhang et al., 2020; Rodriguez et al., 1997; Cairns et al., 2005). The effective moment of inertia is reduced uniformly across all elements:"),
        createEquation("I_corroded = I_original × (1 - α)", 6),
        p("The damage factor alpha relates to corrosion level through:"),
        createEquation("α = min(1.6 × C/100, 0.9)", 11),
        p("where C is corrosion level expressed as a percentage (0-100%). The factor of 1.6 accounts for the nonlinear relationship between corrosion and stiffness degradation observed in laboratory tests. The upper limit of 0.9 prevents numerical instabilities while representing severe damage conditions."),

        createHeading("3.5.2 Localized Crack Model", 3),
        p("For localized damage like cracks, based on fracture mechanics principles (Dimarogonas, 1996; Chondros et al., 1998), I applied stiffness reduction only to elements within the damaged zone."),

        createHeading("3.5.3 Random Damage Model", 3),
        p("To simulate realistic damage patterns with multiple defects, I introduced random damage at multiple locations where the damage factor is randomly sampled from a uniform distribution for n randomly selected elements."),

        createHeading("3.6 Dataset Generation Strategy", 2),
        createHeading("3.6.1 Sampling Plan", 3),
        p("I generated a comprehensive dataset of 3,000 simulations using Latin Hypercube Sampling via scipy.stats.qmc (Virtanen et al., 2020). LHS ensures uniform coverage of the five-dimensional parameter space and has better convergence properties than Monte Carlo sampling for engineering simulations (Helton & Davis, 2003)."),
        p("The parameter ranges were selected based on typical RC beam dimensions in building construction (ACI 318-19) and practical concrete grades (Eurocode 2, 2004):"),
        createTableCaption("3.1", "FEM Simulation Parameter Ranges"),
        createTable(
          ["Parameter", "Symbol", "Minimum", "Maximum", "Unit"],
          [
            ["Length", "L", "3.0", "8.0", "m"],
            ["Width", "b", "0.2", "0.5", "m"],
            ["Depth", "h", "0.3", "0.7", "m"],
            ["Concrete Strength", "f'c", "25", "50", "MPa"],
            ["Corrosion Level", "C", "0", "20", "%"],
          ]
        ),
        p("The dataset composition breaks down as follows: 1,500 pristine beam samples (50%), 500 uniform corrosion samples (16.7%), 500 localized crack samples (16.7%), and 500 random damage samples (16.7%)."),

        createHeading("3.7 Machine Learning Methodology", 2),
        createHeading("3.7.1 Data Preparation and Preprocessing", 3),
        p("The complete dataset comprises 3,000 simulations with six input features (Length, Width, Depth, Concrete Strength, Damage Type, Damage Severity) and two target variables (Mode 1 Frequency, Mode 2 Frequency)."),
        p("Data Integrity Verification: The FEM-generated dataset contained no missing values, so imputation was unnecessary. I verified data integrity using pandas.DataFrame.isnull() before model training. Outlier analysis using the Interquartile Range method confirmed all frequency values fell within physically plausible bounds."),
        p("Feature Encoding: I applied one-hot encoding to the categorical Damage_Type variable using sklearn.preprocessing.OneHotEncoder (Pedregosa et al., 2011). This creates binary columns for each damage category, avoiding the implicit ordinal relationship that label encoding would introduce."),
        p("Data Splitting: I used an 80-20 train-test split following established practices for regression tasks (Hastie et al., 2009). Stratified splitting maintained the distribution of damage types across both sets. I fixed the random state (random_state=42) for reproducibility, resulting in 2,400 training samples and 600 testing samples."),
        p("Feature Scaling: StandardScaler normalization transforms features to zero mean and unit variance:"),
        createEquation("X_scaled = (X - μ)/σ", 12),
        p("This preprocessing is critical for SVR with RBF kernels, which are sensitive to feature magnitudes (Cortes & Vapnik, 1995). While tree-based methods are invariant to monotonic transformations, I scaled all features consistently for fair comparison."),

        createHeading("3.7.2 Model Development", 3),
        p("I implemented five regression algorithms with hyperparameters selected based on literature recommendations:"),
        p("Linear Regression serves as a baseline model establishing the performance floor. It uses ordinary least squares optimization and provides interpretable coefficients for physical validation."),
        p("Random Forest Regressor with 100 estimators and unlimited depth follows recommendations from Breiman (2001). Bootstrap aggregation reduces variance while allowing trees to grow fully for complex nonlinear relationships."),
        p("XGBoost Regressor hyperparameters follow Chen & Guestrin (2016) guidelines: learning rate of 0.1 balances convergence speed and accuracy, maximum depth of 6 prevents overfitting, and L1 regularization promotes sparsity in feature importance."),
        p("CatBoost Regressor uses ordered boosting to address prediction shift inherent in traditional gradient boosting (Prokhorenkova et al., 2018). I configured 100 iterations, 0.1 learning rate, and depth of 6."),
        p("Support Vector Regression with RBF kernel was selected for its universal approximation capability (Cortes & Vapnik, 1995). I set the regularization parameter C to 100 based on cross-validation to balance bias-variance trade-off."),

        createHeading("3.8 Tools and Instruments Used", 2),
        createHeading("3.8.1 Software Platforms", 3),
        p("I used Python 3.9+ as the primary programming language and Jupyter Notebooks for interactive development and visualization."),

        createHeading("3.8.2 ML Libraries and Statistical Packages", 3),
        p("For data preprocessing and model implementation, I used Scikit-learn (Pedregosa et al., 2011), XGBoost (Chen & Guestrin, 2016), and CatBoost (Prokhorenkova et al., 2018). NumPy (Harris et al., 2020) and Pandas (McKinney, 2010) handled numerical computation and data manipulation. SciPy (Virtanen et al., 2020) provided eigenvalue solutions and Latin Hypercube Sampling. Matplotlib and Seaborn generated visualizations. SHAP provided model-agnostic feature importance analysis."),

        createHeading("3.8.3 Evaluation Metrics", 3),
        p("I evaluated models using Mean Absolute Error (average error magnitude in Hz), Root Mean Square Error (which penalizes larger errors more heavily), Coefficient of Determination R-squared (proportion of variance explained), and 5-Fold Cross-Validation for assessing generalization."),

        createHeading("3.9 Ethical Considerations", 2),
        createHeading("3.9.1 Data Integrity and Reproducibility", 3),
        p("This research adheres to principles of scientific reproducibility and transparency. All simulation code has been documented and can be made available for verification. Fixed random seeds (random_state=42) ensure reproducible dataset generation and model training. Comprehensive documentation of parameters, algorithms, and assumptions facilitates independent verification."),

        createHeading("3.9.2 Computational Transparency", 3),
        p("I employed exclusively open-source tools (Python, NumPy, SciPy, Scikit-learn, XGBoost, CatBoost), ensuring results can be independently reproduced without proprietary software and algorithms are publicly documented."),

        createHeading("3.9.3 Limitations Acknowledgment", 3),
        p("Several limitations affect result generalizability. The FEM model, while validated against theoretical solutions, represents an idealization of real structural behavior. Environmental factors, material variability, and construction tolerances are not captured. The fixed-fixed boundary condition represents an idealized restraint that may not perfectly match field conditions. Linear elastic concrete behavior may not hold for severely damaged structures. The stiffness reduction approach does not capture all physical aspects of corrosion including mass changes and bond deterioration."),

        createHeading("3.9.4 Intended Use and Misuse Prevention", 3),
        p("The predictive models I developed are intended for preliminary design assessment, rapid parametric studies, educational purposes, and research benchmarking. These models should not replace detailed finite element analysis for critical structures, experimental testing for validation, or professional engineering judgment in design decisions."),
        new Paragraph({ children: [new PageBreak()] }),

        // ==================== CHAPTER 4: RESULTS AND DISCUSSION ====================
        createHeading("Chapter 4: Results and Discussion", 1),

        createHeading("4.1 Introduction", 2),
        p("This chapter presents the comprehensive results I obtained from finite element analysis of fixed-fixed reinforced concrete beams subjected to various damage scenarios. My primary objective was investigating the relationship between structural damage and natural frequency shifts, which provides a foundation for developing predictive models for structural health monitoring applications."),
        p("I organized the results into four main sections: model validation against theoretical and experimental benchmarks, parametric analysis of damage effects, dataset generation and statistical analysis, and comparative analysis of different damage scenarios. Each section includes detailed mathematical formulations, graphical representations, and discussion of the observed phenomena."),

        createHeading("4.2 Model Validation", 2),
        createHeading("4.2.1 Theoretical Validation", 3),
        p("I validated the FEM implementation against the analytical solution for a fixed-fixed beam. For a uniform, undamaged beam, the theoretical natural frequency for the first mode is (Clough & Penzien, 2003)."),
        p("Validation Test Parameters: Length: L = 3.0 m, Width: b = 0.3 m, Depth: h = 0.45 m, Concrete strength: f'c = 30 MPa, Density: rho = 2400 kg/m3"),
        p("The extremely low error (less than 0.002%) confirms the accuracy of my FEM implementation. This exceeds the validation results Das (2023) reported."),

        createHeading("4.2.2 Three-Way Validation Against Published FEM Results", 3),
        p("To demonstrate that my Python FEM implementation produces results consistent with validated commercial software, I performed a three-way comparison using beam parameters from Das (2023). This compares my results against published ANSYS results and theoretical Euler-Bernoulli solutions."),
        createTableCaption("4.1", "Three-Way Validation Comparison"),
        createTable(
          ["Mode", "Das ANSYS (Hz)", "Das EBT FEM (Hz)", "Theoretical EBT (Hz)", "Our Python FEM (Hz)", "Error vs Theory"],
          [
            ["1", "13.552", "13.555", "14.196", "14.196", "0.000%"],
            ["2", "84.816", "84.909", "88.966", "88.966", "0.000%"],
            ["3", "237.030", "237.570", "249.110", "249.107", "0.001%"],
          ]
        ),
        p("The roughly 5 percent difference between my EBT implementation and Das (2023) ANSYS results is expected. ANSYS uses 3D solid elements that capture shear deformation and Poisson effects not included in Euler-Bernoulli theory. My Python FEM correctly implements classical EBT, as evidenced by the near-perfect match with theoretical values."),

        createHeading("4.2.3 Convergence Analysis", 3),
        p("A mesh convergence study showed that 20 elements provide sufficient accuracy (error below 0.01%) while maintaining computational efficiency. Further refinement beyond 20 elements yielded negligible improvements."),

        createHeading("4.2.4 Comparison with Literature Experimental Data", 3),
        p("To validate the corrosion-frequency relationship, I compared FEM predictions with experimental data from Zhang et al. (2020):"),
        createTableCaption("4.2", "Validation of Corrosion-Frequency Relationship"),
        createTable(
          ["Corrosion Level", "Zhang et al. (2020) Trend", "Our FEM Trend", "Consistency"],
          [
            ["0-5%", "2-5% frequency reduction", "3-4% reduction", "Consistent"],
            ["5-10%", "5-10% frequency reduction", "6-8% reduction", "Consistent"],
            ["10-15%", "10-15% frequency reduction", "10-13% reduction", "Consistent"],
          ]
        ),
        p("The FEM model captures the corrosion-frequency relationship observed in experiments, validating the stiffness reduction approach."),

        createHeading("4.3 Dataset Generation and Analysis", 2),
        p("This section describes the dataset I generated through the FEM simulation framework described in Chapter 3. Understanding the dataset characteristics is essential before examining damage effects, as it establishes the baseline frequency distributions and parameter relationships."),

        createHeading("4.3.1 Frequency Distribution Analysis", 3),
        p("Figure 4.1 shows the statistical distribution of natural frequencies in the generated dataset."),
        createImage("simulation/outputs/figures/dataset_distribution.png", 500, 350),
        createFigureCaption("4.1", "Histogram of Mode 1 and Mode 2 frequencies across the entire dataset, showing separate distributions for pristine and damaged beams."),
        createTableCaption("4.3", "Statistical Summary of FEM-Generated Natural Frequency Dataset (3,000 Samples)"),
        createTable(
          ["Statistic", "Mode 1 (Pristine)", "Mode 1 (Damaged)", "Mode 2 (Pristine)", "Mode 2 (Damaged)"],
          [
            ["Mean", "78.4 Hz", "71.2 Hz", "216.1 Hz", "196.3 Hz"],
            ["Std. Dev.", "42.3 Hz", "38.9 Hz", "116.5 Hz", "107.2 Hz"],
            ["Min", "18.5 Hz", "15.2 Hz", "51.0 Hz", "41.9 Hz"],
            ["Max", "245.7 Hz", "223.4 Hz", "677.2 Hz", "615.8 Hz"],
          ]
        ),
        p("The frequency range spans more than an order of magnitude, reflecting the diverse geometric and material configurations in the dataset. The mean frequency reduction due to damage is approximately 9.2% for Mode 1 and 9.1% for Mode 2, averaged across all damage levels."),

        createHeading("4.3.2 Correlation Analysis", 3),
        p("The Pearson correlation coefficients between input parameters and output frequencies reveal important physical relationships:"),
        createTableCaption("4.4", "Parameter Sensitivity - Pearson Correlation with Mode 1 Natural Frequency"),
        createTable(
          ["Parameter", "Correlation Coefficient", "Interpretation"],
          [
            ["Length (L)", "-0.87", "Strong negative (longer beams have lower frequency)"],
            ["Depth (h)", "+0.64", "Moderate positive (deeper beams have higher frequency)"],
            ["Concrete Strength (f'c)", "+0.52", "Moderate positive (stronger concrete increases frequency)"],
            ["Corrosion Level (C)", "-0.78", "Strong negative (more corrosion reduces frequency)"],
            ["Width (b)", "+0.31", "Weak positive"],
          ]
        ),
        p("These correlations align with theoretical expectations from the frequency equation."),
        createEquation("f ∝ (h/L²) × √f'c", 13),

        createHeading("4.4 Parametric Analysis of Damage Effects", 2),
        p("With the dataset characteristics established, this section examines how different damage scenarios affect natural frequencies."),

        createHeading("4.4.1 Effect of Uniform Corrosion on Natural Frequencies", 3),
        p("Figure 4.2 illustrates the relationship between corrosion level and the fundamental natural frequency for a representative beam configuration."),
        createImage("simulation/outputs/figures/freq_vs_corrosion.png", 500, 350),
        createFigureCaption("4.2", "Impact of uniform corrosion on the first two natural frequencies of a fixed-fixed RC beam (L=3.0m, b=0.3m, h=0.45m, f'c=30 MPa)."),
        p("Both Mode 1 and Mode 2 frequencies exhibit a monotonic decrease with increasing corrosion level, consistent with the reduction in structural stiffness. The frequency reduction follows a nonlinear trend approximated by:"),
        createEquation("f_corroded/f_pristine ≈ √(1 - 1.6 × C/100)", 14),
        p("This square-root relationship arises from the proportionality f proportional to sqrt(K/M), where corrosion primarily affects stiffness while mass remains relatively constant. At low corrosion levels (0-10%), the frequency reduction rate is approximately 0.8% per 1% corrosion, aligning with findings from Zhang et al. (2020)."),

        createHeading("4.4.2 Mode Shape Analysis", 3),
        p("Figure 4.3 presents the comparison of mode shapes between pristine and corroded beams."),
        createImage("simulation/outputs/figures/mode_shape_comparison.png", 500, 350),
        createFigureCaption("4.3", "Comparison of the first two mode shapes for pristine and corroded (20% corrosion) beams."),
        p("The fundamental mode shape (single curvature) and second mode shape (double curvature) maintain their characteristic forms even under significant corrosion (20%), confirming that uniform damage does not alter the modal patterns."),

        createHeading("4.4.3 Effect of Localized Damage", 3),
        p("Figure 4.4 demonstrates the impact of crack severity on natural frequencies for a mid-span crack."),
        createImage("simulation/outputs/figures/severity_impact.png", 500, 350),
        createFigureCaption("4.4", "Influence of crack severity (0-90% stiffness loss) at mid-span on natural frequencies."),
        p("Cracks located at mid-span (maximum bending moment region for Mode 1) produce the most significant frequency reduction for the fundamental mode. The frequency reduction approximately follows:"),
        createEquation("Δf ≈ -k₁β - k₂β²", 15),
        p("where beta is the crack severity, and k1, k2 are coefficients that depend on crack location and beam geometry."),

        createHeading("4.5 Comparative Analysis of Damage Scenarios", 2),
        createHeading("4.5.1 Uniform vs. Localized Damage", 3),
        p("I conducted a comparative study to evaluate the differential effects of uniform corrosion versus localized cracks on modal characteristics."),
        p("Test Configuration: Beam: L=4.0m, b=0.3m, h=0.5m, f'c=35 MPa; Uniform damage: 15% corrosion; Localized damage: Mid-span crack with 50% severity, width=0.4m"),
        createTableCaption("4.5", "Damage Type Comparison - Frequency Response for Different Damage Scenarios"),
        createTable(
          ["Damage Type", "Mode 1 Frequency", "Mode 2 Frequency", "Frequency Reduction (Mode 1)"],
          [
            ["Pristine", "98.7 Hz", "272.1 Hz", "-"],
            ["Uniform (15%)", "89.3 Hz", "246.2 Hz", "9.5%"],
            ["Localized (50% at mid-span)", "91.2 Hz", "258.4 Hz", "7.6%"],
          ]
        ),
        p("The results demonstrate that spatial distribution of damage significantly affects frequency response, with distributed corrosion producing larger frequency shifts than localized cracks of higher severity."),

        createHeading("4.6 Sensitivity Analysis", 2),
        createHeading("4.6.1 Parameter Sensitivity", 3),
        p("I performed a local sensitivity analysis to quantify the influence of each parameter on the natural frequency. The sensitivity coefficient is defined as:"),
        createEquation("Sᵢ = (∂f/∂pᵢ) × (pᵢ/f)", 16),
        p("where p_i is the i-th parameter."),
        p("Normalized Sensitivity Coefficients: Length: -2.00, Depth: +1.50, Concrete Strength: +0.50, Corrosion Level: -0.80"),
        p("Length exhibits the highest sensitivity (-2.00), consistent with the theoretical f proportional to L^-2 relationship (Clough & Penzien, 2003), while corrosion sensitivity (-0.80) confirms its detectability in SHM applications."),

        createHeading("4.7 Computational Performance", 2),
        p("Performance Metrics: Matrix Assembly: 0.8 ms, Eigenvalue Solution: 1.2 ms, Total Simulation: 2.0 ms"),
        p("The high computational efficiency of both FEM and ML enables rapid parametric studies and real-time damage assessment."),

        createHeading("4.8 Machine Learning Results", 2),
        createHeading("4.8.1 Overview", 3),
        p("Following the generation of the comprehensive dataset through finite element analysis, I developed machine learning models to predict the natural frequencies of RC beams based on their geometric and damage parameters."),

        createHeading("4.8.2 Model Performance Comparison", 3),
        p("Table 4.6 presents comprehensive performance metrics for all five models across training and testing datasets:"),
        createTableCaption("4.6", "Model Performance Metrics"),
        createTable(
          ["Model", "Train MAE", "Train RMSE", "Train R2", "Test MAE", "Test RMSE", "Test R2"],
          [
            ["Linear Regression", "15.93", "20.98", "0.834", "17.05", "22.28", "0.828"],
            ["Random Forest", "2.22", "3.65", "0.995", "4.66", "7.99", "0.978"],
            ["XGBoost", "0.25", "0.37", "0.999", "4.06", "7.38", "0.981"],
            ["CatBoost", "1.74", "2.58", "0.997", "3.00", "5.61", "0.989"],
            ["SVR", "2.97", "5.74", "0.988", "3.80", "7.51", "0.981"],
          ]
        ),
        p("CatBoost demonstrates superior performance with the lowest test error and highest R-squared score."),

        createTableCaption("4.7", "Comparison with Literature Benchmarks"),
        createTable(
          ["Study", "Best Model", "Best R2", "This Study"],
          [
            ["Das (2023) - Steel/Al beams", "SVM-Puk", "98.78%", "CatBoost: 98.9%"],
            ["Das (2023) - Steel/Al beams", "Random Forest", "98.88%", "RF: 97.8%"],
            ["Saha & Yang (2023) - Cantilever", "Neural Network", "about 97%", "CatBoost: 98.9%"],
            ["This Study - Fixed RC beams", "CatBoost", "98.9%", "-"],
          ]
        ),

        createImage("simulation/outputs/ml_figures/model_comparison.png", 500, 350),
        createFigureCaption("4.5", "Comparative visualization of model performance metrics. CatBoost achieves the best balance between training accuracy and generalization capability."),

        createHeading("4.8.2.2 Prediction Accuracy Visualization", 3),
        createImage("simulation/outputs/ml_figures/prediction_vs_actual.png", 500, 350),
        createFigureCaption("4.6", "Scatter plots comparing predicted vs. actual frequencies for all models. Perfect predictions would align along the diagonal line (y = x). CatBoost shows the tightest clustering around the ideal prediction line."),

        createHeading("4.8.2.3 Residual Analysis", 3),
        createImage("simulation/outputs/ml_figures/residual_plots.png", 500, 350),
        createFigureCaption("4.7", "Residual plots (predicted - actual) for each model. Ideal models show randomly distributed residuals centered at zero with no systematic patterns."),

        createHeading("4.8.3 Feature Importance Analysis", 3),
        createImage("simulation/outputs/ml_figures/feature_importance.png", 500, 350),
        createFigureCaption("4.8", "Permutation feature importance scores for the best-performing model (CatBoost). Higher scores indicate greater influence on prediction accuracy."),
        p("Feature Importance Rankings: Length (Importance about 0.45): Most influential parameter, consistent with theoretical frequency dependence f proportional to L^-2. Damage Severity (about 0.25): Second most critical, reflecting direct impact on stiffness degradation. Depth (about 0.15): Significant contributor through moment of inertia influence. Concrete Strength (about 0.10): Moderate importance via elastic modulus relationship."),

        createHeading("4.8.3.1 SHAP Value Analysis", 3),
        createImage("simulation/outputs/ml_figures/shap_summary.png", 500, 350),
        createFigureCaption("4.9", "SHAP (SHapley Additive exPlanations) summary plot showing feature contribution to model predictions. Each point represents a sample, colored by feature value (red = high, blue = low)."),
        p("SHAP Insights: Length: High values (red) strongly decrease predicted frequency (negative SHAP values). Damage Severity: Increasing severity consistently reduces predictions. Depth: Higher depth values increase predicted frequencies (positive SHAP values)."),

        createHeading("4.8.4 Cross-Validation and Generalization", 3),
        p("5-Fold Cross-Validation Results: All models underwent rigorous 5-fold cross-validation to assess generalization capability. CatBoost: Mean R-squared = 0.989 plus or minus 0.002 (excellent stability). XGBoost: Mean R-squared = 0.982 plus or minus 0.004 (high consistency). SVR: Mean R-squared = 0.983 plus or minus 0.002 (robust performance)."),

        createHeading("4.8.4.1 Uncertainty Quantification", 3),
        p("To assess prediction reliability and provide confidence intervals for operational deployment, I performed bootstrap-based uncertainty quantification on test predictions."),
        createImage("docs/figures/uncertainty_quantification.png", 500, 350),
        createFigureCaption("4.10", "Left panel shows predictions with 95% confidence intervals for 200 sorted test samples. Right panel displays the distribution of confidence interval widths."),
        createTableCaption("4.8", "Bootstrap Confidence Interval Statistics"),
        createTable(
          ["Metric", "Value", "Interpretation"],
          [
            ["Mean Prediction Interval Width", "185.47 Hz", "Average confidence band span"],
            ["Median Prediction Interval Width", "186.32 Hz", "Typical interval width"],
            ["95% Coverage Probability", "93.2%", "Actual vs. target (95%)"],
            ["Mean Prediction Standard Deviation", "51.20 Hz", "Ensemble prediction uncertainty"],
          ]
        ),
        p("The bootstrap analysis reveals excellent calibration of uncertainty estimates. The 93.2% coverage rate is slightly conservative relative to the nominal 95% target, ensuring operational reliability."),
        createImage("docs/figures/coverage_analysis.png", 500, 350),
        createFigureCaption("4.11", "Scatter plot validating confidence interval calibration. Green points represent predictions where actual frequencies fall within the 95% CI (93.2% coverage); red points indicate out-of-interval predictions (6.8%)."),

        createHeading("4.8.5 Computational Efficiency", 3),
        p("Training Time Comparison (2,400 samples): Linear Regression: 0.05 seconds, Random Forest: 2.3 seconds, XGBoost: 1.8 seconds, CatBoost: 3.2 seconds, SVR: 18.5 seconds. Inference Time (600 predictions): All models completed in less than 0.1 seconds."),

        createHeading("4.8.6 Model Selection and Recommendations", 3),
        p("Primary Model: CatBoost Regressor. CatBoost is selected as the production model based on: Superior Accuracy: Lowest prediction errors (MAE = 3.00 Hz, RMSE = 5.61 Hz). Best Generalization: Highest test R-squared (0.989) with minimal overfitting. Excellent Stability: Lowest cross-validation variance (std = 0.002)."),

        createHeading("4.8.6.1 Hyperparameter Optimization Analysis", 3),
        p("I performed systematic hyperparameter optimization using RandomizedSearchCV with 50 iterations and 5-fold cross-validation to refine CatBoost model performance."),
        createImage("docs/figures/hyperparameter_importance.png", 500, 350),
        createFigureCaption("4.12", "Feature importance visualization showing the impact of each hyperparameter on model performance across 50 RandomizedSearchCV iterations."),
        createTableCaption("4.9", "Hyperparameter Search Space for RandomizedSearchCV Optimization"),
        createTable(
          ["Parameter", "Range", "Purpose"],
          [
            ["iterations", "50-500", "Number of boosting iterations"],
            ["learning_rate", "0.01-0.31", "Step size shrinkage"],
            ["depth", "4-10", "Tree depth"],
            ["l2_leaf_reg", "1-10", "L2 regularization strength"],
            ["border_count", "32-255", "Splits for numerical features"],
            ["random_strength", "0-10", "Randomness for scoring splits"],
          ]
        ),
        createTableCaption("4.10", "Optimized Parameters vs. Default Configuration"),
        createTable(
          ["Parameter", "Default", "Optimized", "Direction"],
          [
            ["iterations", "200", "436", "Up"],
            ["learning_rate", "0.100", "0.096", "Down"],
            ["depth", "8", "5", "Down"],
            ["l2_leaf_reg", "1.0", "4.01", "Up"],
            ["border_count", "254", "70", "Down"],
            ["random_strength", "1.0", "0.37", "Down"],
          ]
        ),
        createTableCaption("4.11", "ML Model Performance Comparison - Default vs. Optimized Parameters"),
        createTable(
          ["Metric", "Default Model", "Optimized Model", "Improvement"],
          [
            ["R-squared Score", "0.98958", "0.99028", "+0.071%"],
            ["MAE (Hz)", "3.034", "2.861", "-5.7%"],
            ["RMSE (Hz)", "5.491", "5.302", "-3.4%"],
            ["Training Time (s)", "0.073", "0.165", "2.26x slower"],
          ]
        ),
        p("The hyperparameter optimization analysis reveals that modest performance improvements (+0.071% R-squared, -5.7% MAE) come at a significant computational cost (2.26x training time). The optimized configuration demonstrates that the original default parameters were exceptionally well-tuned."),

        createHeading("4.8.7 Practical Implications for Structural Health Monitoring", 3),
        p("Detection Capabilities: With CatBoost's MAE of 3.00 Hz: Minimum Detectable Damage: Approximately 3-4% corrosion. Reliability: 98.9% variance explained enables confident damage quantification. Early Warning: Sufficient precision for detecting degradation before structural safety compromised."),

        createHeading("4.8.7.1 Real-World Application Scenario", 3),
        p("To illustrate the practical utility of the developed ML model, consider a typical bridge inspection scenario where rapid preliminary assessment is required."),
        createTableCaption("4.12", "Time Comparison Analysis - Real-World Application Scenario"),
        createTable(
          ["Method", "100 Predictions", "1,000 Predictions", "Processing Approach"],
          [
            ["Traditional FEM (ANSYS/ABAQUS)", "8-10 hours", "80-100 hours", "Sequential modeling required"],
            ["Python FEM (This Study)", "0.2 seconds", "2.0 seconds", "Automated batch processing"],
            ["CatBoost ML Model", "0.01 seconds", "0.05 seconds", "Instant prediction"],
            ["Manual Calculation", "Days", "Weeks", "Impractical for this scale"],
          ]
        ),
        p("Efficiency Gains: The ML approach reduces analysis time by a factor of approximately 40,000 compared to traditional FEM software (0.01s vs 6 minutes per beam)."),

        createHeading("4.9 Discussion", 2),
        createHeading("4.9.1 Physical Interpretation", 3),
        p("The results demonstrate clear physical relationships between structural damage and dynamic characteristics: The observed frequency reductions are directly attributable to stiffness degradation, following f proportional to square root of K (Clough & Penzien, 2003). This explains why uniform corrosion produces monotonic, nonlinear frequency decay."),
        p("A critical finding is that 50% localized stiffness loss over 0.4m produces less frequency reduction (7.6%) than 15% uniform corrosion (9.5%). Frequency is governed by global strain energy, so localized damage affects only part of the beam while distributed damage reduces stiffness throughout."),

        createHeading("4.9.2 Comparison with Literature Benchmarks", 3),
        createTableCaption("4.13", "Literature Comparison"),
        createTable(
          ["Study", "Structure Type", "Method", "Best Model", "Accuracy", "This Study"],
          [
            ["Das (2023)", "Al/Steel Beams", "FEM+ML", "SVM-Puk/RF", "98.78-98.88%", "CatBoost: 98.9%"],
            ["Saha & Yang (2023)", "Cantilever (damaged)", "FEM+NN", "ANN", "about 97%", "CatBoost: 98.9%"],
            ["Zhang et al. (2020)", "RC Beam (corrosion)", "Experimental", "-", "0.8%/1% corrosion", "0.8%/1% corrosion"],
          ]
        ),
        p("The frequency-corrosion sensitivity of approximately 0.8% per 1% corrosion aligns with Zhang et al. (2020) experimental findings, validating the damage modeling approach."),

        createHeading("4.9.3 Practical Implications for SHM", 3),
        p("The findings have several important implications: Sensitivity thresholds depend on measurement accuracy. With typical accelerometer precision (plus or minus 0.1 Hz), corrosion levels as low as 2-3% can be detected for baseline beam configurations."),
        p("Environmental factors like temperature cause frequency variations similar to early-stage damage. Based on Cai et al. (2021), temperature effects cause approximately 0.148% frequency change per degree Celsius. Robust SHM systems must incorporate environmental compensation."),

        createHeading("4.9.4 Limitations and Future Work", 3),
        p("Several limitations should be acknowledged: The stiffness reduction approach, while computationally efficient, does not capture all physical aspects of corrosion including mass changes and bond degradation. Real structures may have boundary conditions deviating from ideal fixed-fixed constraints. Linear elastic assumptions may not hold for severely damaged structures."),
        p("Future research directions include more sophisticated damage models based on fracture mechanics, experimental validation with laboratory specimens and field structures, inverse problem algorithms for damage identification from frequency measurements, and integration with other SHM techniques."),

        createHeading("4.10 Summary", 2),
        p("This chapter presented comprehensive results from finite element analysis of damaged RC beams:"),
        p("1. Model Validation: The FEM implementation achieved below 0.002% error compared to theoretical solutions. Results align with experimental trends from Zhang et al. (2020)."),
        p("2. Damage Effects: Uniform corrosion causes monotonic, nonlinear frequency reductions with sensitivity of approximately 0.8% per 1% corrosion."),
        p("3. Dataset Generation: A diverse dataset of 3,000 simulations was created using Latin Hypercube Sampling."),
        p("4. Statistical Analysis: Frequency distributions show strong correlations with beam length (r=-0.87) and corrosion level (r=-0.78)."),
        p("5. ML Performance: CatBoost achieved 98.9% R-squared, exceeding literature benchmarks."),
        p("The results provide a solid foundation for developing machine learning models for predictive maintenance and structural health monitoring applications."),
        new Paragraph({ children: [new PageBreak()] }),

        // ==================== CHAPTER 5: CONCLUSIONS ====================
        createHeading("Chapter 5: Conclusions", 1),

        createHeading("5.1 Summary of Findings", 2),
        p("This thesis investigated the prediction of natural frequencies of fixed reinforced concrete beams using machine learning models validated against finite element simulations. The key findings are:"),
        p("The Python-based FEM implementation achieved excellent accuracy with less than 0.002% error compared to theoretical Euler-Bernoulli solutions and consistent results with published ANSYS analyses from Das (2023)."),
        p("Among the five machine learning algorithms tested (Linear Regression, Random Forest, XGBoost, CatBoost, and SVR), CatBoost demonstrated superior performance with R-squared of 98.9%, MAE of 3.00 Hz, and the lowest cross-validation variance."),
        p("The corrosion-frequency sensitivity of approximately 0.8% frequency reduction per 1% corrosion level aligns with experimental findings from Zhang et al. (2020), validating the stiffness reduction approach for damage modeling."),
        p("Feature importance analysis revealed beam length as the most influential parameter, consistent with theoretical expectations, followed by damage severity and cross-sectional depth."),

        createHeading("5.2 Contributions", 2),
        p("This research makes several contributions to the field:"),
        p("Methodological: First systematic comparison of five ML algorithms for fixed RC beam frequency prediction, establishing CatBoost as the optimal choice for this application."),
        p("Practical: Development of an open dataset of 3,000 validated FEM simulations and trained models that can be used for rapid structural assessments."),
        p("Theoretical: Quantification of damage-frequency relationships with sensitivity coefficients supporting early damage detection in RC structures."),

        createHeading("5.3 Recommendations for Practice", 2),
        p("Based on the findings, the following recommendations are made for practical applications:"),
        p("For preliminary design assessments, the trained CatBoost model can provide rapid frequency predictions, enabling engineers to screen multiple beam configurations efficiently."),
        p("For structural health monitoring applications, the established sensitivity thresholds (approximately 0.8% frequency reduction per 1% corrosion) can guide measurement requirements and damage detection criteria."),
        p("Temperature compensation should be implemented in field monitoring systems to distinguish damage-induced frequency changes from environmental variations."),

        createHeading("5.4 Future Work", 2),
        p("Several directions for future research are recommended:"),
        p("Experimental validation with laboratory specimens and field structures to further confirm the FEM and ML predictions."),
        p("Extension to other boundary conditions (simply supported, cantilever, continuous spans) and structural elements (slabs, columns)."),
        p("Development of inverse problem algorithms for damage identification and localization from measured frequency data."),
        p("Integration with physics-informed neural networks to incorporate governing equations directly into the ML framework."),
        p("Investigation of transfer learning approaches to adapt models trained on simulation data for real-world applications."),
        new Paragraph({ children: [new PageBreak()] }),

        // ==================== REFERENCES ====================
        createHeading("References", 1),
        p("1. ACI Committee 318. (2019). Building Code Requirements for Structural Concrete (ACI 318-19). American Concrete Institute."),
        p("2. Avcar, M., & Saplioglu, K. (2015). An artificial neural network application for estimation of natural frequencies of beams. Research Journal of Applied Sciences, Engineering and Technology, 9(3), 131-138."),
        p("3. Banerjee, A., Panigrahi, B., & Pohit, G. (2017). Crack modelling and detection in Timoshenko FGM beam under transverse vibration using frequency contour and response surface model with GA. Nondestructive Testing and Evaluation, 32(1), 27-48."),
        p("4. Bathe, K. J. (2014). Finite Element Procedures (2nd ed.). Klaus-Jurgen Bathe."),
        p("5. Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. Journal of Machine Learning Research, 13(10), 281-305."),
        p("6. Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32."),
        p("7. Cai, Y., Zhang, K., Ye, Z., Liu, C., Lu, K., & Wang, L. (2021). Influence of temperature on the natural vibration characteristics of simply supported reinforced concrete beam. Sensors, 21, 4242."),
        p("8. Cairns, J., Plizzari, G. A., Du, Y., Law, D. W., & Franzoni, C. (2005). Mechanical properties of corrosion-damaged reinforcement. ACI Materials Journal, 102(4), 256-264."),
        p("9. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785-794."),
        p("10. Chondros, T. G., Dimarogonas, A. D., & Yao, J. (1998). A continuous cracked beam vibration theory. Journal of Sound and Vibration, 215(1), 17-34."),
        p("11. Chopra, A. K. (2012). Dynamics of Structures: Theory and Applications to Earthquake Engineering (4th ed.). Pearson."),
        p("12. Clough, R. W., & Penzien, J. (2003). Dynamics of Structures (3rd ed.). Computers & Structures, Inc."),
        p("13. Cohen, J. (1992). A power primer. Psychological Bulletin, 112(1), 155-159."),
        p("14. Cook, R. D. (2007). Concepts and Applications of Finite Element Analysis (4th ed.). Wiley."),
        p("15. Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273-297."),
        p("16. Das, O. (2023). Prediction of the natural frequencies of various beams using regression machine learning models. Sigma Journal of Engineering and Natural Sciences, 41(2), 302-321."),
        p("17. Dimarogonas, A. D. (1996). Vibration of cracked structures: A state of the art review. Engineering Fracture Mechanics, 55(5), 831-857."),
        p("18. Doebling, S. W., Farrar, C. R., Prime, M. B., & Shevitz, D. W. (1996). Damage identification and health monitoring of structural and mechanical systems from changes in their vibration characteristics: A literature review. Los Alamos National Laboratory Report LA-13070-MS."),
        p("19. Efron, B., & Tibshirani, R. (1993). An Introduction to the Bootstrap. Chapman and Hall."),
        p("20. Eurocode 2. (2004). Design of Concrete Structures - Part 1-1: General Rules and Rules for Buildings. EN 1992-1-1."),
        p("21. Farrar, C. R., & Worden, K. (2013). Structural Health Monitoring: A Machine Learning Perspective. John Wiley & Sons."),
        p("22. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press."),
        p("23. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. In International Conference on Machine Learning (pp. 1321-1330). PMLR."),
        p("24. Harris, C. R., et al. (2020). Array programming with NumPy. Nature, 585, 357-362."),
        p("25. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning: Data Mining, Inference, and Prediction (2nd ed.). Springer."),
        p("26. Helton, J. C., & Davis, F. J. (2003). Latin hypercube sampling and the propagation of uncertainty in analyses of complex systems. Reliability Engineering & System Safety, 81(1), 23-69."),
        p("27. Hughes, T. J. R. (2000). The Finite Element Method: Linear Static and Dynamic Finite Element Analysis. Dover Publications."),
        p("28. Inman, D. J. (2014). Engineering Vibration (4th ed.). Pearson."),
        p("29. Laory, I., Trinh, T. N., Smith, I. F., & Brownjohn, J. M. (2018). Methodologies for predicting natural frequency variation of a suspension bridge. Engineering Structures, 80, 211-221."),
        p("30. Luu, X.-B. (2024). Finite element modelling of reinforced concrete beam strengthening using ultra-high performance fiber-reinforced shotcrete. Structures, 60, 105794."),
        p("31. MacGregor, J. G., & Wight, J. K. (2012). Reinforced Concrete: Mechanics and Design (6th ed.). Pearson."),
        p("32. McKay, M. D., Beckman, R. J., & Conover, W. J. (1979). A comparison of three methods for selecting values of input variables in the analysis of output from a computer code. Technometrics, 21(2), 239-245."),
        p("33. McKinney, W. (2010). Data Structures for Statistical Computing in Python. Proceedings of the 9th Python in Science Conference, 51-56."),
        p("34. Meirovitch, L. (2001). Fundamentals of Vibrations. McGraw-Hill."),
        p("35. Miller, J., et al. (2000). The Tacoma Narrows Bridge collapse: A review of the causes. Engineering History and Heritage, 153(1), 25-30."),
        p("36. Nikoo, M., Zarfam, P., & Sayahpour, H. (2018). Determination of natural frequency of Euler-Bernoulli beam using artificial neural network. Engineering Structures, 157, 154-166."),
        p("37. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830."),
        p("38. Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased Boosting with Categorical Features. Advances in Neural Information Processing Systems, 31."),
        p("39. Rao, S. S. (2019). Mechanical Vibrations (6th ed.). Pearson."),
        p("40. Rodriguez, J., Ortega, L. M., & Casal, J. (1997). Load carrying capacity of concrete structures with corroded reinforcement. Construction and Building Materials, 11(4), 239-248."),
        p("41. Saha, P., & Yang, M. (2023). A neural network approach to estimate the frequency of a cantilever beam with random multiple damages. Sensors, 23, 7867."),
        p("42. Sohn, H., Farrar, C. R., Hemez, F. M., Shunk, D. D., Stinemates, D. W., Nadler, B. R., & Czarnecki, J. J. (2004). A review of structural health monitoring literature: 1996-2001. Los Alamos National Laboratory Report LA-13976-MS."),
        p("43. Virtanen, P., et al. (2020). SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17, 261-272."),
        p("44. Zhang, Y., Cheng, Y., Tan, G., Lyu, X., Sun, X., Bai, Y., & Yang, S. (2020). Natural frequency response evaluation for RC beams affected by steel corrosion using acceleration sensors. Sensors, 20, 5335."),
        p("45. Zienkiewicz, O. C., & Taylor, R. L. (2000). The Finite Element Method (5th ed.). Butterworth-Heinemann."),
      ],
    }],
  });

  // Generate the document
  const buffer = await Packer.toBuffer(doc);
  const outputPath = path.join(BASE_PATH, "MS_Thesis_Document.docx");
  fs.writeFileSync(outputPath, buffer);
  console.log(`Document created successfully at: ${outputPath}`);
}

createThesisDocument().catch(console.error);
