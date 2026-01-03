const fs = require('fs');
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
        AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType,
        VerticalAlign, LevelFormat, PageBreak } = require('docx');

// Helper function to create table borders
const tableBorder = { style: BorderStyle.SINGLE, size: 1, color: "999999" };
const cellBorders = { top: tableBorder, bottom: tableBorder, left: tableBorder, right: tableBorder };

// Helper function to read image and get dimensions
function createImageRun(imagePath, width = 600) {
  try {
    const imageData = fs.readFileSync(imagePath);
    const ext = imagePath.split('.').pop().toLowerCase();
    return new ImageRun({
      type: ext === 'jpg' ? 'jpeg' : ext,
      data: imageData,
      transformation: { width: width, height: width * 0.6 }, // Approximate aspect ratio
      altText: { title: "Figure", description: "Thesis figure", name: "Figure" }
    });
  } catch (e) {
    console.log(`Warning: Could not load image ${imagePath}`);
    return new TextRun({ text: `[IMAGE: ${imagePath}]`, italics: true, color: "FF0000" });
  }
}

// Create the document
const doc = new Document({
  styles: {
    default: {
      document: { run: { font: "Times New Roman", size: 24 } } // 12pt
    },
    paragraphStyles: [
      {
        id: "Title",
        name: "Title",
        basedOn: "Normal",
        run: { size: 28, bold: true, color: "000000", font: "Times New Roman" },
        paragraph: { spacing: { before: 0, after: 240 }, alignment: AlignmentType.CENTER }
      },
      {
        id: "Heading1",
        name: "Heading 1",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { size: 28, bold: true, color: "000000", font: "Times New Roman" },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 0 }
      },
      {
        id: "Heading2",
        name: "Heading 2",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { size: 26, bold: true, color: "000000", font: "Times New Roman" },
        paragraph: { spacing: { before: 180, after: 100 }, outlineLevel: 1 }
      },
      {
        id: "Heading3",
        name: "Heading 3",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { size: 24, bold: true, color: "000000", font: "Times New Roman" },
        paragraph: { spacing: { before: 140, after: 80 }, outlineLevel: 2 }
      },
      {
        id: "Heading4",
        name: "Heading 4",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { size: 24, bold: true, italics: true, color: "000000", font: "Times New Roman" },
        paragraph: { spacing: { before: 120, after: 60 }, outlineLevel: 3 }
      },
      {
        id: "BodyText",
        name: "Body Text",
        basedOn: "Normal",
        run: { size: 24, color: "000000", font: "Times New Roman" },
        paragraph: { spacing: { before: 0, after: 120 }, alignment: AlignmentType.JUSTIFIED }
      },
      {
        id: "Caption",
        name: "Caption",
        basedOn: "Normal",
        run: { size: 22, color: "000000", font: "Times New Roman" },
        paragraph: { spacing: { before: 60, after: 120 }, alignment: AlignmentType.CENTER }
      },
      {
        id: "Equation",
        name: "Equation",
        basedOn: "Normal",
        run: { size: 24, color: "000000", font: "Cambria Math", italics: true },
        paragraph: { spacing: { before: 120, after: 120 }, alignment: AlignmentType.CENTER }
      }
    ]
  },
  numbering: {
    config: [
      {
        reference: "research-questions-list",
        levels: [{
          level: 0,
          format: LevelFormat.DECIMAL,
          text: "%1.",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } }
        }]
      },
      {
        reference: "research-objectives-list",
        levels: [{
          level: 0,
          format: LevelFormat.DECIMAL,
          text: "%1.",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } }
        }]
      },
      {
        reference: "bullet-list",
        levels: [{
          level: 0,
          format: LevelFormat.BULLET,
          text: "•",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } }
        }]
      }
    ]
  },
  sections: [{
    properties: {
      page: {
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
      }
    },
    children: [
      // Title
      new Paragraph({
        heading: HeadingLevel.TITLE,
        children: [new TextRun("Prediction of Natural Frequencies of Fixed Reinforced Concrete Beams Using Machine Learning: A Finite Element Validated Approach")]
      }),

      new Paragraph({ children: [new PageBreak()] }),

      // Abstract
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun("Abstract")]
      }),

      new Paragraph({
        style: "BodyText",
        children: [new TextRun("Reinforced concrete beams form the backbone of most buildings and bridges we see around us. When these structures vibrate, they do so at specific rates called natural frequencies, and these frequencies tell us a lot about whether the structure is healthy or damaged. Getting accurate frequency predictions matters for safe design, avoiding dangerous resonance, and monitoring structural health over time.")]
      }),

      new Paragraph({
        style: "BodyText",
        children: [new TextRun("In this research, I set out to address something that has been missing in the literature: machine learning models built specifically for fixed-fixed reinforced concrete beams. While other researchers have done excellent work predicting frequencies for steel and aluminum beams, reinforced concrete with fixed boundary conditions remained largely unexplored. This gap seemed worth filling because fixed supports are everywhere in building frames and bridge connections.")]
      }),

      new Paragraph({
        style: "BodyText",
        children: [new TextRun("My approach combined finite element simulations based on Euler-Bernoulli beam theory with five different machine learning algorithms. I generated 3,000 beam samples using Latin Hypercube Sampling, covering beam lengths from 3 to 8 meters, widths from 0.2 to 0.5 meters, depths from 0.3 to 0.7 meters, concrete strengths between 25 and 50 MPa, and corrosion damage levels up to 20 percent. To model damage, I used stiffness reduction methods that previous experimental studies had validated.")]
      }),

      new Paragraph({
        style: "BodyText",
        children: [new TextRun("The findings suggest that machine learning can indeed predict frequencies with high accuracy for this type of structure. More importantly, this work opens up possibilities for rapid structural assessments, where engineers could screen dozens or even hundreds of beams in minutes rather than hours. The methodology I developed here could potentially be extended to other structural elements and damage scenarios.")]
      }),

      new Paragraph({
        style: "BodyText",
        children: [
          new TextRun({ text: "Keywords: ", bold: true }),
          new TextRun("Machine Learning, Natural Frequency, Reinforced Concrete, Finite Element Method, Structural Health Monitoring, Damage Detection")
        ]
      }),

      new Paragraph({ children: [new PageBreak()] })
    ]
  }]
});

console.log("Creating thesis document...");
console.log("Note: Due to the large size of the document, this script creates the initial structure.");
console.log("The full document will need to be completed in multiple parts.");

// Save the document
Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("MS_Thesis_New.docx", buffer);
  console.log("Document created: MS_Thesis_New.docx");
  console.log("\nNext steps:");
  console.log("1. This creates the basic structure");
  console.log("2. Due to file size, I'll create separate scripts for each chapter");
  console.log("3. Run generate_thesis_complete.js for the full document");
});
