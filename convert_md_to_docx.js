const fs = require('fs');
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
        AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType,
        VerticalAlign, LevelFormat, PageBreak } = require('docx');

// Read the markdown file
const mdContent = fs.readFileSync('documentation_humanized.md', 'utf8');
const lines = mdContent.split('\n');

// Helper functions
const tableBorder = { style: BorderStyle.SINGLE, size: 1, color: "999999" };
const cellBorders = { top: tableBorder, bottom: tableBorder, left: tableBorder, right: tableBorder };

function createImageRun(imagePath, width = 580) {
  try {
    const imageData = fs.readFileSync(imagePath);
    const ext = imagePath.split('.').pop().toLowerCase();
    return new ImageRun({
      type: ext === 'jpg' ? 'jpeg' : ext,
      data: imageData,
      transformation: { width: width, height: width * 0.7 },
      altText: { title: "Figure", description: "Thesis figure", name: "Figure" }
    });
  } catch (e) {
    console.log(`Warning: Could not load image ${imagePath}: ${e.message}`);
    return new TextRun({ text: `[IMAGE: ${imagePath}]`, italics: true, color: "FF0000" });
  }
}

function isEquation(line) {
  return line.trim().startsWith('$$') || (line.includes('$$') && line.trim().length > 4);
}

function isHeading(line) {
  return line.trim().startsWith('#');
}

function getHeadingLevel(line) {
  const match = line.match(/^(#{1,4})\s/);
  if (!match) return null;
  return match[1].length;
}

function isTableRow(line) {
  return line.trim().startsWith('|');
}

function isFigure(line) {
  return line.trim().startsWith('![');
}

function parseTableRow(line) {
  return line.split('|').map(cell => cell.trim()).filter(cell => cell.length > 0);
}

function createParagraphFromText(text, style = "BodyText", numbered = null) {
  const children = [];

  // Handle bold and italic markdown
  let remaining = text;
  const boldRegex = /\*\*(.+?)\*\*/g;
  const italicRegex = /\*(.+?)\*/g;

  // Simple text run for now - can be enhanced
  children.push(new TextRun(text));

  const config = {
    style: style,
    children: children
  };

  if (numbered) {
    config.numbering = numbered;
  }

  return new Paragraph(config);
}

// Document structure
const docChildren = [];

// Parse the markdown
let i = 0;
let inCodeBlock = false;
let currentTable = [];
let inTable = false;
let tableHeaders = [];
let currentNumberingRef = null;

// Numbering configurations
const numberingConfigs = [
  {
    reference: "research-questions",
    levels: [{
      level: 0,
      format: LevelFormat.DECIMAL,
      text: "%1.",
      alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 720, hanging: 360 } } }
    }]
  },
  {
    reference: "research-objectives",
    levels: [{
      level: 0,
      format: LevelFormat.DECIMAL,
      text: "%1.",
      alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 720, hanging: 360 } } }
    }]
  }
];

console.log("Parsing markdown file...");
console.log(`Total lines: ${lines.length}`);

while (i < lines.length) {
  const line = lines[i];

  // Skip empty lines unless in special context
  if (line.trim().length === 0) {
    if (inTable) {
      // Table ended
      if (currentTable.length > 0) {
        // Create table
        const rows = currentTable.map((rowData, idx) => {
          const cells = rowData.map((cellText, cellIdx) => {
            const width = 9360 / rowData.length; // Divide page width equally
            return new TableCell({
              borders: cellBorders,
              width: { size: width, type: WidthType.DXA },
              shading: idx === 0 ? { fill: "E7E6E6", type: ShadingType.CLEAR } : undefined,
              verticalAlign: VerticalAlign.CENTER,
              children: [new Paragraph({
                alignment: idx === 0 ? AlignmentType.CENTER : AlignmentType.LEFT,
                children: [new TextRun({ text: cellText, bold: idx === 0, size: 22 })]
              })]
            });
          });

          return new TableRow({
            tableHeader: idx === 0,
            children: cells
          });
        });

        const columnWidths = Array(currentTable[0].length).fill(9360 / currentTable[0].length);
        docChildren.push(new Table({
          columnWidths: columnWidths,
          margins: { top: 100, bottom: 100, left: 100, right: 100 },
          rows: rows
        }));

        currentTable = [];
        inTable = false;
        tableHeaders = [];
      }
    }
    i++;
    continue;
  }

  // Handle code blocks
  if (line.trim().startsWith('```')) {
    inCodeBlock = !inCodeBlock;
    i++;
    continue;
  }

  if (inCodeBlock) {
    i++;
    continue; // Skip code blocks for now
  }

  // Handle equations
  if (isEquation(line)) {
    const equation = line.replace(/\$\$/g, '').trim();
    docChildren.push(new Paragraph({
      style: "Equation",
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: equation, italics: true, color: "0000FF" })]
    }));
    i++;
    continue;
  }

  // Handle headings
  if (isHeading(line)) {
    const level = getHeadingLevel(line);
    const text = line.replace(/^#{1,4}\s/, '').trim();

    // Check for section 1.3 or 1.4 for special numbering
    if (text.includes("1.3 Research Questions")) {
      currentNumberingRef = "research-questions";
    } else if (text.includes("1.4 Research Objectives")) {
      currentNumberingRef = "research-objectives";
    } else if (text.includes("1.5") || text.includes("Chapter 2") || text.includes("2.1")) {
      currentNumberingRef = null; // Reset numbering
    }

    let heading = HeadingLevel.HEADING_1;
    if (level === 1) heading = HeadingLevel.HEADING_1;
    else if (level === 2) heading = HeadingLevel.HEADING_2;
    else if (level === 3) heading = HeadingLevel.HEADING_3;
    else if (level === 4) heading = HeadingLevel.HEADING_4;

    docChildren.push(new Paragraph({
      heading: heading,
      children: [new TextRun(text)]
    }));
    i++;
    continue;
  }

  // Handle images
  if (isFigure(line)) {
    const match = line.match(/!\[(.+?)\]\((.+?)\)/);
    if (match) {
      const altText = match[1];
      const imagePath = match[2];

      docChildren.push(new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [createImageRun(imagePath)]
      }));

      // Add caption
      docChildren.push(new Paragraph({
        style: "Caption",
        children: [new TextRun({ text: altText, bold: true, size: 22 })]
      }));
    }
    i++;
    continue;
  }

  // Handle tables
  if (isTableRow(line)) {
    if (!inTable) {
      inTable = true;
      tableHeaders = parseTableRow(line);
      currentTable.push(tableHeaders);
    } else {
      // Check if it's a separator line
      if (line.includes('---') || line.includes('===')) {
        // Skip separator
      } else {
        const rowData = parseTableRow(line);
        if (rowData.length > 0 && rowData.length === tableHeaders.length) {
          currentTable.push(rowData);
        }
      }
    }
    i++;
    continue;
  }

  // Handle regular paragraphs
  if (line.trim().length > 0) {
    // Check for numbered list items in sections 1.3 and 1.4
    const trimmed = line.trim();

    // Check if this is a list item that should be numbered
    if (currentNumberingRef && !trimmed.startsWith('#') && trimmed.length > 20) {
      docChildren.push(new Paragraph({
        numbering: { reference: currentNumberingRef, level: 0 },
        style: "BodyText",
        children: [new TextRun(trimmed)]
      }));
    } else {
      // Regular paragraph
      docChildren.push(createParagraphFromText(trimmed, "BodyText"));
    }
  }

  i++;
}

console.log(`Processed ${docChildren.length} document elements`);

// Create the document
const doc = new Document({
  styles: {
    default: {
      document: { run: { font: "Times New Roman", size: 24 } }
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
        run: { size: 28, bold: true, color: "2E74B5", font: "Times New Roman" },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 0 }
      },
      {
        id: "Heading2",
        name: "Heading 2",
        basedOn: "Normal",
        run: { size: 26, bold: true, color: "2E74B5", font: "Times New Roman" },
        paragraph: { spacing: { before: 180, after: 100 }, outlineLevel: 1 }
      },
      {
        id: "Heading3",
        name: "Heading 3",
        basedOn: "Normal",
        run: { size: 24, bold: true, color: "1F4D78", font: "Times New Roman" },
        paragraph: { spacing: { before: 140, after: 80 }, outlineLevel: 2 }
      },
      {
        id: "Heading4",
        name: "Heading 4",
        basedOn: "Normal",
        run: { size: 24, bold: true, italics: true, color: "2E74B5", font: "Times New Roman" },
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
        run: { size: 24, color: "0000FF", font: "Cambria Math", italics: true },
        paragraph: { spacing: { before: 120, after: 120 }, alignment: AlignmentType.CENTER }
      }
    ]
  },
  numbering: {
    config: numberingConfigs
  },
  sections: [{
    properties: {
      page: {
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
      }
    },
    children: docChildren
  }]
});

// Save the document
console.log("Generating DOCX file...");
Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("MS_Thesis_Generated.docx", buffer);
  console.log("✓ Document created successfully: MS_Thesis_Generated.docx");
  console.log("\nNote: Equations are shown in blue italics - they will need to be converted to proper Word equations.");
  console.log("You can copy equations from the original MS_Thesis_Document.docx");
});
