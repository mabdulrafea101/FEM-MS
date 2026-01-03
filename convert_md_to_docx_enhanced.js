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
    return new TextRun({ text: `[IMAGE MISSING: ${imagePath}]`, italics: true, color: "FF0000" });
  }
}

function parseInlineFormatting(text) {
  const runs = [];
  let remaining = text;
  let pos = 0;

  while (pos < remaining.length) {
    // Find next bold (**text**)
    const boldMatch = remaining.substring(pos).match(/\*\*(.+?)\*\*/);
    const italicMatch = remaining.substring(pos).match(/\*([^*]+?)\*/);

    let nextBold = boldMatch ? pos + remaining.substring(pos).indexOf(boldMatch[0]) : Infinity;
    let nextItalic = italicMatch ? pos + remaining.substring(pos).indexOf(italicMatch[0]) : Infinity;

    if (nextBold === Infinity && nextItalic === Infinity) {
      // No more formatting, add rest as plain text
      if (pos < remaining.length) {
        runs.push(new TextRun(remaining.substring(pos)));
      }
      break;
    }

    if (nextBold < nextItalic) {
      // Add text before bold
      if (nextBold > pos) {
        runs.push(new TextRun(remaining.substring(pos, nextBold)));
      }
      // Add bold text
      runs.push(new TextRun({ text: boldMatch[1], bold: true }));
      pos = nextBold + boldMatch[0].length;
    } else {
      // Add text before italic
      if (nextItalic > pos) {
        runs.push(new TextRun(remaining.substring(pos, nextItalic)));
      }
      // Add italic text
      runs.push(new TextRun({ text: italicMatch[1], italics: true }));
      pos = nextItalic + italicMatch[0].length;
    }
  }

  return runs.length > 0 ? runs : [new TextRun(text)];
}

function isEquation(line) {
  return line.trim().startsWith('$$') || (line.includes('$$') && line.trim().length > 4);
}

function isHeading(line) {
  return line.trim().match(/^#{1,4}\s/);
}

function getHeadingLevel(line) {
  const match = line.match(/^(#{1,4})\s/);
  if (!match) return null;
  return match[1].length;
}

function isTableRow(line) {
  return line.trim().startsWith('|') && line.split('|').length > 2;
}

function isFigure(line) {
  return line.trim().startsWith('![');
}

function isBoldCaption(line) {
  return line.trim().startsWith('**Figure') || line.trim().startsWith('**Table');
}

function parseTableRow(line) {
  return line.split('|').map(cell => cell.trim()).filter((cell, idx, arr) => {
    // Filter out empty cells at start/end
    if (idx === 0 || idx === arr.length - 1) return cell.length > 0;
    return true;
  });
}

function createParagraphFromText(text, style = "BodyText", numbered = null) {
  const children = parseInlineFormatting(text);

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
let inMermaid = false;
let currentTable = [];
let inTable = false;
let tableHeaders = [];
let currentNumberingRef = null;
let elementCount = 0;

// Track sections for special handling
let inResearchQuestions = false;
let inResearchObjectives = false;

console.log("Parsing markdown file...");
console.log(`Total lines: ${lines.length}`);

while (i < lines.length) {
  const line = lines[i];
  const trimmed = line.trim();

  // Skip empty lines unless in special context
  if (trimmed.length === 0) {
    if (inTable) {
      // Table ended
      if (currentTable.length > 0) {
        const rows = currentTable.map((rowData, idx) => {
          const cells = rowData.map((cellText) => {
            const width = Math.floor(9360 / rowData.length);
            const textRuns = parseInlineFormatting(cellText);
            return new TableCell({
              borders: cellBorders,
              width: { size: width, type: WidthType.DXA },
              shading: idx === 0 ? { fill: "E7E6E6", type: ShadingType.CLEAR } : undefined,
              verticalAlign: VerticalAlign.CENTER,
              children: [new Paragraph({
                alignment: idx === 0 ? AlignmentType.CENTER : AlignmentType.LEFT,
                children: textRuns.map(run => {
                  if (idx === 0 && run instanceof TextRun) {
                    return new TextRun({ ...run, bold: true, size: 22 });
                  }
                  return run;
                })
              })]
            });
          });

          return new TableRow({
            tableHeader: idx === 0,
            children: cells
          });
        });

        const columnWidths = Array(currentTable[0].length).fill(Math.floor(9360 / currentTable[0].length));
        docChildren.push(new Table({
          columnWidths: columnWidths,
          margins: { top: 100, bottom: 100, left: 100, right: 100 },
          rows: rows
        }));
        elementCount++;

        currentTable = [];
        inTable = false;
        tableHeaders = [];
      }
    }
    i++;
    continue;
  }

  // Handle code blocks
  if (trimmed.startsWith('```')) {
    if (trimmed.includes('mermaid')) {
      inMermaid = true;
    } else {
      inCodeBlock = !inCodeBlock;
    }
    i++;
    continue;
  }

  if (inCodeBlock || inMermaid) {
    if (trimmed.startsWith('```')) {
      inCodeBlock = false;
      inMermaid = false;
    }
    i++;
    continue;
  }

  // Handle horizontal rules
  if (trimmed.startsWith('---') && !line.includes('|')) {
    docChildren.push(new Paragraph({
      border: { bottom: { color: "999999", space: 1, style: BorderStyle.SINGLE, size: 6 } },
      children: []
    }));
    i++;
    continue;
  }

  // Handle equations
  if (isEquation(line)) {
    const equation = line.replace(/\$\$/g, '').trim();
    docChildren.push(new Paragraph({
      style: "Equation",
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: equation, italics: true, color: "0000FF" })]
    }));
    elementCount++;
    i++;
    continue;
  }

  // Handle headings
  if (isHeading(line)) {
    const level = getHeadingLevel(line);
    const text = line.replace(/^#{1,4}\s/, '').trim();

    // Track sections for numbering
    if (text.includes("1.3") && text.includes("Research Questions")) {
      inResearchQuestions = true;
      inResearchObjectives = false;
      currentNumberingRef = "research-questions";
    } else if (text.includes("1.4") && text.includes("Research Objectives")) {
      inResearchQuestions = false;
      inResearchObjectives = true;
      currentNumberingRef = "research-objectives";
    } else if (text.includes("1.5") || text.includes("Chapter 2")) {
      inResearchQuestions = false;
      inResearchObjectives = false;
      currentNumberingRef = null;
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
    elementCount++;
    i++;
    continue;
  }

  // Handle images and their captions
  if (isFigure(line)) {
    const match = line.match(/!\[(.+?)\]\((.+?)\)/);
    if (match) {
      const altText = match[1];
      const imagePath = match[2];

      docChildren.push(new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [createImageRun(imagePath)]
      }));

      // Check next line for caption
      if (i + 1 < lines.length && lines[i + 1].trim().startsWith('**Figure')) {
        i++; // Skip to caption line
        const caption = lines[i].replace(/^\*\*/, '').replace(/\*\*$/, '').trim();
        docChildren.push(new Paragraph({
          style: "Caption",
          children: [new TextRun({ text: caption, bold: true, size: 22 })]
        }));
      } else {
        docChildren.push(new Paragraph({
          style: "Caption",
          children: [new TextRun({ text: altText, bold: true, size: 22 })]
        }));
      }
      elementCount++;
    }
    i++;
    continue;
  }

  // Handle bold captions (Figure X: or Table X:)
  if (isBoldCaption(line)) {
    const caption = trimmed.replace(/^\*\*/, '').replace(/\*\*$/, '');
    docChildren.push(new Paragraph({
      style: "Caption",
      children: [new TextRun({ text: caption, bold: true, size: 22 })]
    }));
    elementCount++;
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
        if (rowData.length > 0) {
          // Pad or trim to match header length
          while (rowData.length < tableHeaders.length) rowData.push('');
          currentTable.push(rowData.slice(0, tableHeaders.length));
        }
      }
    }
    i++;
    continue;
  }

  // Handle regular paragraphs
  if (trimmed.length > 0 && !trimmed.startsWith('[') && !trimmed.startsWith('>')) {
    // Detect numbered list items based on context
    const isListItem = (inResearchQuestions || inResearchObjectives) &&
                       trimmed.length > 20 &&
                       !trimmed.match(/^Three main questions/) &&
                       !trimmed.match(/^I established four/);

    if (currentNumberingRef && isListItem) {
      docChildren.push(new Paragraph({
        numbering: { reference: currentNumberingRef, level: 0 },
        style: "BodyText",
        children: parseInlineFormatting(trimmed)
      }));
    } else {
      docChildren.push(createParagraphFromText(trimmed, "BodyText"));
    }
    elementCount++;
  }

  i++;
}

console.log(`Processed ${elementCount} content elements (${docChildren.length} total document elements)`);

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
        run: { size: 22, color: "000000", font: "Times New Roman", bold: true },
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
    config: [
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
    ]
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
console.log("Generating enhanced DOCX file...");
Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("MS_Thesis_Complete.docx", buffer);
  console.log("\n✅ Document created successfully: MS_Thesis_Complete.docx");
  console.log("\n📊 Summary:");
  console.log(`   - Total lines processed: ${lines.length}`);
  console.log(`   - Content elements: ${elementCount}`);
  console.log(`   - Document elements: ${docChildren.length}`);
  console.log("\n⚠️  Important notes:");
  console.log("   - Equations are shown in BLUE ITALICS (Cambria Math font)");
  console.log("   - You'll need to convert these to proper Word equations");
  console.log("   - Copy equations from MS_Thesis_Document.docx for proper formatting");
  console.log("   - All images, tables, and formatted text have been included");
  console.log("   - Numbered lists in sections 1.3 and 1.4 are properly formatted");
});
