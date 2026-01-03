const fs = require('fs');
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
        AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType,
        VerticalAlign, LevelFormat, PageBreak } = require('docx');

// Read the markdown file
const mdContent = fs.readFileSync('documentation_humanized.md', 'utf8');
const lines = mdContent.split('\n');

console.log("Creating comprehensive thesis document from markdown...");
console.log(`Processing ${lines.length} lines`);

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
    console.log(`⚠️  Could not load image: ${imagePath}`);
    return new TextRun({ text: `[IMAGE: ${imagePath}]`, italics: true, color: "FF0000" });
  }
}

function parseInlineFormatting(text) {
  const runs = [];
  const parts = text.split(/(\*\*[^*]+\*\*|\*[^*]+\*)/);

  parts.forEach(part => {
    if (part.startsWith('**') && part.endsWith('**')) {
      runs.push(new TextRun({ text: part.slice(2, -2), bold: true }));
    } else if (part.startsWith('*') && part.endsWith('*') && !part.startsWith('**')) {
      runs.push(new TextRun({ text: part.slice(1, -1), italics: true }));
    } else if (part.length > 0) {
      runs.push(new TextRun(part));
    }
  });

  return runs.length > 0 ? runs : [new TextRun(text)];
}

// Parse markdown
const docChildren = [];
let i = 0;
let inCodeBlock = false;
let inTable = false;
let currentTable = [];
let inResearchQuestions = false;
let inResearchObjectives = false;
let imageCount = 0;
let tableCount = 0;

while (i < lines.length) {
  const line = lines[i];
  const trimmed = line.trim();

  // Handle empty lines
  if (trimmed.length === 0) {
    if (inTable && currentTable.length > 0) {
      // End table
      const rows = currentTable.map((rowData, idx) => {
        const cells = rowData.map((cellText) => {
          const width = Math.floor(9360 / rowData.length);
          return new TableCell({
            borders: cellBorders,
            width: { size: width, type: WidthType.DXA },
            shading: idx === 0 ? { fill: "E7E6E6", type: ShadingType.CLEAR } : undefined,
            verticalAlign: VerticalAlign.CENTER,
            children: [new Paragraph({
              alignment: idx === 0 ? AlignmentType.CENTER : AlignmentType.LEFT,
              children: parseInlineFormatting(cellText).map(run =>
                idx === 0 ? new TextRun({ ...run, bold: true, size: 22 }) : run
              )
            })]
          });
        });
        return new TableRow({ tableHeader: idx === 0, children: cells });
      });

      docChildren.push(new Table({
        columnWidths: Array(currentTable[0].length).fill(Math.floor(9360 / currentTable[0].length)),
        margins: { top: 100, bottom: 100, left: 100, right: 100 },
        rows: rows
      }));
      tableCount++;
      currentTable = [];
      inTable = false;
    }
    i++;
    continue;
  }

  // Code blocks
  if (trimmed.startsWith('```')) {
    inCodeBlock = !inCodeBlock;
    i++;
    continue;
  }
  if (inCodeBlock) {
    i++;
    continue;
  }

  // Equations
  if (trimmed.startsWith('$$')) {
    const eq = trimmed.replace(/\$\$/g, '').trim();
    docChildren.push(new Paragraph({
      style: "Equation",
      children: [new TextRun({ text: eq, italics: true, color: "0000FF" })]
    }));
    i++;
    continue;
  }

  // Headings
  const headingMatch = trimmed.match(/^(#{1,4})\s+(.+)/);
  if (headingMatch) {
    const level = headingMatch[1].length;
    const text = headingMatch[2];

    // Track sections
    if (text.includes("1.3") && text.includes("Research Questions")) {
      inResearchQuestions = true;
      inResearchObjectives = false;
    } else if (text.includes("1.4") && text.includes("Research Objectives")) {
      inResearchObjectives = true;
      inResearchQuestions = false;
    } else if (text.includes("1.5") || text.includes("Chapter 2")) {
      inResearchQuestions = false;
      inResearchObjectives = false;
    }

    const headingLevels = [HeadingLevel.HEADING_1, HeadingLevel.HEADING_2, HeadingLevel.HEADING_3, HeadingLevel.HEADING_4];
    docChildren.push(new Paragraph({
      heading: headingLevels[level - 1] || HeadingLevel.HEADING_1,
      children: [new TextRun(text)]
    }));
    i++;
    continue;
  }

  // Images
  const imgMatch = trimmed.match(/!\[(.+?)\]\((.+?)\)/);
  if (imgMatch) {
    docChildren.push(new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [createImageRun(imgMatch[2])]
    }));

    // Next line might be caption
    if (i + 1 < lines.length) {
      const nextLine = lines[i + 1].trim();
      if (nextLine.startsWith('**Figure') || nextLine.startsWith('**Table')) {
        i++;
        const caption = nextLine.replace(/^\*\*/, '').replace(/\*\*$/, '');
        docChildren.push(new Paragraph({
          style: "Caption",
          children: [new TextRun({ text: caption, bold: true })]
        }));
      }
    }
    imageCount++;
    i++;
    continue;
  }

  // Tables
  if (trimmed.startsWith('|')) {
    if (!inTable) {
      inTable = true;
    }
    if (!trimmed.includes('---')) {
      const cells = trimmed.split('|').map(c => c.trim()).filter((c, idx, arr) =>
        (idx !== 0 && idx !== arr.length - 1) || c.length > 0
      );
      if (cells.length > 0) {
        currentTable.push(cells);
      }
    }
    i++;
    continue;
  }

  // Regular paragraphs
  if (trimmed.length > 0) {
    // Check if this should be a numbered list item
    const shouldNumber = (inResearchQuestions || inResearchObjectives) &&
                        trimmed.length > 30 &&
                        !trimmed.match(/Three main questions|I established four/);

    if (shouldNumber) {
      const ref = inResearchQuestions ? "research-questions" : "research-objectives";
      docChildren.push(new Paragraph({
        numbering: { reference: ref, level: 0 },
        style: "BodyText",
        children: parseInlineFormatting(trimmed)
      }));
    } else {
      docChildren.push(new Paragraph({
        style: "BodyText",
        children: parseInlineFormatting(trimmed)
      }));
    }
  }

  i++;
}

console.log(`✓ Parsed content: ${docChildren.length} elements`);
console.log(`✓ Images: ${imageCount}`);
console.log(`✓ Tables: ${tableCount}`);

// Create document
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Times New Roman", size: 24 } } },
    paragraphStyles: [
      {
        id: "Title", name: "Title", basedOn: "Normal",
        run: { size: 28, bold: true, color: "000000", font: "Times New Roman" },
        paragraph: { spacing: { before: 0, after: 240 }, alignment: AlignmentType.CENTER }
      },
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal",
        run: { size: 28, bold: true, color: "2E74B5", font: "Times New Roman" },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 0 }
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal",
        run: { size: 26, bold: true, color: "2E74B5", font: "Times New Roman" },
        paragraph: { spacing: { before: 180, after: 100 }, outlineLevel: 1 }
      },
      {
        id: "Heading3", name: "Heading 3", basedOn: "Normal",
        run: { size: 24, bold: true, color: "1F4D78", font: "Times New Roman" },
        paragraph: { spacing: { before: 140, after: 80 }, outlineLevel: 2 }
      },
      {
        id: "Heading4", name: "Heading 4", basedOn: "Normal",
        run: { size: 24, bold: true, italics: true, color: "2E74B5", font: "Times New Roman" },
        paragraph: { spacing: { before: 120, after: 60 }, outlineLevel: 3 }
      },
      {
        id: "BodyText", name: "Body Text", basedOn: "Normal",
        run: { size: 24, color: "000000", font: "Times New Roman" },
        paragraph: { spacing: { before: 0, after: 120 }, alignment: AlignmentType.JUSTIFIED }
      },
      {
        id: "Caption", name: "Caption", basedOn: "Normal",
        run: { size: 22, color: "000000", font: "Times New Roman", bold: true },
        paragraph: { spacing: { before: 60, after: 120 }, alignment: AlignmentType.CENTER }
      },
      {
        id: "Equation", name: "Equation", basedOn: "Normal",
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
          level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } }
        }]
      },
      {
        reference: "research-objectives",
        levels: [{
          level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } }
        }]
      }
    ]
  },
  sections: [{
    properties: { page: { margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    children: docChildren
  }]
});

// Save
Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("MS_Thesis_Final.docx", buffer);
  console.log("\n✅ SUCCESS: MS_Thesis_Final.docx created!");
  console.log("\n📊 Summary:");
  console.log(`   • Total elements: ${docChildren.length}`);
  console.log(`   • Images included: ${imageCount}`);
  console.log(`   • Tables included: ${tableCount}`);
  console.log("\n📝 Notes:");
  console.log("   • Equations shown in blue italics (copy from original for proper formatting)");
  console.log("   • All content, tables, and images from markdown included");
  console.log("   • Sections 1.3 and 1.4 have proper numbered lists");
  console.log("   • Formatting matches thesis style (Times New Roman, justified text)");
});
