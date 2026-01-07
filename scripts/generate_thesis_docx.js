/**
 * MS Thesis DOCX Generator
 * Converts documentation_humanized.md to a professionally formatted Word document
 *
 * Author: Generated for MS Thesis
 * Date: January 2025
 */

const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, AlignmentType, BorderStyle, PageBreak, Footer, Header,
  convertInchesToTwip, LevelFormat, PageNumber, ShadingType, VerticalAlign
} = require("docx");
const fs = require("fs");
const path = require("path");

const BASE_PATH = "/Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project";
const MARKDOWN_FILE = path.join(BASE_PATH, "documentation_humanized.md");

// Table border style
const tableBorder = { style: BorderStyle.SINGLE, size: 1, color: "AAAAAA" };
const cellBorders = { top: tableBorder, bottom: tableBorder, left: tableBorder, right: tableBorder };

// Parse markdown content
function parseMarkdown(content) {
  const lines = content.split('\n');
  const elements = [];
  let i = 0;
  let inCodeBlock = false;
  let codeContent = [];
  let inTable = false;
  let tableRows = [];

  while (i < lines.length) {
    const line = lines[i];

    // Handle code blocks
    if (line.startsWith('```')) {
      if (inCodeBlock) {
        elements.push({ type: 'code', content: codeContent.join('\n') });
        codeContent = [];
        inCodeBlock = false;
      } else {
        inCodeBlock = true;
      }
      i++;
      continue;
    }

    if (inCodeBlock) {
      codeContent.push(line);
      i++;
      continue;
    }

    // Handle tables
    if (line.startsWith('|') && line.includes('|')) {
      if (!inTable) {
        inTable = true;
        tableRows = [];
      }
      // Skip separator lines
      if (!line.match(/^\|[\s\-\|:]+\|$/)) {
        const cells = line.split('|').filter(c => c.trim() !== '').map(c => c.trim());
        if (cells.length > 0) {
          tableRows.push(cells);
        }
      }
      i++;
      continue;
    } else if (inTable) {
      if (tableRows.length > 0) {
        elements.push({ type: 'table', rows: tableRows });
      }
      inTable = false;
      tableRows = [];
    }

    // Handle horizontal rules
    if (line.match(/^---+$/)) {
      i++;
      continue;
    }

    // Handle headings
    if (line.startsWith('#')) {
      const match = line.match(/^(#+)\s*(.+)$/);
      if (match) {
        const level = match[1].length;
        const text = match[2].trim();
        elements.push({ type: 'heading', level, text });
      }
      i++;
      continue;
    }

    // Handle equations ($$...$$)
    if (line.startsWith('$$')) {
      let eqContent = line.substring(2);
      if (!line.endsWith('$$') || line === '$$') {
        i++;
        while (i < lines.length && !lines[i].endsWith('$$')) {
          eqContent += '\n' + lines[i];
          i++;
        }
        if (i < lines.length) {
          eqContent += '\n' + lines[i].replace(/\$\$$/, '');
        }
      } else {
        eqContent = eqContent.substring(0, eqContent.length - 2);
      }
      elements.push({ type: 'equation', content: eqContent.trim() });
      i++;
      continue;
    }

    // Handle figure references
    if (line.startsWith('![')) {
      const match = line.match(/!\[([^\]]*)\]\(([^)]+)\)/);
      if (match) {
        elements.push({ type: 'figure', alt: match[1], path: match[2] });
      }
      i++;
      continue;
    }

    // Handle bold text lines (like **Table X.X: ...**)
    if (line.startsWith('**') && line.includes(':**')) {
      const text = line.replace(/\*\*/g, '');
      elements.push({ type: 'caption', text });
      i++;
      continue;
    }

    // Handle regular paragraphs
    if (line.trim() !== '') {
      elements.push({ type: 'paragraph', text: line });
    }

    i++;
  }

  // Handle any remaining table
  if (inTable && tableRows.length > 0) {
    elements.push({ type: 'table', rows: tableRows });
  }

  return elements;
}

// Create paragraph with formatting
function createParagraph(text, options = {}) {
  // Process inline formatting
  const runs = [];
  let remaining = text;

  // Simple regex-based inline formatting
  const parts = remaining.split(/(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/);

  for (const part of parts) {
    if (part.startsWith('**') && part.endsWith('**')) {
      runs.push(new TextRun({
        text: part.slice(2, -2),
        font: "Times New Roman",
        size: options.size || 24,
        bold: true,
      }));
    } else if (part.startsWith('*') && part.endsWith('*') && !part.startsWith('**')) {
      runs.push(new TextRun({
        text: part.slice(1, -1),
        font: "Times New Roman",
        size: options.size || 24,
        italics: true,
      }));
    } else if (part.startsWith('`') && part.endsWith('`')) {
      runs.push(new TextRun({
        text: part.slice(1, -1),
        font: "Courier New",
        size: options.size || 22,
      }));
    } else if (part.trim() !== '') {
      runs.push(new TextRun({
        text: part,
        font: "Times New Roman",
        size: options.size || 24,
        bold: options.bold || false,
        italics: options.italics || false,
      }));
    }
  }

  if (runs.length === 0) {
    runs.push(new TextRun({ text: text, font: "Times New Roman", size: options.size || 24 }));
  }

  return new Paragraph({
    children: runs,
    spacing: { after: 200, line: 360 },
    alignment: options.alignment || AlignmentType.JUSTIFIED,
    indent: options.indent,
  });
}

// Create heading paragraph
function createHeading(text, level) {
  const headingLevel = level === 1 ? HeadingLevel.HEADING_1 :
                       level === 2 ? HeadingLevel.HEADING_2 :
                       level === 3 ? HeadingLevel.HEADING_3 : HeadingLevel.HEADING_4;
  const size = level === 1 ? 32 : level === 2 ? 28 : level === 3 ? 26 : 24;

  return new Paragraph({
    children: [
      new TextRun({
        text: text,
        font: "Times New Roman",
        size,
        bold: true,
      }),
    ],
    heading: headingLevel,
    spacing: { before: level === 1 ? 480 : 360, after: 240, line: 360 },
  });
}

// Create table from rows
function createTable(rows) {
  if (rows.length === 0) return null;

  const numCols = Math.max(...rows.map(r => r.length));
  const colWidth = Math.floor(9000 / numCols);

  const tableRows = rows.map((row, rowIndex) => {
    const isHeader = rowIndex === 0;
    return new TableRow({
      children: row.map(cell =>
        new TableCell({
          children: [new Paragraph({
            children: [new TextRun({
              text: cell,
              font: "Times New Roman",
              size: 22,
              bold: isHeader,
            })],
            alignment: AlignmentType.CENTER,
          })],
          borders: cellBorders,
          width: { size: colWidth, type: WidthType.DXA },
          shading: isHeader ? { fill: "E6E6E6", type: ShadingType.CLEAR } : undefined,
          verticalAlign: VerticalAlign.CENTER,
        })
      ),
    });
  });

  return new Table({
    rows: tableRows,
    width: { size: 100, type: WidthType.PERCENTAGE },
  });
}

// Create equation paragraph
function createEquation(content) {
  // Clean up LaTeX for display
  let cleanEq = content
    .replace(/\\quad/g, '    ')
    .replace(/\\text\{([^}]+)\}/g, '$1')
    .replace(/\\frac\{([^}]+)\}\{([^}]+)\}/g, '($1)/($2)')
    .replace(/\\sqrt\{([^}]+)\}/g, '√($1)')
    .replace(/\\times/g, '×')
    .replace(/\\cdot/g, '·')
    .replace(/\\leq/g, '≤')
    .replace(/\\geq/g, '≥')
    .replace(/\\approx/g, '≈')
    .replace(/\\pi/g, 'π')
    .replace(/\\omega/g, 'ω')
    .replace(/\\rho/g, 'ρ')
    .replace(/\\alpha/g, 'α')
    .replace(/\\beta/g, 'β')
    .replace(/\\lambda/g, 'λ')
    .replace(/\\Delta/g, 'Δ')
    .replace(/\\sum/g, 'Σ')
    .replace(/\\begin\{[^}]+\}|\end\{[^}]+\}/g, '')
    .replace(/\\\\/g, ' ')
    .replace(/&/g, ' ')
    .replace(/_\{([^}]+)\}/g, '_$1')
    .replace(/\^\{([^}]+)\}/g, '^$1')
    .replace(/\{|\}/g, '')
    .trim();

  return new Paragraph({
    children: [
      new TextRun({
        text: cleanEq,
        font: "Cambria Math",
        size: 24,
      }),
    ],
    spacing: { before: 240, after: 240 },
    alignment: AlignmentType.CENTER,
  });
}

// Create code block
function createCodeBlock(content) {
  const lines = content.split('\n');
  return lines.map(line =>
    new Paragraph({
      children: [
        new TextRun({
          text: line || ' ',
          font: "Courier New",
          size: 20,
        }),
      ],
      spacing: { after: 0 },
      shading: { fill: "F5F5F5", type: ShadingType.CLEAR },
    })
  );
}

// Create figure placeholder
function createFigure(alt, figPath) {
  return [
    new Paragraph({
      children: [
        new TextRun({
          text: `[Figure: ${figPath}]`,
          font: "Times New Roman",
          size: 22,
          italics: true,
        }),
      ],
      alignment: AlignmentType.CENTER,
      spacing: { before: 200, after: 100 },
    }),
    new Paragraph({
      children: [
        new TextRun({
          text: alt || 'Figure',
          font: "Times New Roman",
          size: 22,
        }),
      ],
      alignment: AlignmentType.CENTER,
      spacing: { after: 200 },
    }),
  ];
}

// Create caption paragraph
function createCaption(text) {
  return new Paragraph({
    children: [
      new TextRun({
        text: text,
        font: "Times New Roman",
        size: 22,
        bold: true,
      }),
    ],
    alignment: AlignmentType.CENTER,
    spacing: { before: 200, after: 100 },
  });
}

// Main function
async function createThesisDocument() {
  console.log("Reading markdown file...");
  const markdownContent = fs.readFileSync(MARKDOWN_FILE, 'utf-8');

  console.log("Parsing markdown content...");
  const elements = parseMarkdown(markdownContent);

  console.log(`Found ${elements.length} elements`);

  // Build document children
  const children = [];
  let pageBreakNeeded = false;

  for (const el of elements) {
    // Add page break before major chapters
    if (el.type === 'heading' && el.level === 1 && children.length > 0) {
      children.push(new Paragraph({ children: [new PageBreak()] }));
    }

    switch (el.type) {
      case 'heading':
        children.push(createHeading(el.text, el.level));
        break;

      case 'paragraph':
        children.push(createParagraph(el.text));
        break;

      case 'table':
        const table = createTable(el.rows);
        if (table) children.push(table);
        children.push(new Paragraph({ spacing: { after: 200 } }));
        break;

      case 'equation':
        children.push(createEquation(el.content));
        break;

      case 'code':
        children.push(...createCodeBlock(el.content));
        children.push(new Paragraph({ spacing: { after: 200 } }));
        break;

      case 'figure':
        children.push(...createFigure(el.alt, el.path));
        break;

      case 'caption':
        children.push(createCaption(el.text));
        break;
    }
  }

  console.log("Creating document...");

  const doc = new Document({
    styles: {
      default: {
        document: {
          run: { font: "Times New Roman", size: 24 },
        },
        heading1: {
          run: { font: "Times New Roman", size: 32, bold: true },
          paragraph: { spacing: { before: 480, after: 240 } },
        },
        heading2: {
          run: { font: "Times New Roman", size: 28, bold: true },
          paragraph: { spacing: { before: 360, after: 180 } },
        },
        heading3: {
          run: { font: "Times New Roman", size: 26, bold: true },
          paragraph: { spacing: { before: 240, after: 120 } },
        },
      },
      paragraphStyles: [
        {
          id: "Heading1",
          name: "Heading 1",
          basedOn: "Normal",
          next: "Normal",
          quickFormat: true,
          run: { font: "Times New Roman", size: 32, bold: true },
          paragraph: { spacing: { before: 480, after: 240 }, outlineLevel: 0 },
        },
        {
          id: "Heading2",
          name: "Heading 2",
          basedOn: "Normal",
          next: "Normal",
          quickFormat: true,
          run: { font: "Times New Roman", size: 28, bold: true },
          paragraph: { spacing: { before: 360, after: 180 }, outlineLevel: 1 },
        },
        {
          id: "Heading3",
          name: "Heading 3",
          basedOn: "Normal",
          next: "Normal",
          quickFormat: true,
          run: { font: "Times New Roman", size: 26, bold: true },
          paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 2 },
        },
      ],
    },
    numbering: {
      config: [
        {
          reference: "bullet-list",
          levels: [{
            level: 0,
            format: LevelFormat.BULLET,
            text: "\u2022",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } },
          }],
        },
        {
          reference: "numbered-list",
          levels: [{
            level: 0,
            format: LevelFormat.DECIMAL,
            text: "%1.",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } },
          }],
        },
      ],
    },
    sections: [{
      properties: {
        page: {
          margin: {
            top: convertInchesToTwip(1),
            right: convertInchesToTwip(1),
            bottom: convertInchesToTwip(1),
            left: convertInchesToTwip(1.25),
          },
        },
      },
      headers: {
        default: new Header({
          children: [new Paragraph({
            alignment: AlignmentType.RIGHT,
            children: [new TextRun({
              text: "MS Thesis - Natural Frequency Prediction of RC Beams",
              font: "Times New Roman",
              size: 20,
              italics: true,
            })],
          })],
        }),
      },
      footers: {
        default: new Footer({
          children: [new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [
              new TextRun({ text: "Page ", font: "Times New Roman", size: 20 }),
              new TextRun({ children: [PageNumber.CURRENT], font: "Times New Roman", size: 20 }),
              new TextRun({ text: " of ", font: "Times New Roman", size: 20 }),
              new TextRun({ children: [PageNumber.TOTAL_PAGES], font: "Times New Roman", size: 20 }),
            ],
          })],
        }),
      },
      children: children,
    }],
  });

  // Generate the document
  console.log("Generating DOCX file...");
  const buffer = await Packer.toBuffer(doc);
  const outputPath = path.join(BASE_PATH, "MS_Thesis_Document.docx");
  fs.writeFileSync(outputPath, buffer);

  console.log(`\n✓ Document created successfully!`);
  console.log(`  Location: ${outputPath}`);
  console.log(`  Size: ${(buffer.length / 1024).toFixed(1)} KB`);

  return outputPath;
}

// Run the script
createThesisDocument().catch(err => {
  console.error("Error creating document:", err);
  process.exit(1);
});
