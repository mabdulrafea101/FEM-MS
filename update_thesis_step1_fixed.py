#!/usr/bin/env python3
"""
Step 1: Update sections 1.3 and 1.4 to use numbered lists
Fixed version - collect all nodes first
"""
import sys
sys.path.insert(0, '/Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/.claude/skills/docx')

from scripts.document import Document

# Initialize document
doc = Document('docx_unpacked', author="Updated", initials="UP", rsid="6E1AE87D")

print("Updating Section 1.3 Research Questions to numbered list...")

# COLLECT ALL NODES FIRST before any replacements
nodes_to_replace = []

# Section 1.3 nodes
try:
    node1 = doc["word/document.xml"].get_node(tag="w:r", contains="First, how accurately can machine learning")
    nodes_to_replace.append(('1.3-1', node1.parentNode.parentNode))
except:
    print("Warning: Could not find 'First, how accurately...' - may already be updated")

try:
    node2 = doc["word/document.xml"].get_node(tag="w:r", contains="Second, which algorithm performs best")
    nodes_to_replace.append(('1.3-2', node2.parentNode.parentNode))
except:
    print("Warning: Could not find 'Second, which algorithm...' - may already be updated")

try:
    node3 = doc["word/document.xml"].get_node(tag="w:r", contains="Third, what are the most important parameters")
    nodes_to_replace.append(('1.3-3', node3.parentNode.parentNode))
except:
    print("Warning: Could not find 'Third, what are...' - may already be updated")

# Section 1.4 nodes
print("Collecting Section 1.4 nodes...")
try:
    node4 = doc["word/document.xml"].get_node(tag="w:r", contains="The first was generating a reliable dataset")
    nodes_to_replace.append(('1.4-1', node4.parentNode.parentNode))
except:
    print("Warning: Could not find 'The first was generating...' - may already be updated")

try:
    node5 = doc["word/document.xml"].get_node(tag="w:r", contains="The second objective involved developing and testing")
    nodes_to_replace.append(('1.4-2', node5.parentNode.parentNode))
except:
    print("Warning: Could not find 'The second objective...' - may already be updated")

try:
    node6 = doc["word/document.xml"].get_node(tag="w:r", contains="Third, I wanted to quantify which parameters")
    nodes_to_replace.append(('1.4-3', node6.parentNode.parentNode))
except:
    print("Warning: Could not find 'Third, I wanted to quantify...' - may already be updated")

try:
    node7 = doc["word/document.xml"].get_node(tag="w:r", contains="Finally, I planned to validate everything")
    nodes_to_replace.append(('1.4-4', node7.parentNode.parentNode))
except:
    print("Warning: Could not find 'Finally, I planned...' - may already be updated")

# NOW DO THE REPLACEMENTS
print(f"Found {len(nodes_to_replace)} nodes to replace")

for label, parent_node in nodes_to_replace:
    print(f"Replacing {label}...")

    if label == '1.3-1':
        replacement = '''<w:p w14:paraId="374A4748" w14:textId="77777777" w:rsidR="00206B36" w:rsidRDefault="00000000">
  <w:pPr>
    <w:pStyle w:val="ListParagraph"/>
    <w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr>
    <w:spacing w:after="200" w:line="360" w:lineRule="auto"/>
    <w:jc w:val="both"/>
  </w:pPr>
  <w:r>
    <w:t>How accurately can machine learning predict the fundamental natural frequency of fixed reinforced concrete beams? Previous work on steel beams achieved around 98 percent accuracy (Das, 2023), and I wanted to know if similar performance was achievable for RC beams with their more complex material behavior.</w:t>
  </w:r>
</w:p>'''
        doc["word/document.xml"].replace_node(parent_node, replacement)

    elif label == '1.3-2':
        replacement = '''<w:p w14:paraId="379D6508" w14:textId="77777777" w:rsidR="00206B36" w:rsidRDefault="00000000">
  <w:pPr>
    <w:pStyle w:val="ListParagraph"/>
    <w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr>
    <w:spacing w:after="200" w:line="360" w:lineRule="auto"/>
    <w:jc w:val="both"/>
  </w:pPr>
  <w:r>
    <w:t>Which algorithm performs best for this specific application? I compared Linear Regression, Random Forest, XGBoost, CatBoost, and Support Vector Regression because each brings different strengths to regression problems.</w:t>
  </w:r>
</w:p>'''
        doc["word/document.xml"].replace_node(parent_node, replacement)

    elif label == '1.3-3':
        replacement = '''<w:p w14:paraId="2C39AF83" w14:textId="77777777" w:rsidR="00206B36" w:rsidRDefault="00000000">
  <w:pPr>
    <w:pStyle w:val="ListParagraph"/>
    <w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr>
    <w:spacing w:after="200" w:line="360" w:lineRule="auto"/>
    <w:jc w:val="both"/>
  </w:pPr>
  <w:r>
    <w:t>What are the most important parameters? Understanding which geometric and material properties most strongly influence frequency could help engineers prioritize their measurements and design decisions.</w:t>
  </w:r>
</w:p>'''
        doc["word/document.xml"].replace_node(parent_node, replacement)

    elif label == '1.4-1':
        replacement = '''<w:p w14:paraId="0AD03887" w14:textId="77777777" w:rsidR="00206B36" w:rsidRDefault="00000000">
  <w:pPr>
    <w:pStyle w:val="ListParagraph"/>
    <w:numPr><w:ilvl w:val="0"/><w:numId w:val="2"/></w:numPr>
    <w:spacing w:after="200" w:line="360" w:lineRule="auto"/>
    <w:jc w:val="both"/>
  </w:pPr>
  <w:r>
    <w:t>Generating a reliable dataset. I aimed for 3,000 samples of natural frequency data from finite element simulations, targeting less than 0.01 percent error compared to theoretical solutions. This would ensure the ML models had trustworthy training data.</w:t>
  </w:r>
</w:p>'''
        doc["word/document.xml"].replace_node(parent_node, replacement)

    elif label == '1.4-2':
        replacement = '''<w:p w14:paraId="3C0EF2A2" w14:textId="77777777" w:rsidR="00206B36" w:rsidRDefault="00000000">
  <w:pPr>
    <w:pStyle w:val="ListParagraph"/>
    <w:numPr><w:ilvl w:val="0"/><w:numId w:val="2"/></w:numPr>
    <w:spacing w:after="200" w:line="360" w:lineRule="auto"/>
    <w:jc w:val="both"/>
  </w:pPr>
  <w:r>
    <w:t>Developing and testing five regression models, with a goal of achieving at least 95 percent R-squared on the test set.</w:t>
  </w:r>
</w:p>'''
        doc["word/document.xml"].replace_node(parent_node, replacement)

    elif label == '1.4-3':
        replacement = '''<w:p w14:paraId="247EF39D" w14:textId="77777777" w:rsidR="00206B36" w:rsidRDefault="00000000">
  <w:pPr>
    <w:pStyle w:val="ListParagraph"/>
    <w:numPr><w:ilvl w:val="0"/><w:numId w:val="2"/></w:numPr>
    <w:spacing w:after="200" w:line="360" w:lineRule="auto"/>
    <w:jc w:val="both"/>
  </w:pPr>
  <w:r>
    <w:t>Quantifying which parameters matter most using SHAP analysis and permutation importance. This would reveal whether length, depth, width, concrete strength, or damage severity has the greatest influence on frequency.</w:t>
  </w:r>
</w:p>'''
        doc["word/document.xml"].replace_node(parent_node, replacement)

    elif label == '1.4-4':
        replacement = '''<w:p w14:paraId="3554A1CE" w14:textId="77777777" w:rsidR="00206B36" w:rsidRDefault="00000000">
  <w:pPr>
    <w:pStyle w:val="ListParagraph"/>
    <w:numPr><w:ilvl w:val="0"/><w:numId w:val="2"/></w:numPr>
    <w:spacing w:after="200" w:line="360" w:lineRule="auto"/>
    <w:jc w:val="both"/>
  </w:pPr>
  <w:r>
    <w:t>Validating everything against published experimental data to confirm the physical realism of my simulations.</w:t>
  </w:r>
</w:p>'''
        doc["word/document.xml"].replace_node(parent_node, replacement)

# Save the document
print("Saving changes...")
doc.save()
print("Step 1 complete! Numbered lists have been added to sections 1.3 and 1.4")
