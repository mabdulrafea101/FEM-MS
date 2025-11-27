#!/usr/bin/env python3
import re

# Read PDF as binary and extract printable text
with open('help.pdf', 'rb') as f:
    content = f.read()
    
# Decode and extract text
text = content.decode('latin-1', errors='ignore')

# Find reference section
refs_start = text.find('References')
if refs_start == -1:
    refs_start = text.find('REFERENCES')
if refs_start == -1:
    refs_start = text.find('Bibliography')
    
print("=== FULL EXTRACTED TEXT (First 10000 chars) ===\n")
print(text[:10000])

if refs_start != -1:
    print("\n\n=== REFERENCES SECTION ===\n")
    print(text[refs_start:refs_start+5000])

# Save to file
with open('help_extracted.txt', 'w') as f:
    f.write(text)
    
print("\n\n=== Saved to help_extracted.txt ===")
