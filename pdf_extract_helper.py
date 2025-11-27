#!/usr/bin/env python3
"""
Extract text from help.pdf to analyze references
"""

import subprocess
import sys

def extract_pdf_simple():
    """Try to extract PDF using simple string extraction"""
    try:
        # Try using strings command as fallback
        result = subprocess.run(
            ['strings', 'help.pdf'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except Exception as e:
        print(f"Error extracting PDF: {e}", file=sys.stderr)
        return None

if __name__ == "__main__":
    print("Extracting text from help.pdf...")
    content = extract_pdf_simple()
    if content:
        with open('help_content_extracted.txt', 'w') as f:
            f.write(content)
        print("Content extracted to help_content_extracted.txt")
        # Print first 200 lines to see structure
        lines = content.split('\n')
        for i, line in enumerate(lines[:200]):
            if line.strip():
                print(line)
    else:
        print("Failed to extract content")
