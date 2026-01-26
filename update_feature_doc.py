#!/usr/bin/env python3
"""
Update feature-rag-application-1.md with content from SPARC documents.

This script reads the SPARC documentation files (Specification.md, Pseudocode.md,
Architecture.md, Refinement.md, and Completion.md) and updates the feature
implementation document with the latest comprehensive documentation.

The script follows PEP 8 style guidelines and uses type hints for all
function parameters and returns.
"""

import os
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path


def read_file_content(file_path: str) -> str:
    """
    Read and return the content of a file.

    Parameters:
    file_path (str): The path to the file to read.

    Returns:
    str: The content of the file.

    Raises:
    FileNotFoundError: If the file does not exist.
    IOError: If there is an error reading the file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except IOError as e:
        raise IOError(f"Error reading file {file_path}: {str(e)}")


def extract_section(content: str, start_marker: str,
                    end_marker: Optional[str] = None) -> str:
    """
    Extract a section from markdown content between markers.

    Parameters:
    content (str): The full markdown content.
    start_marker (str): The heading or marker to start extraction from.
    end_marker (Optional[str]): The heading or marker to end extraction at.
                                If None, extracts to the end of content.

    Returns:
    str: The extracted section content.
    """
    lines = content.split('\n')
    start_idx = -1
    end_idx = len(lines)

    # Find start marker
    for i, line in enumerate(lines):
        if start_marker in line:
            start_idx = i
            break

    if start_idx == -1:
        return ""

    # Find end marker if provided
    if end_marker:
        for i in range(start_idx + 1, len(lines)):
            if end_marker in lines[i]:
                end_idx = i
                break

    return '\n'.join(lines[start_idx:end_idx])


def merge_sparc_documents(sparc_dir: str) -> Dict[str, str]:
    """
    Read all SPARC documents and organize their content by type.

    Parameters:
    sparc_dir (str): The directory containing SPARC documents.

    Returns:
    Dict[str, str]: A dictionary with document types as keys and
                    their content as values.

    Raises:
    FileNotFoundError: If any required SPARC document is not found.
    """
    required_docs = [
        'Specification.md',
        'Pseudocode.md',
        'Architecture.md',
        'Refinement.md',
        'Completion.md'
    ]

    sparc_content = {}

    for doc_name in required_docs:
        doc_path = os.path.join(sparc_dir, doc_name)
        try:
            content = read_file_content(doc_path)
            # Store content with doc type as key (without .md extension)
            doc_type = doc_name.replace('.md', '')
            sparc_content[doc_type] = content
            print(f"✓ Successfully read {doc_name}")
        except (FileNotFoundError, IOError) as e:
            print(f"✗ Error reading {doc_name}: {str(e)}")
            raise

    return sparc_content


def generate_updated_feature_doc(sparc_content: Dict[str, str],
                                 original_content: str) -> str:
    """
    Generate updated feature document content by merging SPARC documentation.

    Parameters:
    sparc_content (Dict[str, str]): Dictionary containing SPARC document
                                    contents.
    original_content (str): The original feature document content.

    Returns:
    str: The updated feature document content.
    """
    # Extract the YAML frontmatter from original document
    lines = original_content.split('\n')
    frontmatter_end = -1

    for i in range(1, len(lines)):
        if lines[i].strip() == '---' and i > 0:
            frontmatter_end = i
            break

    # Preserve frontmatter and update last_updated date
    frontmatter = []
    if frontmatter_end > 0:
        for line in lines[:frontmatter_end + 1]:
            if line.startswith('last_updated:'):
                frontmatter.append(
                    f"last_updated: {datetime.now().strftime('%Y-%m-%d')}"
                )
            else:
                frontmatter.append(line)
    else:
        # Create default frontmatter if none exists
        frontmatter = [
            '---',
            'goal: Implement PayerPolicy_ChatBot - Privacy-Focused Local RAG'
            ' Application with Ollama',
            'version: 1.0',
            f"date_created: {datetime.now().strftime('%Y-%m-%d')}",
            f"last_updated: {datetime.now().strftime('%Y-%m-%d')}",
            'owner: Development Team',
            "status: 'In Progress'",
            "tags: ['feature', 'rag', 'llm', 'document-processing',"
            " 'vector-database', 'flask', 'ollama']",
            '---'
        ]

    # Build the updated document
    updated_doc = '\n'.join(frontmatter) + '\n\n'

    # Add Introduction section
    updated_doc += "# PayerPolicy ChatBot - Feature Implementation Plan\n\n"
    updated_doc += "![Status: In Progress]"
    updated_doc += "(https://img.shields.io/badge/status-In%20Progress-yellow)\n\n"

    updated_doc += "This comprehensive implementation plan integrates "
    updated_doc += "documentation from the SPARC methodology "
    updated_doc += "(Specification, Pseudocode, Architecture, Refinement, "
    updated_doc += "and Completion) to provide a complete "
    updated_doc += "development roadmap for the PayerPolicy_ChatBot.\n\n"

    # Add table of contents
    updated_doc += "## Table of Contents\n\n"
    updated_doc += "1. [Specification](#1-specification)\n"
    updated_doc += "2. [Pseudocode](#2-pseudocode)\n"
    updated_doc += "3. [Architecture](#3-architecture)\n"
    updated_doc += "4. [Refinement](#4-refinement)\n"
    updated_doc += "5. [Completion](#5-completion)\n\n"

    updated_doc += "---\n\n"

    # Section 1: Specification
    updated_doc += "## 1. Specification\n\n"
    updated_doc += "This section defines the project requirements, "
    updated_doc += "constraints, and functional specifications.\n\n"
    if 'Specification' in sparc_content:
        spec_content = sparc_content['Specification']
        # Extract content after the main title
        spec_lines = spec_content.split('\n')
        # Skip the first few lines (title and SPARC heading)
        content_start = 0
        for i, line in enumerate(spec_lines):
            if line.strip().startswith('## **S - SPECIFICATION**'):
                content_start = i + 1
                break
        if content_start > 0:
            updated_doc += '\n'.join(spec_lines[content_start:]) + '\n\n'
    updated_doc += "---\n\n"

    # Section 2: Pseudocode
    updated_doc += "## 2. Pseudocode\n\n"
    updated_doc += "This section provides detailed pseudocode for all "
    updated_doc += "major system components.\n\n"
    if 'Pseudocode' in sparc_content:
        pseudo_content = sparc_content['Pseudocode']
        # Extract content after the main title
        pseudo_lines = pseudo_content.split('\n')
        # Skip title line
        content_start = 2 if len(pseudo_lines) > 2 else 0
        if content_start > 0:
            updated_doc += '\n'.join(pseudo_lines[content_start:]) + '\n\n'
    updated_doc += "---\n\n"

    # Section 3: Architecture
    updated_doc += "## 3. Architecture\n\n"
    updated_doc += "This section describes the system architecture, "
    updated_doc += "technology stack, and component design.\n\n"
    if 'Architecture' in sparc_content:
        arch_content = sparc_content['Architecture']
        # Extract content after the main title
        arch_lines = arch_content.split('\n')
        # Skip title line
        content_start = 2 if len(arch_lines) > 2 else 0
        if content_start > 0:
            updated_doc += '\n'.join(arch_lines[content_start:]) + '\n\n'
    updated_doc += "---\n\n"

    # Section 4: Refinement
    updated_doc += "## 4. Refinement\n\n"
    updated_doc += "This section details optimization strategies, "
    updated_doc += "code quality improvements, and documentation standards.\n\n"
    if 'Refinement' in sparc_content:
        refine_content = sparc_content['Refinement']
        # Extract content after the main title
        refine_lines = refine_content.split('\n')
        # Skip title line
        content_start = 2 if len(refine_lines) > 2 else 0
        if content_start > 0:
            updated_doc += '\n'.join(refine_lines[content_start:]) + '\n\n'
    updated_doc += "---\n\n"

    # Section 5: Completion
    updated_doc += "## 5. Completion\n\n"
    updated_doc += "This section provides installation instructions, "
    updated_doc += "deployment guides, and final documentation.\n\n"
    if 'Completion' in sparc_content:
        complete_content = sparc_content['Completion']
        # Extract content after the main title
        complete_lines = complete_content.split('\n')
        # Skip title line
        content_start = 2 if len(complete_lines) > 2 else 0
        if content_start > 0:
            updated_doc += '\n'.join(complete_lines[content_start:]) + '\n\n'

    return updated_doc


def write_file_content(file_path: str, content: str) -> None:
    """
    Write content to a file.

    Parameters:
    file_path (str): The path to the file to write.
    content (str): The content to write to the file.

    Raises:
    IOError: If there is an error writing to the file.
    """
    try:
        # Create backup of original file
        if os.path.exists(file_path):
            backup_path = f"{file_path}.backup"
            with open(file_path, 'r', encoding='utf-8') as src:
                with open(backup_path, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())
            print(f"✓ Created backup at {backup_path}")

        # Write updated content
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"✓ Successfully wrote to {file_path}")
    except IOError as e:
        raise IOError(f"Error writing to file {file_path}: {str(e)}")


def main() -> None:
    """
    Main function to orchestrate the document update process.

    This function reads SPARC documents, generates updated content,
    and writes it to the feature implementation document.

    Raises:
    Exception: If any error occurs during the update process.
    """
    try:
        # Define paths
        base_dir = Path(__file__).parent
        sparc_dir = base_dir / 'SPARC_Documents'
        feature_doc_path = base_dir / 'plan' / 'feature-rag-application-1.md'

        print("=" * 70)
        print("PayerPolicy ChatBot - Feature Document Update Tool")
        print("=" * 70)
        print()

        # Validate paths exist
        if not sparc_dir.exists():
            raise FileNotFoundError(
                f"SPARC_Documents directory not found: {sparc_dir}"
            )

        if not feature_doc_path.exists():
            raise FileNotFoundError(
                f"Feature document not found: {feature_doc_path}"
            )

        print(f"Base directory: {base_dir}")
        print(f"SPARC directory: {sparc_dir}")
        print(f"Feature document: {feature_doc_path}")
        print()

        # Step 1: Read SPARC documents
        print("Step 1: Reading SPARC documents...")
        print("-" * 70)
        sparc_content = merge_sparc_documents(str(sparc_dir))
        print()

        # Step 2: Read original feature document
        print("Step 2: Reading original feature document...")
        print("-" * 70)
        original_content = read_file_content(str(feature_doc_path))
        print(f"✓ Successfully read {feature_doc_path.name}")
        print()

        # Step 3: Generate updated content
        print("Step 3: Generating updated feature document...")
        print("-" * 70)
        updated_content = generate_updated_feature_doc(
            sparc_content,
            original_content
        )
        print("✓ Successfully generated updated content")
        print(f"  - Document size: {len(updated_content):,} characters")
        print(f"  - Document lines: {len(updated_content.split(chr(10))):,}")
        print()

        # Step 4: Write updated content
        print("Step 4: Writing updated feature document...")
        print("-" * 70)
        write_file_content(str(feature_doc_path), updated_content)
        print()

        # Success message
        print("=" * 70)
        print("✓ Feature document update completed successfully!")
        print("=" * 70)
        print()
        print(f"Updated file: {feature_doc_path}")
        print(f"Backup file: {feature_doc_path}.backup")
        print()

    except Exception as e:
        print()
        print("=" * 70)
        print("✗ Error occurred during document update:")
        print("=" * 70)
        print(f"Error: {str(e)}")
        print()
        raise


if __name__ == "__main__":
    main()
