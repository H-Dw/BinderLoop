#!/usr/bin/env python3
"""Fix missing Union/List/Dict/Tuple/Set imports across all Python files."""
from __future__ import print_function
from pathlib import Path

ROOT = Path("/aceph/daweihuang/program/binder-harness")

# Scan for files using Union/Lists/Dicts without imports
TYPES = ['Union', 'List', 'Dict', 'Tuple', 'Set', 'Optional']

def fix_missing_imports(filepath):
    content = filepath.read_text(encoding="utf-8")
    lines = content.split('\n')
    
    # Find all types used in this file
    used_types = []
    for t in TYPES:
        if t + '[' in content:
            used_types.append(t)
    
    if not used_types:
        return False
    
    # Find the typing import line
    import_idx = None
    for i, line in enumerate(lines):
        if 'from typing import' in line:
            import_idx = i
            break
    
    if import_idx is None:
        return False
    
    # Check which types are missing
    import_line = lines[import_idx]
    missing = [t for t in used_types if t not in import_line]
    
    if not missing:
        return False
    
    # Add missing types
    new_import = import_line.rstrip()
    for t in missing:
        new_import += ', ' + t
    
    lines[import_idx] = new_import
    filepath.write_text('\n'.join(lines), encoding="utf-8")
    return True

count = 0
for fp in sorted(list(ROOT.glob("binderloop/**/*.py")) + list(ROOT.glob("scripts/**/*.py")) + list(ROOT.glob("binderloop/*.py"))):
    if not fp.exists() or 'fix_py36' in fp.name or 'fix_py39' in fp.name:
        continue
    if fix_missing_imports(fp):
        print("FIXED: {}".format(fp.relative_to(ROOT)))
        count += 1

print("\nFixed {} files.".format(count))
