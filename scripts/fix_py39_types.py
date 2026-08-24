#!/usr/bin/env python3
"""Fix Python 3.9+ builtin generic types (list[X], dict[X,Y], tuple[X,Y], set[X])
to Python 3.6 compatible typing.List, typing.Dict, typing.Tuple, typing.Set."""
from __future__ import print_function
import re
from pathlib import Path

ROOT = Path("/aceph/daweihuang/program/binder-harness")

PY_FILES = sorted(
    list(ROOT.glob("binderloop/**/*.py"))
    + list(ROOT.glob("scripts/**/*.py"))
    + list(ROOT.glob("binderloop/*.py"))
)

SUBSTITUTIONS = [
    # List[X] -> List[X] (but not already List[X] or inside strings)
    (r'\blist\[', 'List['),
    # Dict[X, Y] -> Dict[X, Y]
    (r'\bdict\[', 'Dict['),
    # Tuple[X, Y] -> Tuple[X, Y] (but be careful with Tuple[int, int] patterns)
    (r'\btuple\[', 'Tuple['),
    # Set[X] -> Set[X]
    (r'\bset\[', 'Set['),
    # FrozenSet[X] -> FrozenSet[X]
    (r'\bfrozenset\[', 'FrozenSet['),
]

def fix_file(filepath):
    content = filepath.read_text(encoding="utf-8")
    original = content
    
    # Apply all substitutions, but only to lines that look like type annotations
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        # Only fix lines that have type annotation context (: or ->)
        if ': ' in line or '->' in line or line.strip().startswith('def ') or line.strip().startswith('class '):
            fixed = line
            for pattern, replacement in SUBSTITUTIONS:
                fixed = re.sub(pattern, replacement, fixed)
            new_lines.append(fixed)
        else:
            new_lines.append(line)
    
    content = '\n'.join(new_lines)
    
    if content != original:
        filepath.write_text(content, encoding="utf-8")
        return True
    return False


def ensure_imports(filepath):
    """Ensure List, Dict, Tuple, Set are imported if used."""
    content = filepath.read_text(encoding="utf-8")
    needs = []
    for name in ('List', 'Dict', 'Tuple', 'Set', 'FrozenSet'):
        if re.search(r'\b' + name + r'\[', content):
            if name == 'FrozenSet':
                # FrozenSet needs special import from typing
                pass
            needs.append(name)
    
    if not needs:
        return False
    
    lines = content.split('\n')
    import_idx = None
    for i, line in enumerate(lines):
        if 'from typing import' in line:, List, Dict, Tuple, Set, FrozenSet
            import_idx = i
            break
    
    if import_idx is None:
        return False
    
    # Check what's already imported
    existing = lines[import_idx]
    for name in needs:
        if name not in existing:
            # Add to import
            lines[import_idx] = existing.rstrip() + ', ' + name
            existing = lines[import_idx]
    
    filepath.write_text('\n'.join(lines), encoding="utf-8")
    return True


fixed_count = 0
for fp in PY_FILES:
    if not fp.exists() or 'fix_py36' in fp.name:
        continue
    if fix_file(fp):
        print("FIXED: {}".format(fp.relative_to(ROOT)))
        fixed_count += 1

print("\nFixed {} files for builtin generic types.".format(fixed_count))

imp_count = 0
for fp in PY_FILES:
    if not fp.exists() or 'fix_py36' in fp.name:
        continue
    if ensure_imports(fp):
        imp_count += 1

print("Updated imports in {} files.".format(imp_count))
