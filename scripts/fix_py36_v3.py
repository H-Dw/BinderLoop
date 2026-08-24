#!/usr/bin/env python3
"""Fix Python 3.10+ pipe union syntax -> Python 3.6 Union/Optional.
Usage: python3 fix_py36_v3.py
"""
from __future__ import print_function
import re
import sys
from pathlib import Path

ROOT = Path("/aceph/daweihuang/program/binder-harness")

# All .py files to process
PY_FILES = sorted(
    list(ROOT.glob("binderloop/**/*.py"))
    + list(ROOT.glob("scripts/**/*.py"))
    + list(ROOT.glob("binderloop/*.py"))
)

def split_union(expr):
    """Split 'A | B | C' at toplevel |, respecting brackets."""
    parts = []
    current = []
    depth = 0
    for ch in expr:
        if ch == '|' and depth == 0:
            parts.append(''.join(current).strip())
            current = []
        else:
            if ch in '[(':
                depth += 1
            elif ch in '])':
                depth -= 1
            current.append(ch)
    if current:
        parts.append(''.join(current).strip())
    return [p for p in parts if p]

def to_union_or_optional(parts):
    """Convert ['X','Y'] or ['X','None'] to Union/Optional string."""
    parts = [p.strip() for p in parts]
    # Check for None
    none_parts = [p for p in parts if p == 'None']
    other_parts = [p for p in parts if p != 'None']
    if none_parts and not other_parts:
        return 'None'
    if none_parts:
        if len(other_parts) == 1:
            return 'Optional[{}]'.format(other_parts[0])
        else:
            return 'Optional[Union[{}]]'.format(', '.join(other_parts))
    if len(parts) == 1:
        return parts[0]
    return 'Union[{}]'.format(', '.join(parts))

def fix_line(line):
    """Fix all X|Y unions in a line. Returns (fixed_line, changed)."""
    # Quick check
    if '|' not in line:
        return line, False
    
    # Only process lines with type annotation context
    stripped = line.strip()
    if not (re.search(r':\s*\w', line) or re.search(r'->\s*', line)):
        return line, False
    
    # Find all | operators at bracket depth 0 (not inside [] or ())
    # We need to find contiguous sequences of: Union[TYPE, TYPE]n[TYPE, TYPE]
    # where TYPE = word (maybe with [brackets])
    
    # Scan for | at depth 0, extract the full expression around it
    result = line
    changed = False
    
    # Mark bracket depths
    n = len(line)
    depth = [0] * (n + 1)
    for i, ch in enumerate(line):
        d = depth[i]
        if ch in '[(':
            d += 1
        elif ch in '])':
            d = max(0, d - 1)
        depth[i+1] = d
    
    # Find all | at depth 0
    pipe_positions = []
    for i, ch in enumerate(line):
        if ch == '|' and depth[i] == 0 and depth[i+1] == 0:
            pipe_positions.append(i)
    
    if not pipe_positions:
        return line, False
    
    # Group consecutive pipe expressions (they form a union chain)
    # For each pipe, find the full left and right expressions
    replacements = []  # list of (start, end, replacement)
    
    for pi in pipe_positions:
        # Find left expression boundary
        left_end = pi
        while left_end > 0 and line[left_end - 1] in ' \t':
            left_end -= 1
        
        left_start = left_end
        ld = 0
        while left_start > 0:
            c = line[left_start - 1]
            if c == ')': ld += 1
            elif c == '(': ld -= 1
            elif c == ']': ld += 1
            elif c == '[': ld -= 1
            elif ld == 0 and c in ' \t,=:(`"\'#\n':
                break
            elif ld < 0:
                left_start -= 1
                break
            left_start -= 1
        
        # Find right expression boundary
        right_start = pi + 1
        while right_start < n and line[right_start] in ' \t':
            right_start += 1
        
        right_end = right_start
        rd = 0
        while right_end < n:
            c = line[right_end]
            if c == '(': rd += 1
            elif c == ')': rd -= 1
            elif c == '[': rd += 1
            elif c == ']': rd -= 1
            elif rd == 0 and c in ' \t,=):`"\'#\n':
                break
            elif rd < 0:
                break
            right_end += 1
        
        left = line[left_start:left_end].strip()
        right = line[right_start:right_end].strip()
        
        if not left or not right:
            continue
        
        # Check if both sides look like types
        left_is_type = bool(re.match(r'^[A-Za-z_"\']|^Optional\[|^Union\[|^Callable\[', left))
        right_is_type = bool(re.match(r'^[A-Za-z_"\']|^Optional\[|^Union\[|^None$|^Callable\[', right))
        
        if left_is_type and right_is_type:
            full = line[left_start:right_end].strip()
            parts = split_union(full)
            if len(parts) >= 2:
                replacement = to_union_or_optional(parts)
                replacements.append((left_start, right_end, replacement))
    
    if not replacements:
        return line, False
    
    # Apply replacements from right to left to preserve positions
    replacements.sort(key=lambda x: -x[0])
    for start, end, repl in replacements:
        result = result[:start] + repl + result[end:]
        changed = True
    
    return result, changed


def process_file(filepath):
    """Process a single file, return True if changed."""
    content = filepath.read_text(encoding="utf-8")
    lines = content.split('\n')
    new_lines = []
    file_changed = False
    
    for line in lines:
        fixed, changed = fix_line(line)
        new_lines.append(fixed)
        if changed:
            file_changed = True
    
    if file_changed:
        filepath.write_text('\n'.join(new_lines), encoding="utf-8")
    
    return file_changed


def ensure_imports(filepath):
    """Ensure Optional and Union are imported if used."""
    content = filepath.read_text(encoding="utf-8")
    uses_union = 'Union[' in content
    uses_optional = 'Optional[' in content
    
    if not (uses_union or uses_optional):
        return False
    
    lines = content.split('\n')
    import_line_idx = None
    for i, line in enumerate(lines):
        if 'from typing import' in line:, Union, Optional
            import_line_idx = i
            break
    
    if import_line_idx is None:
        return False
    
    line = lines[import_line_idx]
    needed = []
    if uses_union and 'Union' not in line:
        needed.append('Union')
    if uses_optional and 'Optional' not in line:
        needed.append('Optional')
    
    if needed:
        lines[import_line_idx] = line.rstrip() + ', ' + ', '.join(needed)
        filepath.write_text('\n'.join(lines), encoding="utf-8")
        return True
    return False


# Main
fixed_files = 0
for fp in PY_FILES:
    if not fp.exists():
        continue
    if process_file(fp):
        print("FIXED: {}".format(fp.relative_to(ROOT)))
        fixed_files += 1

print("\nFixed {} files for pipe unions.".format(fixed_files))

# Ensure imports
import_fixed = 0
for fp in PY_FILES:
    if not fp.exists():
        continue
    if ensure_imports(fp):
        import_fixed += 1

print("Added imports to {} files.".format(import_fixed))
