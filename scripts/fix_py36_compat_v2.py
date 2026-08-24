#!/usr/bin/env python3
"""Comprehensive fix for Python 3.10+ pipe union syntax → Python 3.6 Union/Optional.

Handles ALL forms:
  X | None          → Optional[X]
  X | Y | Z         → Union[X, Y, Z]
  X | Optional[Y]   → Union[X, Optional[Y]]
  -> Union[X, Y]          → -> Union[X, Y]
"""
import re
import sys
from pathlib import Path

ROOT = Path("/aceph/daweihuang/program/binder-harness")

# Files to fix (all .py files in binderloop/ and scripts/)
PY_FILES = sorted(
    list(ROOT.glob("binderloop/**/*.py"))
    + list(ROOT.glob("scripts/**/*.py"))
    + list(ROOT.glob("binderloop/*.py"))
)

# Pattern: in type annotation position (: or ->), match expressions containing |
# We match balanced brackets to get full type expressions.
# The pattern looks for `|` between type-like tokens in annotation context.
#
# Strategy: find all lines with `|` in annotation context, extract the type
# expression, parse the union parts, and rewrite.

def fix_union_in_line(line: str) -> tuple[str, bool]:
    """Fix all X|Y unions in a single line. Returns (fixed_line, changed)."""
    changed = False
    
    # Pattern 1: find `: TYPE_EXPR` or `-> TYPE_EXPR` containing |
    # We need to be careful about string literals and comments.
    
    # Remove string contents temporarily
    parts = []
    in_str = False
    str_char = None
    current = []
    
    i = 0
    while i < len(line):
        ch = line[i]
        if in_str:
            current.append(ch)
            if ch == '\\' and i + 1 < len(line):
                i += 1
                current.append(line[i])
            elif ch == str_char:
                in_str = False
        elif ch in ('"', "'"):
            # Check for triple quotes
            if line[i:i+3] in ('"""', "'''"):
                in_str = True
                str_char = line[i:i+3]
                current.append(line[i:i+3])
                i += 3
                continue
            else:
                in_str = True
                str_char = ch
                current.append(ch)
        elif ch == '#' and not in_str:
            # Comment - rest of line is comment
            current.append(line[i:])
            break
        else:
            current.append(ch)
        i += 1
    
    line_no_str = ''.join(current)
    
    # Now find type annotations containing |
    # Key insight: in annotation context, `|` between alphanumeric/bracket tokens
    # We match: `: Union[WORD_OR_BRACKETS, WORD_OR_BRACKETS]`
    
    # Find all `| ` or ` |` in the line that are NOT inside brackets we already handled
    if '|' not in line_no_str:
        return line, False
    
    # Simple approach: find all `Union[X, Y]` patterns and replace them
    # Match: (token | token | ...) where token = simple_type or bracket_type
    
    # Match a single type token (with optional brackets)
    type_token = r'(?:[A-Za-z_]\w*(?:\s*\[[^\[\]]*\])?)'
    
    # Match: type_token (| type_token)+ 
    # But this needs to handle complex bracket types like Mapping[str, Any]
    # Use a balanced bracket approach
    
    def replace_union(m):
        full = m.group(0)
        # Split by | not inside brackets
        parts = split_union_parts(full)
        if len(parts) < 2:
            return full
        # Clean up whitespace
        parts = [p.strip() for p in parts]
        
        # Check if the last part is just "None"
        if parts[-1] == "None":
            # This is Optional[X | Y | None] or just Optional[base]
            # If only X | None → Optional[X]
            if len(parts) == 2:
                return f"Optional[{parts[0]}]"
            else:
                # X | Y | None → Optional[Union[X, Y]]
                inner = ", ".join(parts[:-1])
                return f"Optional[Union[{inner}]]"
        else:
            # X | Y (no None) → Union[X, Y]
            inner = ", ".join(parts)
            return f"Union[{inner}]"
    
    # Find type expressions with | in annotation context
    # Pattern: find `|` surrounded by type-like tokens
    # We look for sequences of: Union[WORD([...])?, WORD([...])?] | ...
    
    # Simplified: find any occurrence Union[of, that] appears to be between type tokens
    # A "type token" starts with uppercase/lowercase letter and may include brackets
    
    def find_union_expr(start_pos, text):
        """Starting from start_pos, find a complete type expression with |"""
        # Go backward to find start of type expression
        i = start_pos
        depth = 0
        # Expand left
        left_start = start_pos
        while left_start > 0:
            ch = text[left_start - 1]
            if ch == ')':
                depth += 1
                left_start -= 1
            elif ch == '(':
                depth -= 1
                left_start -= 1
            elif ch == ']':
                depth += 1
                left_start -= 1
            elif ch == '[':
                depth -= 1
                left_start -= 1
            elif depth == 0 and ch in ' \t\n\'\"#,;:(':
                break
            elif depth < 0:
                # unmatched bracket, stop
                left_start -= 1
                break
            else:
                left_start -= 1
        
        # Expand right
        right_end = start_pos
        while right_end < len(text):
            ch = text[right_end]
            if ch == '(':
                depth += 1
                right_end += 1
            elif ch == ')':
                depth -= 1
                right_end += 1
            elif ch == '[':
                depth += 1
                right_end += 1
            elif ch == ']':
                depth -= 1
                right_end += 1
            elif depth == 0 and ch in ' \t\n\'\"#,;=)':
                break
            elif depth < 0:
                break
            else:
                right_end += 1
        
        return text[left_start:right_end].strip(), left_start, right_end
    
    # Find all | that are type unions (not in comments or strings)
    result = list(line)
    
    # We'll work on a version without strings to find positions
    # Map positions from stripped version to real version
    
    # Simpler approach: use regex on the whole line
    # Match: identifier[possibly_with_brackets] (| identifier[possibly_with_brackets])+
    
    # Actually, let me use a completely different approach.
    # Find all lines that contain `|` in type annotation context
    # and manually process them.
    
    # Detect if line contains a type annotation with pipe
    # Usually: `: Union[TYPE, TYPE]` or `-> Union[TYPE, TYPE]`
    
    # Quick check: does this line have type annotation context?
    if not (': ' in line or '->' in line) and not line.strip().startswith('->'):
        return line, False
    
    return line, False  # Fallback for now if no simple match


def split_union_parts(expr: str) -> list[str]:
    """Split a type expression like 'str | Optional[Path]' into ['str', 'Optional[Path]']"""
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


def fix_file(filepath: Path) -> bool:
    """Fix a single file. Returns True if changes were made."""
    content = filepath.read_text(encoding="utf-8")
    lines = content.splitlines()
    new_lines = []
    changed = False
    
    for line_num, line in enumerate(lines, 1):
        # Skip comment lines and string-only lines
        stripped = line.strip()
        
        if '|' not in line:
            new_lines.append(line)
            continue
        
        # Check if this line has type annotation context
        is_annotation = False
        if re.search(r':\s*\w.*\|', line):
            is_annotation = True
        if re.search(r'->\s*.*\|', line):
            is_annotation = True
            
        if not is_annotation:
            new_lines.append(line)
            continue
        
        # Now find and fix all | unions in this annotation line
        fixed_line = fix_annotations_in_line(line)
        if fixed_line != line:
            changed = True
        new_lines.append(fixed_line)
    
    if changed:
        filepath.write_text('\n'.join(new_lines) + '\n', encoding="utf-8")
        print(f"FIXED: {filepath.relative_to(ROOT)}")
    return changed


def fix_annotations_in_line(line: str) -> str:
    """Fix all pipe unions in a line that's known to contain type annotations."""
    # Strategy: find all sequences like Union[TYPE, TYPE]n[TYPE, TYPE] and replace them
    
    # First, find all "word_or_bracket | word_or_bracket" patterns
    # A "word_or_bracket" can be: simple (like "str", "int", "Path"), 
    # or bracketed (like "Mapping[str, Any]", "Optional[Path]"),
    # or string-quoted (like '"OpenAICompatibleClient"')
    
    # Regex: match a type expression: word (optional [nested brackets]) (optional | more)
    # Then match: (| word_or_brackets)*
    
    # The core type pattern (catches most types)
    type_part = r'(?:[A-Za-z_]\w*(?:\s*\[(?:[^\[\]]|\[(?:[^\[\]]|\[[^\[\]]*\])*\])*\])?)'
    # But this doesn't handle quoted types like "OpenAICompatibleClient | None"
    
    # Pattern for: type (| type)+
    # We need to NOT match | inside brackets
    union_pattern = re.compile(
        r'(?:'
        r'[A-Za-z_]\w*'  # simple type
        r'(?:\s*\[(?:[^\[\]]|\[[^\[\]]*\])*\])?'  # optional brackets (1 level)
        r')\s*\|\s*'
        r'(?:[A-Za-z_]\w*(?:\s*\[(?:[^\[\]]|\[[^\[\]]*\])*\])?\s*)+'  # second type onwards
    )
    
    # Also handle quoted types: "ClassName"
    quoted_union_pattern = re.compile(
        r'"(?:[^"\\]|\\.)*"\s*\|\s*(?:None|"(?:[^"\\]|\\.)*"|[A-Za-z_]\w*)'
    )
    
    def union_replace(m):
        text = m.group(0)
        parts = split_union_parts(text)
        if len(parts) < 2:
            return text
        
        # Check for None in parts
        none_idx = None
        for i, p in enumerate(parts):
            if p.strip() == 'None':
                none_idx = i
                break
        
        if none_idx is not None and len(parts) == 2:
            # X | None → Optional[X]
            other = parts[1 - none_idx].strip()
            return f"Optional[{other}]"
        elif none_idx is not None:
            # X | Y | None → Optional[Union[X, Y]]
            inner = [p.strip() for i, p in enumerate(parts) if i != none_idx]
            return f"Optional[Union[{', '.join(inner)}]]"
        else:
            # X | Y → Union[X, Y]
            inner = [p.strip() for p in parts]
            return f"Union[{', '.join(inner)}]"
    
    line = union_pattern.sub(union_replace, line)
    line = quoted_union_pattern.sub(union_replace, line)
    
    # Handle remaining simple cases that the regex might have missed
    # e.g., stuff like `float | Optional[str]` where Optional is already processed
    # Try another pass with a broader pattern
    
    # Broader: find any `|` that's not inside brackets
    # Scan character by character
    chars = list(line)
    i = 0
    while i < len(chars):
        ch = chars[i]
        if ch == '|':
            # Check if this | is at depth 0 (not inside brackets)
            depth = 0
            for j in range(i):
                if line[j] in '[(':
                    depth += 1
                elif line[j] in '])':
                    depth -= 1
            if depth == 0:
                # Find the full expression around this |
                # Go left
                left_end = i
                while left_end > 0 and chars[left_end - 1] in ' \t':
                    left_end -= 1
                # Find the left type token start
                left_depth = 0
                left_start = left_end
                while left_start > 0:
                    c = chars[left_start - 1]
                    if c == ')':
                        left_depth += 1
                    elif c == '(':
                        left_depth -= 1
                    elif c == ']':
                        left_depth += 1
                    elif c == '[':
                        left_depth -= 1
                    elif left_depth == 0 and c in ' \t,=:(`"\'#':
                        break
                    elif left_depth < 0:
                        left_start -= 1
                        break
                    left_start -= 1
                    
                # Go right
                right_start = i + 1
                while right_start < len(chars) and chars[right_start] in ' \t':
                    right_start += 1
                right_depth = 0
                right_end = right_start
                while right_end < len(chars):
                    c = chars[right_end]
                    if c == '(':
                        right_depth += 1
                    elif c == ')':
                        right_depth -= 1
                    elif c == '[':
                        right_depth += 1
                    elif c == ']':
                        right_depth -= 1
                    elif right_depth == 0 and c in ' \t,=):`"\'#':
                        break
                    elif right_depth < 0:
                        break
                    right_end += 1
                
                left = line[left_start:left_end].strip()
                right = line[right_start:right_end].strip()
                
                if left and right:
                    # Check if this is a union type annotation (not logical OR in code)
                    # Type annotations start with uppercase/lowercase and might have brackets
                    is_type_left = bool(re.match(r'^[A-Za-z_"\']|^Optional\[|^Union\[', left))
                    is_type_right = bool(re.match(r'^[A-Za-z_"\']|^Optional\[|^Union\[|^None$', right))
                    
                    if is_type_left and is_type_right:
                        if right == 'None':
                            replacement = f"Optional[{left}]"
                        else:
                            replacement = f"Union[{left}, {right}]"
                        line = line[:left_start] + replacement + line[right_end:]
                        # Reset and continue scanning
                        chars = list(line)
                        i = left_start + len(replacement) - 1
        i += 1
    
    return line


# Main execution
fixed_count = 0
for filepath in PY_FILES:
    if fix_file(filepath):
        fixed_count += 1

print(f"\nFixed {fixed_count} files.")

# Now ensure Union is imported in files that need it
needs_union = []
for filepath in PY_FILES:
    if not filepath.exists():
        continue
    content = filepath.read_text(encoding="utf-8")
    if "Union[" in content and "from typing import" in content:, Optional
        lines = content.splitlines()
        for i, line in enumerate(lines):
            if "from typing import" in line:
                if "Union" not in line:
                    lines[i] = line.rstrip() + ", Union"
                    filepath.write_text('\n'.join(lines) + '\n', encoding="utf-8")
                    needs_union.append(str(filepath.relative_to(ROOT)))
                break

print(f"Added Union import to {len(needs_union)} files." if needs_union else "All Union imports OK.")

# Also ensure Optional is imported
needs_optional = []
for filepath in PY_FILES:
    if not filepath.exists():
        continue
    content = filepath.read_text(encoding="utf-8")
    if "Optional[" in content and "from typing import" in content:
        lines = content.splitlines()
        for i, line in enumerate(lines):
            if "from typing import" in line:
                if "Optional" not in line:
                    lines[i] = line.rstrip() + ", Optional"
                    filepath.write_text('\n'.join(lines) + '\n', encoding="utf-8")
                    needs_optional.append(str(filepath.relative_to(ROOT)))
                break

print(f"Added Optional import to {len(needs_optional)} files." if needs_optional else "All Optional imports OK.")
