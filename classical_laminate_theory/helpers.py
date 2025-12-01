import numpy as np

from classical_laminate_theory.failuremodels import FailureModel
from classical_laminate_theory.structures import Lamina, Laminate


def parse_layup_string(layup: str):
    """
    Parses a composite layup string into a flat tuple of ply angles.

    Improvements over original:
    1. Supports '/' as a delimiter (e.g., "[0/90]_s").
    2. Corrects symmetry logic: "_2s" now means "Repeat 2x then Mirror" (Standard)
       instead of "Mirror then Repeat 2x".
    3. Better error reporting.
    """
    s = layup.replace(" ", "")  # remove whitespace

    def parse_number(token: str):
        # Handles "90", "+-45", "-+45"
        if token.startswith("+-"):
            try:
                val = float(token[2:])
            except ValueError:
                raise ValueError(f"Invalid ply angle: '{token}'")
            return [val, -val]
        elif token.startswith("-+"):
            try:
                val = float(token[2:])
            except ValueError:
                raise ValueError(f"Invalid ply angle: '{token}'")
            return [-val, val]
        else:
            try:
                val = float(token)
            except ValueError:
                raise ValueError(f"Invalid ply angle: '{token}'")
            return [val]

    def parse_token(i: int):
        """Parse a single token (e.g. '90' or '45_2')."""
        start = i
        # Stop at delimiters or end of group
        while i < len(s) and s[i] not in [",", "/", "]"]:
            i += 1

        token = s[start:i]
        if not token:
            return [], i

            # Handle ply-level repetition (e.g., 90_2)
        if "_" in token:
            parts = token.split("_")
            if len(parts) != 2:
                raise ValueError(f"Invalid token format at pos {start}: '{token}'")
            base_token = parts[0]
            try:
                count = int(parts[1])
            except ValueError:
                raise ValueError(f"Invalid repetition count at pos {start}: '{token}'")
        else:
            base_token = token
            count = 1

        plies = parse_number(base_token)
        return plies * count, i

    def parse_group(i: int):
        """Recursive parser for groups like [0, 90]_s."""
        if s[i] != "[":
            raise ValueError(f"Expected '[' at position {i}")
        i += 1  # skip '['

        group_plies = []
        while i < len(s) and s[i] != "]":
            # Skip delimiters
            if s[i] in [",", "/"]:
                i += 1
                continue

            # Recurse for nested groups
            if s[i] == "[":
                sub_plies, i = parse_group(i)
                group_plies.extend(sub_plies)
            else:
                token_plies, i = parse_token(i)
                group_plies.extend(token_plies)

        if i >= len(s) or s[i] != "]":
            raise ValueError("Unmatched '[' - missing closing bracket.")
        i += 1  # skip ']'

        # Handle Group Modifiers (e.g. _2, _s, _2s)
        count = 1
        is_symmetric = False

        if i < len(s) and s[i] == "_":
            i += 1  # skip '_'
            num_str = ""
            while i < len(s) and s[i].isdigit():
                num_str += s[i]
                i += 1

            # Determine count
            count = int(num_str) if num_str else 1

            # Check for symmetry
            if i < len(s) and s[i] == "s":
                is_symmetric = True
                i += 1

        # Apply Logic: Repeat first, then Mirror (Standard Convention)
        result = group_plies * count
        if is_symmetric:
            result = result + result[::-1]

        return result, i

    # Start parsing
    if not s.startswith("["):
        # Allow bare strings like "0, 90" by wrapping them implicitly?
        # For now, enforce brackets to match strict notation.
        raise ValueError("Layup string must start with '['")

    result, pos = parse_group(0)

    if pos != len(s):
        raise ValueError(f"Parsing ended early at char {pos}. Remaining: '{s[pos:]}'")

    return tuple(result)
