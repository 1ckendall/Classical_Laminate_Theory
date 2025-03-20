def parse_layup_string(layup: str):
    """
    Parses a composite layup expression (with nested groups) into a tuple of ply angles.

    Supported syntax:
      - Groups: [ ... ]
      - Ply tokens: e.g. "90", "0", "+-45", "-+45"
      - Optional count modifiers: e.g. "90_2" repeats the ply 2 times.
      - Optional symmetry: a group ending with _s is mirrored.
      - Optional multiple symmetry: a group ending with _ns is mirrored n times.
      - Items are separated by commas.

    An assumption is made that you aren't repeating a single ply angle or symmetry more than 9 times (you monster).

    Examples:
      [+-45, 90_2]_s        → [-45, 90, 90, 90, 90, -45]
      [0_2, 90, +-45]_s     → [0, 0, 90, 45, -45, -45, 45, 90, 0, 0]
      [90, -+45]            → [90, -45, 45]
      [[90, 30]_s, +-45]_s  → [90, 30, 30, 90, 45, -45, -45, 45, 90, 30, 30, 90]
      [+-45,90]_2s          → [45, -45, 90, 90, -45, 45, 45, -45, 90, 90, -45, 45]
    """
    s = layup.replace(" ", "")  # remove whitespace

    def parse_number(token: str):
        # token may be like "90", "+-45", or "-+45"
        if token.startswith("+-"):
            try:
                val = int(token[2:])
            except ValueError:
                raise ValueError(f"Invalid ply token: {token}")
            return [val, -val]
        elif token.startswith("-+"):
            try:
                val = int(token[2:])
            except ValueError:
                raise ValueError(f"Invalid ply token: {token}")
            return [-val, val]
        else:
            # Should be a plain number token (with possible + or - sign)
            try:
                val = int(token)
            except ValueError:
                raise ValueError(f"Invalid ply token: {token}")
            return [val]

    def parse_token(i: int):
        """
        Parse a ply token starting at index i.
        Returns (list_of_plies, new_index).
        A token is a sequence of characters (possibly +-45 or -+45 or just number),
        followed optionally by _<count>.
        """
        start = i
        # Read characters until a delimiter is reached (comma or ']' or end of string)
        while i < len(s) and s[i] not in [",", "]"]:
            i += 1
        token = s[start:i]
        # Check if there is a count modifier at the end of the token (e.g. _2)
        if "_" in token:
            token_parts = token.split("_")
            token_val = token_parts[0]
            try:
                count = int(token_parts[1])
            except ValueError:
                raise ValueError(f"Invalid count modifier in token: {token}")
        else:
            token_val = token
            count = 1
        plies = parse_number(token_val)
        return plies * count, i

    def parse_group(i: int):
        """
        Parse a group starting at s[i] (expects s[i]=='[').
        Returns (list_of_plies, new_index).
        Also handles an optional trailing symmetry marker (_ns or _s) after the closing bracket.
        """
        if s[i] != "[":
            raise ValueError("Group must start with '['")
        i += 1  # skip '['
        group_plies = []
        while i < len(s) and s[i] != "]":
            if s[i] == ",":
                i += 1  # skip comma
                continue
            if s[i] == "[":
                # Nested group
                sub_plies, i = parse_group(i)
                group_plies.extend(sub_plies)
            else:
                # Should be a ply token
                token_plies, i = parse_token(i)
                group_plies.extend(token_plies)
        if i >= len(s) or s[i] != "]":
            raise ValueError("Unmatched '[' in expression")
        i += 1  # skip the closing ']'
        # After the group, check for symmetry marker or count modifier.
        is_symmetric = False
        symmetry_count = 0
        count = 1  # Default count multiplier

        if i < len(s) and s[i] == "_":
            i += 1  # Move past "_"
            num_str = ""

            # Check if there's a numeric symmetry count (e.g., _2s)
            while i < len(s) and s[i].isdigit():
                num_str += s[i]
                i += 1

            # If the next character is 's', we have a symmetry modifier
            if i < len(s) and s[i] == "s":
                is_symmetric = True
                i += 1  # Move past 's'
                symmetry_count = (
                    int(num_str) if num_str else 1
                )  # Default to 1 if no number was provided
                result = group_plies
                for _ in range(symmetry_count):
                    result = result + result[::-1]
            # Else we have a repeated group
            else:
                j = i + 1
                num_str = ""
                while j < len(s) and s[j].isdigit():
                    num_str += s[j]
                    j += 1
                if num_str:
                    count = int(num_str)
                    i = j
                result = group_plies * count
        else:
            result = group_plies
        return result, i

    # The top-level expression must be a group.
    result, pos = parse_group(0)
    if pos != len(s):
        raise ValueError(
            f"Extra characters after parsing: Parsing stopped at position {pos}, remaining string: '{s[pos:]}"
        )
    return tuple(result)
