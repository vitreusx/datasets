import re


def parse_size(size: str) -> int:
    """Parse a human-readable size string into a number of bytes.

    Single-letter suffixes (5G, 10M) use binary (1024) base.
    B-suffixes (14MB, 3KB) use decimal (1000) base.
    IEC suffixes (8GiB, 2MiB) use binary (1024) base.
    """
    m = re.fullmatch(
        r"(\d+(?:\.\d+)?)\s*([KMGTP]?)(i?)(B?)",
        size.strip(),
        re.IGNORECASE,
    )
    if m is None:
        msg = f"Invalid size string: {size!r}"
        raise ValueError(msg)

    value_str, prefix, iec, has_b = m.groups()
    value = float(value_str)
    prefix = prefix.upper()

    if not prefix:
        return int(value)

    exp = {"K": 1, "M": 2, "G": 3, "T": 4, "P": 5}[prefix]
    base = 1000 if (has_b and not iec) else 1024
    return int(value * base**exp)
