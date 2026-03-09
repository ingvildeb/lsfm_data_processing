def get_underscore_token(name: str, index: int, field_name: str = "token") -> str:
    """
    Return underscore-split token at `index` from `name` with clear bounds checks.
    """
    parts = name.split("_")
    if index < 0:
        raise RuntimeError(f"{field_name} index must be >= 0. Got: {index}")
    if index >= len(parts):
        raise RuntimeError(
            f"Cannot extract {field_name} from:\n{name}\n"
            f"Expected at least {index + 1} underscore-separated parts."
        )
    return parts[index]


def get_underscore_int(name: str, index: int, field_name: str = "integer token") -> int:
    """
    Return integer underscore-split token at `index` from `name`.
    """
    token = get_underscore_token(name, index, field_name)
    try:
        return int(token)
    except ValueError as e:
        raise RuntimeError(
            f"Expected integer {field_name} at underscore index {index} in:\n{name}\nGot: {token}"
        ) from e
