"""Load and render versioned prompt templates from config/prompt_templates.yaml."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_TEMPLATES_PATH = Path(__file__).parent.parent.parent / "config" / "prompt_templates.yaml"


@lru_cache(maxsize=1)
def _load_raw() -> dict[str, Any]:
    with open(_TEMPLATES_PATH) as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


def get_template(name: str, version: str = "v1") -> dict[str, str]:
    """Return {"system": ..., "user": ...} for a named template version."""
    raw = _load_raw()
    try:
        tmpl: dict[str, str] = raw["templates"][name][version]
    except KeyError as exc:
        raise ValueError(
            f"Template {name!r} version {version!r} not found in {_TEMPLATES_PATH}"
        ) from exc
    return tmpl


def render_user_message(name: str, version: str = "v1", **kwargs: str) -> str:
    """Render the user turn of a template with the given variables."""
    tmpl = get_template(name, version)
    return tmpl["user"].format(**kwargs)


def list_templates() -> list[str]:
    """Return all template names defined in the YAML."""
    raw = _load_raw()
    return list(raw.get("templates", {}).keys())
