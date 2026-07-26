from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
from dash import dcc, html


class Widget(ABC):
    """Base class for a reusable Dash component renderer.

    Instances hold only *static* config (ids, styling, defaults). Live data
    is always passed into `render(...)` — never cached on `self` — so one
    widget instance can be safely reused across every callback tick without
    going stale or racing with a concurrent update.
    """

    @abstractmethod
    def render(self, *args, **kwargs):
        ...

    def __call__(self, *args, **kwargs):
        return self.render(*args, **kwargs)


class Title(Widget):
    def __init__(self, text: str):
        self.text = text

    def render(self):
        return html.H1(self.text)


class ExpandableSection(Widget):
    def __init__(self, expanded: bool = False):
        self.expanded = expanded

    def render(self, title: str, content: str):
        return html.Details(
            [html.Summary(title), html.Pre(content)],
            open=self.expanded,
        )


class VarsDisplay(Widget):
    def __init__(self, css_id: str = "modularnet-vars", css_style: Optional[dict] = None):
        self.css_id = css_id
        self.css_style = css_style or {"whiteSpace": "pre-wrap", "wordBreak": "break-all"}
        self._section = ExpandableSection()

    def render(self, all_vars: Optional[Dict[str, Any]] = None):
        all_vars = all_vars or {}
        if not all_vars:
            return html.Div("No variables to display", id=self.css_id, style=self.css_style)

        sections = []
        for name, value in sorted(all_vars.items()):
            if isinstance(value, dict):
                content = "\n".join(f"{k}: {v}" for k, v in sorted(value.items()))
            else:
                content = str(value)
            sections.append(self._section.render(title=name, content=content))

        return html.Div(sections, id=self.css_id, style=self.css_style)


class ModelDisplay(Widget):
    def __init__(self, css_id: str = "modularnet-model"):
        self.css_id = css_id

    def render(self, model_summary: Optional[str] = None):
        return html.Div(html.Pre(model_summary or "No model loaded"), id=self.css_id)


class ConfigDisplay(Widget):
    def __init__(self, css_id: str = "modularnet-config"):
        self.css_id = css_id
        self._section = ExpandableSection(expanded=True)

    def render(self, config: Optional[dict] = None):
        config = config or {}
        content = "\n".join(f"{k}: {v}" for k, v in sorted(config.items()))
        return html.Div(self._section.render(title="Config", content=content), id=self.css_id)