from dash import html
from modularnet.modularui.components import Widget


class Panel(Widget):
    """A single dashboard card: header with title + collapse/maximize
    controls, and a scrollable content body."""

    def __init__(self, panel_id: str, title: str, content: Widget):
        self.panel_id = panel_id
        self.title = title
        self.content = content

    def render(self, maximized: bool = False, grid_area: str | None = None):
        # No explicit height here on purpose: CSS Grid's default
        # align-items: stretch makes short cards match their row's
        # tallest sibling, while cards taller than that just grow the
        # row (and the page) instead of clipping. Same trick for width.
        style = {
            "display": "flex", "flexDirection": "column",
            "width": "100%", "minWidth": 0,
            "backgroundColor": "rgba(255,255,255,0.03)",
            "border": "1px solid rgba(255,255,255,0.08)",
            "borderRadius": "10px",
            "boxShadow": "0 2px 6px rgba(0,0,0,0.25)",
        }
        if grid_area:
            style["gridArea"] = grid_area

        if maximized:
            # The maximized overlay is the one case that DOES need a
            # hard viewport-relative size and its own internal scroll,
            # since it floats above everything else.
            style.update({
                "position": "fixed", "top": "20px", "left": "20px",
                "width": "calc(100vw - 40px)", "height": "calc(100vh - 40px)",
                "zIndex": 9999, "backgroundColor": "#111",
                "boxShadow": "0 10px 30px rgba(0,0,0,0.8)",
                "overflow": "hidden",
            })

        return html.Div(
            id=self.panel_id,
            className="dashboard-card",
            style=style,
            children=[
                html.Div(
                    className="card-header",
                    style={
                        "display": "flex", "justifyContent": "space-between", "alignItems": "center",
                        "padding": "10px 14px",
                        "borderBottom": "1px solid rgba(255,255,255,0.08)",
                        "flexShrink": 0,
                    },
                    children=[
                        html.Span(self.title, style={
                            "fontSize": "0.78em", "fontWeight": 600, "letterSpacing": "0.03em",
                            "textTransform": "uppercase", "opacity": 0.65,
                            "whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis",
                        }),
                        html.Div(
                            style={"display": "flex", "gap": "2px"},
                            children=[
                                html.Button("—", id={"type": "btn-collapse", "index": self.panel_id}, className="card-btn"),
                                html.Button("□", id={"type": "btn-maximize", "index": self.panel_id}, className="card-btn"),
                            ]
                        )
                    ]
                ),
                html.Div(
                    self.content.render(),
                    className="card-body",
                    style=(
                        {
                            "padding": "12px", "flex": "1", "minHeight": 0,
                            "overflowY": "auto", "overflowX": "hidden", "wordWrap": "break-word",
                        }
                        if maximized else
                        {
                            "padding": "12px", "flex": "1 1 auto",
                            "wordWrap": "break-word",
                        }
                    )
                )
            ]
        )


class DashboardLayout(Widget):
    """A clean, static CSS-Grid dashboard: a row of stat cards on top,
    a sidebar card and a main content card below. Fixed, sensible
    proportions — no dragging, no resizing. Reflows to a single column
    on narrow screens via assets/dashboard.css."""

    TOP_AREAS = ["top-a", "top-b", "top-c"]

    def __init__(self, top_panels: list[Panel], left_panel: Panel, main_panel: Panel):
        self.top_panels = top_panels
        self.left_panel = left_panel
        self.main_panel = main_panel
        self.all_panels = {p.panel_id: p for p in [*top_panels, left_panel, main_panel]}

    def render(self, state: dict = None):
        state = state or {}

        def is_visible(pid):
            return not state.get(pid, {}).get("collapsed", False)

        def is_maxed(pid):
            return state.get(pid, {}).get("maximized", False)

        children = []
        for panel, area in zip(self.top_panels, self.TOP_AREAS):
            if is_visible(panel.panel_id):
                children.append(panel.render(maximized=is_maxed(panel.panel_id), grid_area=area))

        if is_visible(self.left_panel.panel_id):
            children.append(self.left_panel.render(maximized=is_maxed(self.left_panel.panel_id), grid_area="sidebar"))

        if is_visible(self.main_panel.panel_id):
            children.append(self.main_panel.render(maximized=is_maxed(self.main_panel.panel_id), grid_area="main"))

        return html.Div(
            children,
            className="dashboard-grid",
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr 1fr",
                "gridTemplateRows": "auto auto",
                "gridTemplateAreas": '"top-a top-b top-c" "sidebar main main"',
                "gap": "12px",
                "width": "100%",
            }
        )