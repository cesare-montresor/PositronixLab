import dash_bootstrap_components as dbc

# ---------------------------------------------------------
# App Themes & Styling
# ---------------------------------------------------------
# Change the theme here ONCE and it updates the entire UI and charts.
# Options: darkly, cyborg, slate, solar, superhero, vapor
THEME_NAME = "cyborg" 

# Resolves the dbc.themes constant dynamically (e.g., dbc.themes.DARKLY)
BOOTSTRAP_THEME = getattr(dbc.themes, THEME_NAME.upper())


# Styles plain dcc components to match the dbc theme
DBC_HELPER_CSS = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.2/dbc.min.css"

# Plotly Figure Templates
PLOTLY_TEMPLATE = THEME_NAME
PLOTLY_PAPER_BGCOLOR = "rgba(0,0,0,0)"
PLOTLY_PLOT_BGCOLOR = "rgba(0,0,0,0)"

# ---------------------------------------------------------
# UI Constants
# ---------------------------------------------------------
# Fixed key order for the four space-loss components.


# Metric family (the "name" of the metrics message) shown as the overall row + per-layer rows.
# The labels inside it are NOT listed here: they come from the dict the loss function returns.
SPACELOSS_FAMILY = "spaceloss"
# Name of the aggregated (mean over layers) trace on the overall charts.
GLOBAL_TRACE_NAME = "mean (all layers)"
# Deprecated, unused by the UI any more (labels are dynamic).
PLOTS_SPACELOSS_KEYS = ["loss", "loss_on", "loss_off", "dict_center"]
PLOTS_ROW_LABEL_STYLE = {"fontSize": "0.75em", "opacity": 0.6, "marginBottom": "2px"}
