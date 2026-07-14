

from dash import Dash, html

class ModularUI():
    def __init__(self, host: str = "127.0.0.1", port: int = 8050, debug: bool = True, reloader: bool = False):
        self.host = host
        self.port = port
        self.debug = debug
        self.reloader = reloader
        self.app = Dash(__name__)
        
        self.update_layout()


    def update_layout(self, all_vars: dict = None, logs: str = None):
        self.app.layout = html.Div([
            html.H1("ModularNet UI"),
            html.Div(id="modularnet-content"),
            html.Pre([str(all_vars) if all_vars else "No variables to display"], id="modularnet-vars", style={"whiteSpace": "pre-wrap", "wordBreak": "break-all"}),
            html.Div([logs] if logs else ["No logs to display"], id="modularnet-logs", style={"whiteSpace": "pre-wrap", "wordBreak": "break-all"}),
        ])
    
    def run(self):
        self.app.run(host=self.host, port=self.port, debug=self.debug, use_reloader=self.reloader)  # use_reloader=False to prevent double initialization

    def stop(self):
        # Stopping the Dash server programmatically is not straightforward.
        pass