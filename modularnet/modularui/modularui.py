

import time

from dash import Dash, html
import threading

class ModularUI():
    def __init__(self, host: str = "127.0.0.1", port: int = 8050, debug: bool = False, reloader: bool = False):
        self.host = host
        self.port = port
        self.debug = debug
        self.reloader = reloader

        
        self.app = Dash(__name__)
        
        self.server_thread = None
        self.update_layout()


    def update_layout(self, all_vars: dict = None, logs: str = None):
        self.app.layout = html.Div([
            html.H1("ModularNet UI"),
            html.Div(id="modularnet-content"),
            html.Pre([str(all_vars) if all_vars else "No variables to display"], id="modularnet-vars", style={"whiteSpace": "pre-wrap", "wordBreak": "break-all"}),
            html.Div([logs] if logs else ["No logs to display"], id="modularnet-logs", style={"whiteSpace": "pre-wrap", "wordBreak": "break-all"}),
        ])
    
    def run(self):
        self.server_thread = threading.Thread(target=self._run)
        self.server_thread.start()


    def _run(self):
        self.app.run(host=self.host, port=self.port, debug=self.debug, use_reloader=self.reloader)  # use_reloader=False to prevent double initialization

    def stop(self):
        if self.server_thread and self.server_thread.is_alive():
            # Note: Dash does not provide a built-in way to stop the server programmatically.
            # This is a workaround to stop the server by terminating the thread.
            # In production, consider using a more robust solution for stopping the server.
            import os
            import signal
            os.kill(os.getpid(), signal.SIGINT)  # Send SIGINT to terminate the process
        pass