"""
Enhanced model downloader component with progress tracking and cancellation.
"""
import os
import time
import threading
import queue
from pathlib import Path
from typing import Dict, Optional, List
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback, no_update, ctx
import dash
from rich.console import Console
import subprocess
import signal

console = Console()


class DownloadManager:
    """Manages model downloads with progress tracking and cancellation."""
    
    def __init__(self):
        self.active_downloads = {}  # {model_id: download_info}
        self.download_queue = queue.Queue()
        self.stop_events = {}  # {model_id: threading.Event}
        
    def start_download(self, model_id: str) -> bool:
        """Start downloading a model."""
        if model_id in self.active_downloads:
            return False  # Already downloading
            
        # Create stop event for this download
        stop_event = threading.Event()
        self.stop_events[model_id] = stop_event
        
        # Initialize download info
        self.active_downloads[model_id] = {
            "status": "starting",
            "progress": 0,
            "speed": "0 MB/s",
            "eta": "calculating...",
            "size": "unknown",
            "downloaded": "0 MB"
        }
        
        # Start download thread
        download_thread = threading.Thread(
            target=self._download_worker,
            args=(model_id, stop_event),
            daemon=True
        )
        download_thread.start()
        
        return True
    
    def stop_download(self, model_id: str) -> bool:
        """Stop downloading a model."""
        if model_id in self.stop_events:
            self.stop_events[model_id].set()
            if model_id in self.active_downloads:
                self.active_downloads[model_id]["status"] = "stopping"
            return True
        return False
    
    def get_download_status(self, model_id: str) -> Optional[Dict]:
        """Get current download status."""
        return self.active_downloads.get(model_id)
    
    def _download_worker(self, model_id: str, stop_event: threading.Event):
        """Worker thread for downloading models."""
        try:
            # Import here to avoid circular imports
            from config import config
            
            model_config = config.get_model_config(model_id)
            if not model_config:
                self.active_downloads[model_id] = {
                    "status": "error",
                    "progress": 0,
                    "error": f"Model {model_id} not found in config"
                }
                return
            
            # Update status
            self.active_downloads[model_id]["status"] = "downloading"
            
            # Prepare download command
            script_path = Path(__file__).parent.parent / "download_models.py"
            cmd = [
                "python", str(script_path),
                "--model", model_id,
                "--progress"
            ]
            
            # Start download process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Monitor progress
            while process.poll() is None:
                if stop_event.is_set():
                    # Kill the download process
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    
                    self.active_downloads[model_id]["status"] = "cancelled"
                    return
                
                # Read progress from stdout
                try:
                    line = process.stdout.readline()
                    if line:
                        self._parse_progress(model_id, line.strip())
                except:
                    pass
                
                time.sleep(0.5)
            
            # Check final result
            return_code = process.returncode
            if return_code == 0:
                self.active_downloads[model_id]["status"] = "completed"
                self.active_downloads[model_id]["progress"] = 100
            else:
                stderr_output = process.stderr.read()
                self.active_downloads[model_id]["status"] = "error"
                self.active_downloads[model_id]["error"] = stderr_output
                
        except Exception as e:
            self.active_downloads[model_id]["status"] = "error"
            self.active_downloads[model_id]["error"] = str(e)
        finally:
            # Clean up
            if model_id in self.stop_events:
                del self.stop_events[model_id]
    
    def _parse_progress(self, model_id: str, line: str):
        """Parse progress from download output."""
        try:
            # Look for progress patterns in the output
            if "%" in line and "MB/s" in line:
                # Extract progress percentage
                parts = line.split()
                for i, part in enumerate(parts):
                    if "%" in part:
                        progress = float(part.replace("%", ""))
                        self.active_downloads[model_id]["progress"] = progress
                        break
                
                # Extract speed
                for part in parts:
                    if "MB/s" in part or "KB/s" in part:
                        self.active_downloads[model_id]["speed"] = part
                        break
            
            elif "Downloading" in line:
                self.active_downloads[model_id]["status"] = "downloading"
            elif "Loading" in line:
                self.active_downloads[model_id]["status"] = "loading"
            elif "Saving" in line:
                self.active_downloads[model_id]["status"] = "saving"
                
        except:
            pass


# Global download manager instance
download_manager = DownloadManager()


def create_download_interface():
    """Create the download interface layout."""
    return dbc.Card([
        dbc.CardHeader([
            html.H5("Model Downloads", className="mb-0"),
            dbc.Button(
                [html.I(className="fas fa-sync-alt me-2"), "Refresh Status"],
                id="refresh-downloads-btn",
                color="outline-primary",
                size="sm",
                className="float-end"
            )
        ]),
        dbc.CardBody([
            # Chart QA Models Section
            html.Div([
                html.H6("🔥 Chart QA Fine-tuned Models (Recommended)", className="text-primary mb-3"),
                html.Div(id="chart-qa-models-list"),
            ], className="mb-4"),
            
            # Base Models Section  
            html.Div([
                html.H6("📊 Base Models", className="text-muted mb-3"),
                html.Div(id="base-models-list"),
            ])
        ])
    ])


def create_model_download_card(model_id: str, model_name: str, is_chart_qa: bool = False):
    """Create a download card for a single model."""
    from config import config
    
    model_config = config.get_model_config(model_id)
    if not model_config:
        return html.Div()
    
    # Check if model is downloaded
    is_downloaded = model_config.local_dir.exists() and any(model_config.local_dir.glob("*.safetensors"))
    
    # Get download status
    download_status = download_manager.get_download_status(model_id)
    is_downloading = download_status is not None and download_status["status"] in ["starting", "downloading", "loading", "saving"]
    
    # Calculate estimated size (rough estimates)
    size_estimates = {
        "llava_v1_5_7b": "14 GB",
        "unichart": "2 GB"
    }
    
    estimated_size = size_estimates.get(model_id, "Unknown")
    
    # Create status badge
    if is_downloaded:
        status_badge = dbc.Badge("Downloaded", color="success", className="me-2")
    elif is_downloading:
        status_badge = dbc.Badge("Downloading", color="info", className="me-2")
    else:
        status_badge = dbc.Badge("Not Downloaded", color="secondary", className="me-2")
    
    # Create action button
    if is_downloading:
        action_btn = dbc.Button(
            [html.I(className="fas fa-stop me-2"), "Stop"],
            id={"type": "stop-btn", "index": model_id},
            color="danger",
            size="sm",
            disabled=download_status and download_status["status"] == "stopping"
        )
    elif is_downloaded:
        action_btn = dbc.Button(
            [html.I(className="fas fa-check me-2"), "Ready"],
            color="success",
            size="sm",
            disabled=True
        )
    else:
        action_btn = dbc.Button(
            [html.I(className="fas fa-download me-2"), "Download"],
            id={"type": "download-btn", "index": model_id},
            color="primary" if is_chart_qa else "outline-primary",
            size="sm"
        )
    
    # Create progress bar for downloading models
    progress_section = html.Div()
    if is_downloading and download_status:
        progress = download_status.get("progress", 0)
        status_text = download_status.get("status", "downloading")
        speed = download_status.get("speed", "")
        
        progress_section = html.Div([
            dbc.Progress(
                value=progress,
                striped=True,
                animated=True,
                color="info",
                className="mb-2"
            ),
            html.Small(
                f"{status_text.title()} - {progress:.1f}% - {speed}",
                className="text-muted"
            )
        ], className="mt-2")
    
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Div([
                    html.H6(model_name, className="mb-1"),
                    html.Small(f"Size: ~{estimated_size}", className="text-muted")
                ], className="flex-grow-1"),
                html.Div([
                    status_badge,
                    action_btn
                ], className="d-flex align-items-center")
            ], className="d-flex align-items-start"),
            progress_section
        ])
    ], className="mb-3", id=f"model-card-{model_id}")


def register_download_callbacks(app):
    """Register callbacks for download functionality."""
    
    @app.callback(
        [Output("chart-qa-models-list", "children"),
         Output("base-models-list", "children")],
        Input("refresh-downloads-btn", "n_clicks"),
        prevent_initial_call=False
    )
    def update_model_lists(n_clicks):
        """Update the model download lists."""
        chart_qa_models = [
            ("llava_v1_5_7b", "LLaVA-v1.5-7B")
        ]
        
        base_models = [
            ("unichart", "UniChart")
        ]
        
        chart_qa_cards = [
            create_model_download_card(model_id, model_name, is_chart_qa=True)
            for model_id, model_name in chart_qa_models
        ]
        
        base_cards = [
            create_model_download_card(model_id, model_name, is_chart_qa=False)
            for model_id, model_name in base_models
        ]
        
        return chart_qa_cards, base_cards
    
    # Dynamic callbacks for download/stop buttons
    @app.callback(
        Output("dummy-output", "children", allow_duplicate=True),
        [Input({"type": "download-btn", "index": dash.ALL}, "n_clicks"),
         Input({"type": "stop-btn", "index": dash.ALL}, "n_clicks")],
        prevent_initial_call=True
    )
    def handle_download_actions(download_clicks, stop_clicks):
        """Handle download and stop button clicks."""
        if not ctx.triggered:
            return no_update
        
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        button_info = eval(button_id)  # Safe here since we control the format
        
        if button_info["type"] == "download-btn":
            model_id = button_info["index"]
            download_manager.start_download(model_id)
        elif button_info["type"] == "stop-btn":
            model_id = button_info["index"]
            download_manager.stop_download(model_id)
        
        return no_update
    
    # Auto-refresh progress every 2 seconds for active downloads
    @app.callback(
        Output("dummy-output", "children", allow_duplicate=True),
        Input("download-progress-interval", "n_intervals"),
        prevent_initial_call=True
    )
    def update_download_progress(n_intervals):
        """Update download progress periodically."""
        # This will trigger the model list refresh
        return no_update


# Add to layout
def get_download_layout_additions():
    """Get additional components needed for download functionality."""
    return [
        dcc.Interval(
            id="download-progress-interval",
            interval=2000,  # Update every 2 seconds
            n_intervals=0
        ),
        html.Div(id="dummy-output", style={"display": "none"})
    ]
