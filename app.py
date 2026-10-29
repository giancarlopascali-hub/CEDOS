"""CEDOS local application bootstrap."""

import os
import json
import socket
import sys
import warnings

# Suppress expected BoTorch/GPyTorch runtime warnings
warnings.filterwarnings("ignore", message=".*The model inputs are of type torch.float32.*")
warnings.filterwarnings("ignore", message=".*Optimization failed in `gen_candidates_scipy`.*")

# Resolve paths for PyInstaller standalone vs. Dev Mode
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS          # PyInstaller temp dir
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR   = os.path.join(BASE_DIR, 'static')

# Secondary path check for --onedir bundles placing resources inside _internal/
if not os.path.exists(os.path.join(TEMPLATE_DIR, 'index.html')):
    _internal = os.path.join(BASE_DIR, '_internal')
    if os.path.exists(os.path.join(_internal, 'templates')):
        TEMPLATE_DIR = os.path.join(_internal, 'templates')
        STATIC_DIR   = os.path.join(_internal, 'static')

# Setup paths inside Python path for absolute imports inside src/
sys.path.insert(0, BASE_DIR)

from flask import Flask, jsonify, render_template
from src.api.routes import api_bp
from src.api.auto_routes import auto_bp

# Initialize the Flask App
app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR
)

# Allow large JSON payloads for datasets (up to 200 MB).
# Without this, large uploads return a 413 HTML error page that the
# frontend cannot parse as JSON, causing "Unexpected token '<'" errors.
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB

# Register the EDOS/TransBO API and the Auto bridge once each.
app.register_blueprint(api_bp)
app.register_blueprint(auto_bp)


def _deployment_info():
    info = {
        'application': 'CEDOS',
        'application_commit': 'development',
        'edos_foundation_commit': 'development',
        'deployed_at': None,
    }
    path = os.path.join(BASE_DIR, 'build-info.json')
    try:
        with open(path, encoding='utf-8') as handle:
            info.update(json.load(handle))
    except (OSError, ValueError, TypeError):
        pass
    return info


@app.get('/health')
def health():
    return jsonify({'status': 'ok', 'application': 'CEDOS'})


@app.get('/version')
def version():
    return jsonify(_deployment_info())


@app.get('/auto-guide')
def auto_guide():
    return render_template('auto_guide.html')

# Ensure template/static directories exist when running in dev mode
if not getattr(sys, 'frozen', False):
    os.makedirs(TEMPLATE_DIR, exist_ok=True)
    os.makedirs(STATIC_DIR, exist_ok=True)

if __name__ == '__main__':
    import multiprocessing
    import threading
    import webbrowser
    import time

    multiprocessing.freeze_support()

    # Port selection:
    # 1. Read $PORT env var if provided (HF Spaces / Docker / Cloud)
    # 2. If $SPACE_ID is present (Hugging Face Spaces), default to 7860
    # 3. CEDOS_PORT optionally fixes the localhost preview port.
    # 4. Otherwise, try 5000 for the local IMEX companion.
    is_hf = bool(os.environ.get("SPACE_ID"))
    env_port = os.environ.get("PORT")
    local_port = os.environ.get("CEDOS_PORT")

    if env_port:
        port = int(env_port)
    elif is_hf:
        port = 7860
    elif local_port:
        port = int(local_port)
        if not 1 <= port <= 65535:
            raise ValueError("CEDOS_PORT must be between 1 and 65535.")
    else:
        preferred_port = 5000
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(('127.0.0.1', preferred_port))
                port = preferred_port
            except OSError:
                probe.bind(('127.0.0.1', 0))
                port = probe.getsockname()[1]

    url = f"http://127.0.0.1:{port}"

    print(f"\n" + "=" * 45)
    print(f"       CEDOS local preview")
    print(f"=" * 45)
    print(f"Server is starting...")
    print(f"URL: {url} (listening on port {port})")
    if not is_hf and not os.environ.get("NO_BROWSER"):
        print(f"Attempting to open browser automatically...")
    print(f"=" * 45 + "\n")

    def open_browser():
        time.sleep(2.0)
        try:
            webbrowser.open(url)
        except Exception as e:
            print(f"Error opening browser: {e}")

    # Auto-open browser only on local dev or frozen executable, never in cloud/HF spaces.
    if not is_hf and not os.environ.get("NO_BROWSER"):
        threading.Thread(target=open_browser, daemon=True).start()

    host = '0.0.0.0' if (os.environ.get("PORT") or is_hf) else '127.0.0.1'
    app.run(host=host, port=port, debug=False, use_reloader=False)
