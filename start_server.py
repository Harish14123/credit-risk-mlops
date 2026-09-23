"""
Production runner to launch FastAPI REST API and/or Streamlit Dashboard.
"""
import sys
import subprocess
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def start_api():
    print("🚀 Starting FastAPI Server on http://0.0.0.0:8000 ...")
    return subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"])

def start_dashboard():
    print("🎨 Starting Streamlit Dashboard on http://0.0.0.0:8501 ...")
    return subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app_dashboard.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.headless", "true"])

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    procs = []
    
    if mode in ("api", "both"):
        procs.append(start_api())
    if mode in ("dashboard", "both"):
        procs.append(start_dashboard())
        
    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        print("\nStopping processes...")
        for p in procs:
            p.terminate()
