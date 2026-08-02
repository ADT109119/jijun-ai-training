import subprocess, os, sys
script_dir = os.path.dirname(os.path.abspath(__file__))
venv_python = os.path.join(script_dir, "venv", "Scripts", "python.exe")
if not os.path.exists(venv_python):
    venv_python = os.path.join(os.path.dirname(sys.executable), "python.exe")
script = os.path.join(os.getcwd(), "generate_dataset.py")
log = os.path.join(os.getcwd(), "dataset", "generation_run.log")
os.makedirs(os.path.join(os.getcwd(), "dataset"), exist_ok=True)
env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"
env["PYTHONUNBUFFERED"] = "1"
with open(log, "w") as f:
    f.write("Starting generation with concurrency=4, max_tokens=1024\n")
proc = subprocess.Popen(
    [venv_python, "-u", script, "--count", "4000", "--concurrency", "4", "--rest_interval", "5", "--rest_duration", "5"],
    stdout=open(log, "a", buffering=1), stderr=subprocess.STDOUT,
    cwd=os.getcwd(),
    env=env,
    creationflags=subprocess.CREATE_NO_WINDOW
)
with open(log, "a") as f:
    f.write(f"PID: {proc.pid}\n")
print(f"Started PID: {proc.pid}")
