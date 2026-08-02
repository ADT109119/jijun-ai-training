import subprocess, sys, os
venv_python = os.path.join(os.path.dirname(sys.executable), "python.exe")
script = os.path.join(os.getcwd(), "generate_dataset.py")
log = os.path.join(os.getcwd(), "dataset", "generation_run.log")
os.makedirs(os.path.join(os.getcwd(), "dataset"), exist_ok=True)
with open(log, "w") as f:
    f.write("Starting generation...\n")
proc = subprocess.Popen(
    [venv_python, script, "--count", "2000", "--concurrency", "3", "--rest_interval", "0"],
    stdout=open(log, "a"), stderr=subprocess.STDOUT,
    cwd=os.getcwd(),
    creationflags=subprocess.CREATE_NO_WINDOW
)
print("PID:", proc.pid)
with open(log, "a") as f:
    f.write(f"PID: {proc.pid}\n")
