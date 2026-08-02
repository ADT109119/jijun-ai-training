import os, subprocess as sp
script_dir = os.path.dirname(os.path.abspath(__file__))
venv_python = os.path.join(script_dir, "venv", "Scripts", "python.exe")
log = os.path.join(os.getcwd(), "dataset", "qwen_budget512.log")
env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"
env["PYTHONUNBUFFERED"] = "1"
proc = sp.Popen(
    [venv_python, "-u", "generate_dataset.py",
     "--model", "vllm/Qwen3.6-27B",
     "--count", "30",
     "--out_dir", "dataset/qwen_budget512",
     "--concurrency", "2",
     "--thinking_budget", "512",
     "--rest_interval", "0",
     "--rest_duration", "0"],
    stdout=open(log, "w", encoding="utf-8", buffering=1), stderr=sp.STDOUT,
    cwd=os.getcwd(), env=env,
    creationflags=sp.CREATE_NO_WINDOW
)
print(f"Budget512 compare started, PID: {proc.pid}, log: {log}")
