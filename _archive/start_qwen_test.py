import os, json, time, sys, re
import subprocess as sp
script_dir = os.path.dirname(os.path.abspath(__file__))
venv_python = os.path.join(script_dir, "venv", "Scripts", "python.exe")
log = os.path.join(os.getcwd(), "dataset", "qwen_test.log")
env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"
env["PYTHONUNBUFFERED"] = "1"
env["CONCURRENCY"] = "2"
proc = sp.Popen(
    [venv_python, "-u", "generate_dataset.py",
     "--model", "vllm/Qwen3.6-27B",
     "--count", "20",
     "--out_dir", "dataset/qwen_test",
     "--concurrency", "2",
     "--rest_interval", "0",
     "--rest_duration", "0"],
    stdout=open(log, "w", encoding="utf-8", buffering=1), stderr=sp.STDOUT,
    cwd=os.getcwd(), env=env,
    creationflags=sp.CREATE_NO_WINDOW
)
print(f"Qwen test started, PID: {proc.pid}")
