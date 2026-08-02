import os, subprocess as sp
script_dir = os.path.dirname(os.path.abspath(__file__))
venv_python = os.path.join(script_dir, "venv", "Scripts", "python.exe")
log = os.path.join(os.getcwd(), "dataset", "qwen_batch2.log")
os.makedirs(os.path.join(os.getcwd(), "dataset", "qwen_batch2"), exist_ok=True)
env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"
env["PYTHONUNBUFFERED"] = "1"
proc = sp.Popen(
    [venv_python, "-u", "generate_dataset.py",
     "--model", "vllm/Qwen3.6-27B",
     "--count", "1000",
     "--out_dir", "dataset/qwen_batch2",
     "--concurrency", "3",
     "--rest_interval", "2",
     "--rest_duration", "1"],
    stdout=open(log, "w", encoding="utf-8", buffering=1), stderr=sp.STDOUT,
    cwd=os.getcwd(), env=env,
    creationflags=sp.CREATE_NO_WINDOW
)
print(f"Qwen general batch2 started, PID: {proc.pid}, log: {log}")
