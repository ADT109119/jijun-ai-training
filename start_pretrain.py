import subprocess as sp, os, sys
script_dir = os.path.dirname(os.path.abspath(__file__))
venv_python = os.path.join(script_dir, "venv", "Scripts", "python.exe")
if not os.path.exists(venv_python):
    venv_python = os.path.join(os.path.dirname(sys.executable), "python.exe")
script = os.path.join(os.getcwd(), "train_custom_pretrain.py")
log = os.path.join(os.getcwd(), "dataset", "pretrain_run.log")
os.makedirs(os.path.join(os.getcwd(), "dataset"), exist_ok=True)
env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"
env["PYTHONUNBUFFERED"] = "1"
_ver = sp.run([venv_python, "-c", "import torch; print(torch.cuda.is_available())"], capture_output=True, text=True, cwd=os.getcwd(), env=env)
with open(log, "w", encoding="utf-8") as f:
    f.write(f"Using Python: {venv_python}\n")
    f.write(f"CUDA check: {_ver.stdout.strip()}\n")
    f.write(f"Starting pretrain...\n")
proc = sp.Popen(
    [venv_python, "-u", script,
     "--pretrain_jsonl", "Z:/Aria2/pretrain.jsonl",
     "--pretrain_limit", "100000",
     "--no_ratio_limit",
     "--epochs", "3",
     "--batch_size", "8",
     "--max_seq_len", "1024",
     "--lr", "3e-4",
     "--save_dir", "./saves",
     "--use_wikipedia"],
    stdout=open(log, "a", buffering=1), stderr=sp.STDOUT,
    cwd=os.getcwd(),
    env=env,
    creationflags=sp.CREATE_NO_WINDOW
)
with open(log, "a") as f:
    f.write(f"PID: {proc.pid}\n")
print(f"Pretrain started, PID: {proc.pid}")
print(f"Log: dataset/pretrain_run.log")
