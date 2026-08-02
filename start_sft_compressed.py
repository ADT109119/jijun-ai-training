import subprocess as sp, os, sys
script_dir = os.path.dirname(os.path.abspath(__file__))
venv_python = os.path.join(script_dir, "venv", "Scripts", "python.exe")
if not os.path.exists(venv_python):
    venv_python = os.path.join(os.path.dirname(sys.executable), "python.exe")
script = os.path.join(os.getcwd(), "train_custom_sft.py")
log = os.path.join(os.getcwd(), "dataset", "sft_compressed.log")
os.makedirs(os.path.join(os.getcwd(), "saves", "compressed"), exist_ok=True)
os.makedirs(os.path.join(os.getcwd(), "dataset"), exist_ok=True)
env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"
env["PYTHONUNBUFFERED"] = "1"
with open(log, "w", encoding="utf-8") as f:
    f.write("Starting compressed-format SFT...\n")
proc = sp.Popen(
    [venv_python, "-u", script,
     "--pretrained_path", "./saves/best_pretrain_model.pt",
     "--format", "compressed",
     "--batch_size", "8",
     "--epochs", "8",
     "--lr", "2e-4",
     "--max_seq_len", "1024",
     "--save_dir", "./saves/compressed",
     "--eval_generate",
     "--max_eval_samples", "20"],
    stdout=open(log, "a", buffering=1), stderr=sp.STDOUT,
    cwd=os.getcwd(),
    env=env,
    creationflags=sp.CREATE_NO_WINDOW
)
with open(log, "a") as f:
    f.write(f"PID: {proc.pid}\n")
print(f"Compressed SFT started, PID: {proc.pid}")
print(f"Log: dataset/sft_compressed.log")
