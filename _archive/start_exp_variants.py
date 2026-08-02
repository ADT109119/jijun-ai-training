import os, subprocess as sp
script_dir = os.path.dirname(os.path.abspath(__file__))
venv_python = os.path.join(script_dir, "venv", "Scripts", "python.exe")
log = os.path.join(os.getcwd(), "dataset", "exp_variants.log")
env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"
env["PYTHONUNBUFFERED"] = "1"
proc = sp.Popen(
    [venv_python, "-X", "utf8", "-u", "exp_prompt_variants.py"],
    stdout=open(log, "w", encoding="utf-8", buffering=1), stderr=sp.STDOUT,
    cwd=os.getcwd(), env=env,
    creationflags=sp.CREATE_NO_WINDOW
)
print(f"Exp started, PID: {proc.pid}, log: {log}")
