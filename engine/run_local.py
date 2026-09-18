#!/usr/bin/env python3
from pathlib import Path
import subprocess, sys

repo = Path(__file__).resolve().parents[1]
p = subprocess.run([sys.executable, str(repo/"engine"/"earprint_engine.py")], cwd=repo)
raise SystemExit(p.returncode)
