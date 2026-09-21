#!/usr/bin/env python3
"""Compatibility entry point for the shared native backend generator."""
from pathlib import Path
import runpy
root = Path(__file__).resolve().parents[2]
runpy.run_path(str(root / 'scripts/embed_backend_kernels.py'), run_name='__main__')
