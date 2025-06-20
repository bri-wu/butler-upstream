# app.py – Day‑1.1 prototype for the “Simulation Butler”
# -------------------------------------------------------
# This refresh fixes the missing‑log issue you hit:
#   • Runs the entry script from **its own directory** so relative paths work.
#   • Recursively searches for the first `log.lammps` produced and plots it.
#   • Adds a little directory browser so you can inspect what files were created.
# This is still *manifest‑less* and single‑script, but robust enough for the
# current LAMMPSStructures repo. Later we’ll re‑introduce the manifest layer.

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Optional

import git  # GitPython
import pandas as pd
import streamlit as st
from plotly import express as px

st.set_page_config(page_title="Simulation Butler – Day‑1 Pilot", layout="wide")

################################################################################
# Helper functions
################################################################################

TMP_ROOT = Path(tempfile.gettempdir()) / "sim_butler_runs"
TMP_ROOT.mkdir(exist_ok=True)


def clone_repo(url: str) -> Path:
    """Clone *url* into a fresh temp dir and return the path."""
    dest = Path(tempfile.mkdtemp(prefix="sim_butler_", dir=TMP_ROOT))
    try:
        git.Repo.clone_from(url, dest)
    except Exception as exc:
        shutil.rmtree(dest, ignore_errors=True)
        raise RuntimeError(f"Git clone failed: {exc}") from exc
    return dest


def detect_entry_script(repo_path: Path) -> Optional[Path]:
    """Heuristic: choose *cylindrical_sheet.py* if present, else first *.py*."""
    for p in repo_path.rglob("cylindrical_sheet.py"):
        return p
    # fallback – first reasonably small Python file under 500 lines
    for p in repo_path.rglob("*.py"):
        if p.stat().st_size < 200_000:  # ~5k LOC
            return p
    return None


def stream_subprocess(cmd: list[str], cwd: Path):
    """Yield lines from a subprocess in real‑time."""
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        yield line.rstrip()
    proc.wait()
    returncode = proc.returncode
    if returncode:
        raise RuntimeError(f"Process exited with status {returncode}")


def find_log_file(run_root: Path, output_folder: str) -> Optional[Path]:
    """Look for *log.lammps* inside *output_folder*; fallback to first found."""
    primary = run_root / output_folder / "log.lammps"
    if primary.exists():
        return primary
    # walk for first match
    for p in run_root.rglob("log.lammps"):
        return p
    return None


def parse_lammps_log(path: Path) -> pd.DataFrame:
    """Return thermo table as DataFrame."""
    with path.open() as fh:
        lines = fh.readlines()
    header_idx = next(i for i, l in enumerate(lines) if l.strip().startswith("Step"))
    headers = lines[header_idx].split()
    data = [l.split() for l in lines[header_idx + 1:] if re.match(r"\s*\d", l)]
    df = pd.DataFrame(data, columns=headers).apply(pd.to_numeric)
    return df

################################################################################
# UI
################################################################################

st.title("🛠️ Simulation Butler – Day‑1 Pilot")

repo_url = st.text_input("Paste a public Git URL to a simulation repo:",
                         value="https://github.com/adguerra/LAMMPSStructures")

if st.button("📥 Clone repo"):
    with st.spinner("Cloning…"):
        try:
            repo_path = clone_repo(repo_url)
            st.session_state["repo_path"] = repo_path
        except Exception as e:
            st.error(str(e))
            st.stop()

# ---- Once a repo is cloned ----
if "repo_path" in st.session_state:
    repo_path: Path = st.session_state["repo_path"]
    st.success(f"Cloned to {repo_path}")

    # Detect entry script
    entry_script = detect_entry_script(repo_path)
    if entry_script is None:
        st.error("❌ No Python script found in repo.")
        st.stop()

    st.text_input("Entry script to run:", value=str(entry_script), key="entry_script")

    # Output folder – script expects one positional arg
    out_folder = st.text_input("Output folder name (argument to script):", value="sim1")

    cols = st.columns(2)
    sim_time = cols[0].number_input("Total physical time [s]", value=0.5)
    timestep = cols[1].number_input("LAMMPS time‑step [s]", value=1e‑6, format="%.1e")

    if st.button("▶️ Run simulation"):
        st.markdown(f"```bash\npython {Path(entry_script).name} {out_folder}\n```")
        console = st.empty()
        log_lines = []
        try:
            for ln in stream_subprocess([
                sys.executable,
                str(entry_script.name),  # run from script dir so use basename
                out_folder,
                # pass extra CLI opts for future versions here
            ], cwd=entry_script.parent):
                log_lines.append(ln)
                console.markdown("\n".join(["```", *log_lines[-20:], "```"]))
            st.success("Simulation finished ✔️")
        except Exception as e:
            st.error(str(e))
            st.stop()

        # --- locate log file ---
        log_path = find_log_file(entry_script.parent, out_folder)
        if not log_path:
            st.warning("Couldn't find log.lammps – nothing to plot")
            # Offer a directory tree for inspection
            with st.expander("📂 Browse run directory"):
                for p in (entry_script.parent / out_folder).rglob("*"):
                    st.markdown(str(p.relative_to(entry_script.parent)))
            st.stop()

        # --- plot ---
        df = parse_lammps_log(log_path)
        st.subheader("Thermo output (first 5 columns)")
        col_choices = df.columns.tolist()[:5]
        for col in col_choices:
            fig = px.line(df, x="Step", y=col, title=col)
            st.plotly_chart(fig, use_container_width=True)
