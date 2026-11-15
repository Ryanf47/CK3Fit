"""Builds CK3Fit.exe from this folder (made by make_app.py). Run from CK3Fit_App:
    python build_exe.py
Result: dist\\CK3Fit\\CK3Fit.exe (with its data folder next to it) and CK3Fit_release.zip to share.
"""
import importlib.metadata as md
import os
import shutil
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.join(HERE, "engine")
DATA = os.path.join(HERE, "data")
DIST = os.path.join(HERE, "dist", "CK3Fit")
META = ["torch", "torchvision", "transformers", "tokenizers", "safetensors", "huggingface-hub", "tqdm", "regex",
        "requests", "packaging", "filelock", "numpy", "pyyaml", "onnx", "onnx2torch", "mediapipe", "opencv-python",
        "opencv-contrib-python", "scipy", "pillow"]
# kept out of the shared zip: downloaded by the app itself on first run (InsightFace's licence: non-commercial research)
NOT_SHARED = [os.path.join("data", "results", "arcface")]


def have(pkg):
    try:
        md.version(pkg)
        return True
    except md.PackageNotFoundError:
        return False


def main():
    for need in (ENGINE, DATA, os.path.join(ENGINE, "ck3fit_app.py")):
        if not os.path.exists(need):
            sys.exit(f"Missing {need} - run 'python make_app.py' in CK3Fit first.")
    if not have("pyinstaller"):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    mods = sorted(f[:-3] for f in os.listdir(ENGINE) if f.endswith(".py") and f != "ck3fit_app.py")
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir", "--windowed",
           "--name", "CK3Fit", "--paths", ENGINE,
           "--distpath", os.path.join(HERE, "dist"), "--workpath", os.path.join(HERE, "build"),
           "--specpath", os.path.join(HERE, "build"),
           "--collect-all", "mediapipe", "--collect-all", "onnx2torch",
           "--collect-submodules", "transformers.models.dinov2",
           "--collect-data", "certifi", "--hidden-import", "certifi"]
    for m in mods:
        cmd += ["--hidden-import", m]
    for p in META:
        if have(p):
            cmd += ["--copy-metadata", p]
    cmd.append(os.path.join(ENGINE, "ck3fit_app.py"))
    print("Building (10-20 minutes; the result is large because of torch)...", flush=True)
    subprocess.check_call(cmd, cwd=HERE)

    print("Copying data next to the exe...", flush=True)
    shutil.rmtree(os.path.join(DIST, "data"), ignore_errors=True)
    shutil.copytree(DATA, os.path.join(DIST, "data"))
    for f in ("README_app.txt",):
        if os.path.exists(os.path.join(HERE, f)):
            shutil.copy2(os.path.join(HERE, f), os.path.join(DIST, "README.txt"))

    zp = os.path.join(HERE, "CK3Fit_release.zip")
    print(f"Zipping {zp} (leaves out the face-recognition files; the app downloads them on first run)...", flush=True)
    skip = [os.path.normcase(os.path.join(DIST, s)) for s in NOT_SHARED]
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for d, _, files in os.walk(DIST):
            if any(os.path.normcase(d).startswith(s) for s in skip):
                continue
            for f in files:
                p = os.path.join(d, f)
                z.write(p, os.path.join("CK3Fit", os.path.relpath(p, DIST)))
    gb = os.path.getsize(zp) / 1e9
    print(f"\nDone.\n  Try it:  {os.path.join(DIST, 'CK3Fit.exe')}\n  Share:   {zp}  ({gb:.1f} GB)")


if __name__ == "__main__":
    main()
