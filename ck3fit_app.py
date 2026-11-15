"""CK3Fit - photos of a person -> Crusader Kings III Ruler Designer DNA.

1. Put each person's photos in their own folder (or pick a folder that holds several people's folders).
2. Choose woman or man.
3. "Check photos" says which photos are usable and what's missing.
4. "Make DNA" writes <person>.txt - open it, copy everything, and paste it in the Ruler Designer (Paste DNA).
"""
import multiprocessing
import os
import queue
import shutil
import sys
import threading
import time
import traceback

APP_NAME = "CK3Fit"
PROFILES = {
    "female": ["--model", os.path.join("results", "model_ft_tilt2.pt")],
    "male": ["--sex", "male", "--model", os.path.join("results", "model_ftm_full_tiltbeard.pt")],
}
PHOTO_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def data_dir():
    if getattr(sys, "frozen", False):
        return os.path.join(os.path.dirname(sys.executable), "data")
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))


def fix_certificates():
    """the packaged app has no list of trusted web certificates of its own -> use certifi's (for the one-time downloads)"""
    try:
        import ssl
        import certifi
        ca = certifi.where()
        if os.path.exists(ca):
            os.environ.setdefault("SSL_CERT_FILE", ca)
            ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=ca)
    except Exception:
        pass


def setup_engine():
    d = data_dir()
    if not os.path.isdir(d):
        raise SystemExit(f"The data folder is missing: {d}\nKeep the 'data' folder next to {APP_NAME}.exe.")
    os.environ["CK3FIT_HOME"] = d
    fix_certificates()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.chdir(d)
    if not getattr(sys, "frozen", False):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import photo_to_dna
    return photo_to_dna


def people_in(folder):
    """a folder of photos = one person; a folder of folders = several people"""
    subs = sorted(os.path.join(folder, d) for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)))
    subs = [s for s in subs if any(f.lower().endswith(PHOTO_EXT) for f in os.listdir(s))]
    if subs:
        return subs
    if any(f.lower().endswith(PHOTO_EXT) for f in os.listdir(folder)):
        return [folder]
    return []


def pname(folder):
    return os.path.basename(os.path.normpath(folder)).replace(" ", "_")


def check(P, folder, sex, say):
    """-> list of (person, ok, summary lines)"""
    results = []
    for person in people_in(folder):
        P.main(["--photos", person, "--check"] + PROFILES[sex])
        log = os.path.join(data_dir(), "results", "photo_check", pname(person), "check.log")
        if not os.path.exists(log):
            results.append((pname(person), False, ["No face found in any photo."]))
            continue
        lines = open(log, encoding="utf-8", errors="replace").read().splitlines()
        head = lines[0] if lines else ""
        flagged = [ln.strip() for ln in lines[3:] if "<-" in ln or "no face" in ln.lower() or "not used" in ln.lower()]
        results.append((pname(person), head.split(": ", 1)[-1].startswith("OK"), [head] + flagged))
    return results


def make(P, folder, sex, out_dir, say):
    run = "app_" + time.strftime("%Y%m%d_%H%M%S")
    made = []
    for person in people_in(folder):
        P.main(["--photos", person, "--out", run] + PROFILES[sex])
        src = os.path.join(data_dir(), "results", run, pname(person), f"{pname(person)}_final.txt")
        if os.path.exists(src):
            os.makedirs(out_dir, exist_ok=True)
            dst = os.path.join(out_dir, f"{pname(person)}.txt")
            shutil.copyfile(src, dst)
            made.append(dst)
        else:
            say(f"\n{pname(person)}: no DNA made - run Check photos to see why.\n")
    shutil.rmtree(os.path.join(data_dir(), "results", run), ignore_errors=True)
    return made


# ---------------------------------------------------------------- window
class Writer:
    def __init__(s, q):
        s.q = q

    def write(s, t):
        if t:
            s.q.put(t)

    def flush(s):
        pass


def gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title(APP_NAME)
    root.geometry("760x560")
    q = queue.Queue()
    state = {"P": None, "busy": False, "last": []}
    photos = tk.StringVar()
    out = tk.StringVar(value=os.path.join(os.path.expanduser("~"), "Documents", "CK3Fit DNA"))
    sex = tk.StringVar(value="female")

    frm = ttk.Frame(root, padding=12)
    frm.pack(fill="both", expand=True)
    frm.columnconfigure(1, weight=1)
    ttk.Label(frm, text="Photo folder").grid(row=0, column=0, sticky="w")
    ttk.Entry(frm, textvariable=photos).grid(row=0, column=1, sticky="ew", padx=6)
    ttk.Button(frm, text="Browse...", command=lambda: photos.set(filedialog.askdirectory() or photos.get())).grid(row=0, column=2)
    ttk.Label(frm, text="Save DNA to").grid(row=1, column=0, sticky="w", pady=(6, 0))
    ttk.Entry(frm, textvariable=out).grid(row=1, column=1, sticky="ew", padx=6, pady=(6, 0))
    ttk.Button(frm, text="Browse...", command=lambda: out.set(filedialog.askdirectory() or out.get())).grid(row=1, column=2, pady=(6, 0))
    sx = ttk.Frame(frm)
    sx.grid(row=2, column=1, sticky="w", pady=8)
    ttk.Radiobutton(sx, text="Woman", value="female", variable=sex).pack(side="left", padx=(0, 16))
    ttk.Radiobutton(sx, text="Man", value="male", variable=sex).pack(side="left")
    btns = ttk.Frame(frm)
    btns.grid(row=3, column=0, columnspan=3, sticky="w", pady=(0, 8))
    log = tk.Text(frm, wrap="word", height=20)
    log.grid(row=4, column=0, columnspan=3, sticky="nsew")
    frm.rowconfigure(4, weight=1)
    status = ttk.Label(frm, text="Pick a photo folder: one person's photos, or a folder with one folder per person.")
    status.grid(row=5, column=0, columnspan=3, sticky="w", pady=(6, 0))

    def say(t):
        q.put(t)

    def job(kind):
        if state["busy"]:
            return
        f = photos.get().strip()
        if not f or not os.path.isdir(f):
            messagebox.showinfo(APP_NAME, "Pick the photo folder first.")
            return
        if not people_in(f):
            messagebox.showinfo(APP_NAME, "No photos found there (jpg, png or webp), and no folders with photos.")
            return
        state["busy"] = True
        for b in btns.winfo_children():
            b.state(["disabled"])
        status.config(text="Working... the first run takes longer (it loads the models"
                           + (" and downloads two face-recognition files" if not os.path.isdir(
                               os.path.join(data_dir(), "results", "arcface")) else "") + ").")

        def work():
            old = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = Writer(q)
            try:
                if state["P"] is None:
                    state["P"] = setup_engine()
                if kind == "check":
                    res = check(state["P"], f, sex.get(), say)
                    say("\n==== PHOTO CHECK ====\n")
                    for nm, ok, lines in res:
                        say(f"\n{nm}: {'OK' if ok else 'NEEDS ATTENTION'}\n" + "".join(f"   {ln}\n" for ln in lines))
                    say("\nAbout 4+ clear front photos and 1+ turned (three-quarter) photo per person works best.\n")
                    q.put(("done", "Photo check finished."))
                else:
                    made = make(state["P"], f, sex.get(), out.get(), say)
                    state["last"] = made
                    say("\n==== DNA ====\n" + "".join(f"   {m}\n" for m in made))
                    q.put(("done", f"{len(made)} DNA file(s) saved." + (" Copied to the clipboard - paste it in the Ruler "
                                                                         "Designer." if len(made) == 1 else "")))
            except SystemExit as e:
                say(f"\n{e}\n")
                q.put(("done", "Stopped - see the message above."))
            except Exception:
                say("\n" + traceback.format_exc())
                q.put(("done", "Something went wrong - see the message above."))
            finally:
                sys.stdout, sys.stderr = old
        threading.Thread(target=work, daemon=True).start()

    def poll():
        try:
            while True:
                m = q.get_nowait()
                if isinstance(m, tuple):
                    state["busy"] = False
                    for b in btns.winfo_children():
                        b.state(["!disabled"])
                    status.config(text=m[1])
                    if len(state["last"]) == 1 and "DNA" in m[1]:
                        root.clipboard_clear()
                        root.clipboard_append(open(state["last"][0], encoding="utf-8").read())
                else:
                    log.insert("end", m)
                    log.see("end")
        except queue.Empty:
            pass
        root.after(100, poll)

    def open_out():
        os.makedirs(out.get(), exist_ok=True)
        os.startfile(out.get())

    ttk.Button(btns, text="Check photos", command=lambda: job("check")).pack(side="left", padx=(0, 8))
    ttk.Button(btns, text="Make DNA", command=lambda: job("make")).pack(side="left", padx=(0, 8))
    ttk.Button(btns, text="Open DNA folder", command=open_out).pack(side="left")
    log.insert("end", __doc__.strip() + "\n")
    root.after(100, poll)
    root.mainloop()


def cli(argv):
    """CK3Fit.exe --photos <folder> [--man] [--out <folder>] [--check]"""
    import argparse
    ap = argparse.ArgumentParser(prog=APP_NAME)
    ap.add_argument("--photos", required=True)
    ap.add_argument("--man", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.expanduser("~"), "Documents", "CK3Fit DNA"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    a.photos, a.out = os.path.abspath(a.photos), os.path.abspath(a.out)
    P = setup_engine()
    sex = "male" if a.man else "female"
    if a.check:
        for nm, ok, lines in check(P, a.photos, sex, print):
            print(f"\n{nm}: {'OK' if ok else 'NEEDS ATTENTION'}\n" + "".join(f"   {ln}\n" for ln in lines))
    else:
        for m in make(P, a.photos, sex, a.out, print):
            print(m)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if len(sys.argv) > 1:
        cli(sys.argv[1:])
    else:
        gui()
