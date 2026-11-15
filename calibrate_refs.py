"""Learn how the network's reading of PHOTOS is off, from the hand-made reference characters.

For every reference person it reads the photos with the trained network, compares with your DNA, and learns one fixed correction per slider. Every number it reports is
leave-one-out: each person is corrected using only the OTHER people, so the scores show what happens on someone the correction has never seen.

    python calibrate_refs.py

Looks inside --refs (default agot_dna2) for:
    f_<N> photo folders (any depth)        -> female\\<N>.txt  or maybes\\female\\female_<NNN>.txt
    extra_f<N> photo folders (any depth)   -> extra_dna\\female\\extra_f<NN>.txt

Writes:
    results\\photo_correction.json        used automatically by photo_to_dna.py (-> <person>_fixed.txt)
    results\\refs\\report.txt              the scores
    results\\refs\\<id>\\  <id>_1_human.txt / <id>_2_raw.txt / <id>_3_fixed.txt   for autoshot side-by-side
"""
import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

HERE = (os.environ.get("CK3FIT_HOME") or os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import photo_to_dna as P          # noqa: E402
import train_reverse as tr         # noqa: E402


def find_pairs(root, sex="female"):
    """Photo folders f_<N> and extra_f<N> (men: m_<N> and extra_m<N>), wherever they sit under root; DNAs in the
    female (male) folders."""
    def first(paths):
        return next((q for q in paths if os.path.exists(q)), None)
    if sex == "male":
        pairs, seen = [], set()
        for d in sorted(glob.glob(os.path.join(root, "**", "m_*"), recursive=True)):
            n = os.path.basename(d)[2:]
            if not os.path.isdir(d) or not n.isdigit() or ("m", n) in seen:
                continue
            seen.add(("m", n))
            c = first([os.path.join(root, "male", f"{n}.txt"), os.path.join(root, "male", f"{int(n):03d}.txt"),
                       os.path.join(root, "maybes", "male", f"male_{int(n):03d}.txt")])
            if c:
                pairs.append((f"m_{int(n):03d}", d, c))
            else:
                print(f"  {os.path.basename(d)}: no DNA found - skipped")
        for d in sorted(glob.glob(os.path.join(root, "**", "extra_m*"), recursive=True)):
            n = os.path.basename(d)[len("extra_m"):]
            if not os.path.isdir(d) or not n.isdigit() or ("e", n) in seen:
                continue
            seen.add(("e", n))
            c = first([os.path.join(root, sub, "male", f"extra_m{int(n):02d}.txt") for sub in ("extra_dna", "extra")])
            if c:
                pairs.append((f"extra_m{int(n):02d}", d, c))
            else:
                print(f"  {os.path.basename(d)}: no DNA found - skipped")
        return pairs
    pairs, seen = [], set()
    for d in sorted(glob.glob(os.path.join(root, "**", "f_*"), recursive=True)):
        n = os.path.basename(d)[2:]
        if not os.path.isdir(d) or not n.isdigit() or ("f", n) in seen:
            continue
        seen.add(("f", n))
        c = first([os.path.join(root, "female", f"{n}.txt"), os.path.join(root, "female", f"{int(n):03d}.txt"),
                   os.path.join(root, "maybes", "female", f"female_{int(n):03d}.txt")])
        if c:
            pairs.append((f"f_{int(n):03d}", d, c))
        else:
            print(f"  {os.path.basename(d)}: no DNA found - skipped")
    for d in sorted(glob.glob(os.path.join(root, "**", "extra_f*"), recursive=True)):
        n = os.path.basename(d)[len("extra_f"):]
        if not os.path.isdir(d) or not n.isdigit() or ("e", n) in seen:
            continue
        seen.add(("e", n))
        c = first([os.path.join(root, sub, "female", f"extra_f{int(n):02d}.txt") for sub in ("extra_dna", "extra")])
        if c:
            pairs.append((f"extra_f{int(n):02d}", d, c))
        else:
            print(f"  {os.path.basename(d)}: no DNA found - skipped")
    return pairs


def dna_targets(text, tg):
    """DNA -> the same 0-1 targets the network predicts (NaN where it can't be compared)."""
    genes, cols = tr.parse(text)
    G = len(tg.genes)
    v, cls = np.full(G, np.nan), {}
    for j, g in enumerate(tg.genes):
        if g not in genes:
            continue
        t, x = genes[g]
        k = tg.kind[g]
        if k in ("plain", "value"):
            v[j] = x / 255
        elif k == "signed":
            v[j] = 0.5 if t.startswith("vanilla_") else (0.5 + x / 510 if tr.positive(t) else 0.5 - x / 510)
        elif t in tg.tmpl[g]:
            cls[j] = tg.tmpl[g].index(t)
            v[j] = x / 255
    c = np.array([q for k in ("hair_color", "skin_color", "eye_color") for q in cols.get(k, (np.nan, np.nan))],
                 float) / 255
    return v, cls, c


def fit(pred, true, method):
    """Per-column correction true ~ a + b*pred. pred/true: people x columns (NaN = missing)."""
    K = pred.shape[1]
    a, b = np.zeros(K), np.ones(K)
    for j in range(K):
        m = ~np.isnan(true[:, j])
        if m.sum() < 3:
            continue
        x, y = pred[m, j], true[m, j]
        if method == "offset":
            a[j] = (y - x).mean()
        else:                                     # linear, slope shrunk toward 1 and kept in 0.5-3
            vx = x.var()
            bb = ((x - x.mean()) * (y - y.mean())).mean() / vx if vx > 1e-6 else 1.0
            n = m.sum()
            bb = (n * bb + 10 * 1.0) / (n + 10)
            b[j] = float(np.clip(bb, 0.5, 3.0))
            a[j] = y.mean() - b[j] * x.mean()
    return a, b


def loo(pred, true, method):
    out = np.full(pred.shape, np.nan)
    for i in range(len(pred)):
        keep = np.arange(len(pred)) != i
        if method == "mean":                      # ignore the photos: guess the other people's average DNA
            out[i] = np.nanmean(true[keep], 0)
        elif method == "raw":
            out[i] = pred[i]
        else:
            a, b = fit(pred[keep], true[keep], method)
            out[i] = np.clip(a + b * pred[i], 0, 1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refs", default="agot_dna2")
    ap.add_argument("--model", default=os.path.join("results", "model_29996_v3_base_photo.pt"))
    ap.add_argument("--template", default="template_hair17.txt")
    ap.add_argument("--data", default=os.path.join("random_faces", "noise_female"))
    a = ap.parse_args()
    # settings photo_to_dna's per-person step expects
    a.front_max, a.max_turned, a.stretch, a.check, a.out_dir = 0.20, 6, [], False, os.path.join("refs", "_work")

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cal = json.load(open(os.path.join(HERE, "results", "photo_calib.json")))
    mesh = P.Mesh()
    net, ck, tg = P.load_model(os.path.join(HERE, a.model) if not os.path.isabs(a.model) else a.model, dev)
    template = open(os.path.join(HERE, a.template), encoding="utf-8").read()

    # catch the network's internal 1024-number description of each face (before it is turned into sliders)
    FEAT, FZ = [], []
    net.trunk.register_forward_hook(lambda m, i, o: FEAT.append(o.float().detach().cpu().numpy()))
    pairs = find_pairs(os.path.join(HERE, a.refs) if not os.path.isabs(a.refs) else a.refs)
    print(f"{len(pairs)} reference people with photos and a hand-made DNA.")
    names, PV, PC, PCL, TV, TC, TCL, texts = [], [], [], [], [], [], [], []
    for pid, folder, dna in pairs:
        FEAT.clear()
        r = P.person(folder, pid, a, mesh, cal, net, ck, tg, dev, template)
        if not r:
            continue
        text = open(dna, encoding="utf-8", errors="ignore").read()
        tv, tcl, tc = dna_targets(text, tg)
        FZ.append(np.concatenate(FEAT).mean(0))
        names.append(pid); PV.append(r["v"]); PC.append(r["c"]); PCL.append(r["cls"])
        TV.append(tv); TC.append(tc); TCL.append(tcl); texts.append(text)
    shutil.rmtree(os.path.join(HERE, "results", "refs", "_work"), ignore_errors=True)
    n = len(names)
    if n < 8:
        sys.exit(f"Only {n} usable people - need at least 8.")
    PV, TV, PC, TC, FZ = map(np.array, (PV, TV, PC, TC, FZ))

    slide = [j for j, g in enumerate(tg.genes)
             if tg.kind[g] != "choice" and ck["sd"][j] * 255 >= 2 and (~np.isnan(TV[:, j])).sum() >= n // 2]
    res = {m: loo(PV[:, slide], TV[:, slide], m) for m in ("mean", "raw", "offset", "linear")}
    err = {m: np.nanmean(np.abs(res[m] - TV[:, slide]), 0) * 255 for m in res}
    cres = {m: loo(PC, TC, m) for m in ("mean", "raw", "offset", "linear")}
    cerr = {m: np.nanmean(np.abs(cres[m] - TC), 0) * 255 for m in cres}

    # learned straight from the network's internal description (ridge regression, leave-one-out)
    for lab, X in (("ridge on all slider outputs", PV), ("ridge on internal features", FZ)):
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        for lam in (10, 100, 1000):
            out = np.full((n, len(slide)), np.nan)
            for i in range(n):
                k = np.arange(n) != i
                for q, j in enumerate(slide):
                    m = k & ~np.isnan(TV[:, j])
                    Xm, y = X[m], TV[m, j]
                    w = Xm.T @ np.linalg.solve(Xm @ Xm.T + lam * np.eye(len(y)), y - y.mean())
                    out[i, q] = np.clip(y.mean() + X[i] @ w, 0, 1)
            e = np.nanmean(np.abs(out - TV[:, slide]), 0) * 255
            key = f"{lab}, strength {lam}"
            res[key], err[key] = out, e
    best = min(("offset", "linear"), key=lambda m: np.median(err[m]))
    lines = [f"{n} reference people. All scores leave-one-out (each person corrected using only the others).",
             "Error = average distance from your hand-made DNA, 0-255 scale. Lower is better.",
             "'guess average' ignores the photos completely - the network has to beat it to be useful.", "",
             "Sliders (median over %d):" % len(slide)]
    for m, lab in (("mean", "guess average   "), ("raw", "network, raw    "), ("offset", "network + offset"),
                   ("linear", "network + linear")):
        beat = int((err[m] < err["mean"]).sum()) if m != "mean" else 0
        lines.append(f"  {lab} {np.median(err[m]):5.1f}" +
                     (f"   beats guessing on {beat} of {len(slide)} sliders" if m != "mean" else ""))
    for key in [k for k in err if k.startswith("ridge")]:
        lines.append(f"  {key:38s} {np.median(err[key]):5.1f}   beats guessing on "
                     f"{int((err[key] < err['mean']).sum())} of {len(slide)} sliders")
    lines += ["", "Colours (hair x/y, skin x/y, eye x/y):"]
    for m in ("mean", "raw", "offset", "linear"):
        lines.append(f"  {m:7s} " + " ".join(f"{x:5.1f}" for x in cerr[m]))
    # buttons
    acc_raw, acc_maj, cnt = 0, 0, 0
    for j in tg.choice:
        for i in range(n):
            if j not in TCL[i]:
                continue
            others = [TCL[k][j] for k in range(n) if k != i and j in TCL[k]]
            maj = max(set(others), key=others.count) if others else -1
            acc_raw += PCL[i][j] == TCL[i][j]
            acc_maj += maj == TCL[i][j]
            cnt += 1
    if cnt:
        lines += ["", f"Buttons: network picks your button {100 * acc_raw / cnt:.0f}% of the time; "
                      f"always picking the most common one: {100 * acc_maj / cnt:.0f}%"]
    lines += ["", f"Saved correction: {best}", "",
              "Per slider (guess / raw / offset / linear), sorted by how much the correction helps:"]
    gain = err["mean"] - err[best]
    for k in np.argsort(-gain):
        lines.append(f"  {tg.genes[slide[k]]:36s} {err['mean'][k]:5.1f} {err['raw'][k]:5.1f} "
                     f"{err['offset'][k]:5.1f} {err['linear'][k]:5.1f}")

    # the correction itself, fitted on everyone
    a_s, b_s = fit(PV[:, slide], TV[:, slide], best)
    ca, cb = fit(PC, TC, best)
    json.dump({"method": best, "people": n, "genes": [tg.genes[j] for j in slide], "a": a_s.tolist(),
               "b": b_s.tolist(), "col_a": ca.tolist(), "col_b": cb.tolist()},
              open(P.CORR_PATH, "w"), indent=1)

    # side-by-side DNAs (the 'fixed' one is leave-one-out, so it's an honest preview)
    root = os.path.join(HERE, "results", "refs")
    os.makedirs(root, exist_ok=True)
    for i, pid in enumerate(names):
        d = os.path.join(root, pid)
        os.makedirs(d, exist_ok=True)
        vf = PV[i].copy()
        vf[slide] = res[best][i]
        cf = np.where(np.isnan(cres[best][i]), PC[i], cres[best][i])
        open(os.path.join(d, f"{pid}_1_human.txt"), "w", encoding="utf-8").write(texts[i])
        open(os.path.join(d, f"{pid}_2_raw.txt"), "w", encoding="utf-8").write(tg.write(template, PV[i], PCL[i], PC[i]))
        open(os.path.join(d, f"{pid}_3_fixed.txt"), "w", encoding="utf-8").write(tg.write(template, vf, PCL[i], cf))
    open(os.path.join(root, "report.txt"), "w").write("\n".join(lines))
    print("\n" + "\n".join(lines[:22]))
    print(f"\nSaved {P.CORR_PATH} and {root}\\report.txt")
    print(f"Screenshots:  foreach ($d in Get-ChildItem results\\refs -Directory) {{ python autoshot.py folder $d.FullName }}")


if __name__ == "__main__":
    main()
