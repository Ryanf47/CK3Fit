"""Reference correction on the FINAL face: learns how the whole pipeline's result is off, from the reference people
(photo folder + a hand-made DNA), and fixes it with one shift per slider.

For each reference person it runs the normal pipeline on the photos (exactly what makes <person>_final.txt) and
compares with the hand-made DNA. The shift per slider is the average difference, scaled by a strength chosen by
leave-one-out. Every score is leave-one-out: each person is corrected using only the OTHER people, so it shows what
happens on someone new. A slider is only corrected if that test says it helps.

    python ref_offset.py --sex male
        -> results\\ref_offset_male.json         used by photo_to_dna.py --sex male (-> <person>_ref.txt)
           results\\ref_offset_report_male.txt   the scores
           results\\refs_male\\<id>\\  <id>_1_human.txt / <id>_2_final.txt / <id>_3_ref.txt   to screenshot side by side
    (women:  python ref_offset.py   -> results\\ref_offset.json, refs_female\\ ...)

Looks inside --refs (default agot_dna2) for m_<N> / extra_m<N> photo folders (women: f_<N> / extra_f<N>) and their
DNAs, like calibrate_refs.py. No game needed; reading the photos takes a few minutes.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

HERE = (os.environ.get("CK3FIT_HOME") or os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import calibrate_refs as CR       # noqa: E402
import photo_to_dna as P          # noqa: E402
import train_reverse as tr        # noqa: E402

KEEP = ("face_detail", "expression", "complexion", "dacal", "decal", "body_type", "asymmetry", "beauty",
        "accessory", "gene_age", "mouth_open", "neck")           # never shifted (texture / expression / options)


# men: the STRENGTH of these option buttons is corrected too (which option stays as read). Our men came out with
# rounder heads, fuller cheeks and flatter eye areas than the hand-made men, in exactly these strengths.
MALE_STRENGTHS = ("gene_bs_head_shape", "face_detail_cheek_fat", "face_detail_eye_socket",
                  "face_detail_eye_lower_lid_def", "face_detail_eye_upper_lid_def", "face_detail_cheek_def")


def offset_path(sex):
    return os.path.join(HERE, "results", "ref_offset.json" if sex == "female" else f"ref_offset_{sex}.json")


def apply(text, tg, off):
    """her DNA text with the slider shifts added; every other line exactly as it was"""
    v, cls, col = CR.dna_targets(text, tg)
    v = v.copy()
    changed = set()
    for j, g in enumerate(tg.genes):
        if g in off and np.isfinite(v[j]):
            v[j] = float(np.clip(v[j] + off[g], 0, 1))
            changed.add(g)
    if not changed:
        return text
    full_cls = {j: cls.get(j, 0) for j in tg.choice}
    colf = np.where(np.isfinite(col), col, 0.5)
    written = tg.write(text, np.where(np.isfinite(v), v, 0.5), full_cls, colf)
    new = {m.group(1): m.group(0) for m in tr.GENE_RE.finditer(written)}
    return tr.GENE_RE.sub(lambda m: new.get(m.group(1), m.group(0)) if m.group(1) in changed else m.group(0), text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sex", choices=["female", "male"], default="female")
    ap.add_argument("--refs", default="agot_dna2")
    ap.add_argument("--bridge-weight", type=float, default=0.5,
                    help="the bridge's share in the mix (0 = main network only); use the same value in photo_to_dna.py")
    ap.add_argument("--no-strengths", action="store_true",
                    help="men: don't correct the head-shape / cheek-fat / eye-area option strengths")
    ap.add_argument("--min-gain", type=float, default=0.3,
                    help="only correct a slider if leave-one-out improves it by at least this much (0-255 scale)")
    a0 = ap.parse_args()
    ab = lambda p: p if os.path.isabs(p) else os.path.join(HERE, p)      # noqa: E731
    pairs = CR.find_pairs(ab(a0.refs), a0.sex)
    if len(pairs) < 10:
        sys.exit(f"Only {len(pairs)} reference {a0.sex} photo folders with a DNA found under {ab(a0.refs)}.")
    work = "refwork_" + a0.sex
    a, mesh, cal, net, ck, tg, dev, template = P.main(
        ["--photos", ab(a0.refs), "--out", work, "--sex", a0.sex, "--hair", "0",
         "--bridge-weight", str(a0.bridge_weight)], setup_only=True)
    if not getattr(a, "_bridge", None):
        sys.exit(f"The bridge didn't load ({a.bridge}).")
    strengths = MALE_STRENGTHS if a0.sex == "male" and not a0.no_strengths else ()
    use = np.array([(tg.kind[g] != "choice" and not any(k in g for k in KEEP)) or g in strengths for g in tg.genes])
    V, T, ids, finals = [], [], [], {}
    for pid, folder, dna in pairs:
        r = P.person(folder, pid, a, mesh, cal, net, ck, tg, dev, template)
        fin = os.path.join(HERE, "results", work, pid, f"{pid}_final.txt")
        if not r or not os.path.exists(fin):
            print(f"  {pid}: no final face (too few usable photos?) - skipped")
            continue
        text = open(fin, encoding="utf-8").read()
        V.append(CR.dna_targets(text, tg)[0])
        T.append(CR.dna_targets(open(dna, encoding="utf-8", errors="ignore").read(), tg)[0])
        ids.append(pid)
        finals[pid] = (text, dna)
    n = len(ids)
    if n < 10:
        sys.exit(f"Only {n} reference people gave a final face - need 10+.")
    V, T = np.array(V, float), np.array(T, float)
    D = T - V                                                           # hand-made minus ours

    def loo(k):
        """per slider: mean absolute error (0-1) after a k-strength shift learned without each person"""
        E = np.full(D.shape, np.nan)
        for i in range(n):
            o = np.nanmean(np.delete(D, i, 0), 0)
            E[i] = np.abs(T[i] - np.clip(V[i] + k * o, 0, 1))
        return np.nanmean(E, 0)
    e0 = np.nanmean(np.abs(T - V), 0)
    best_k, best = 0.0, None
    for k in (0.25, 0.5, 0.75, 1.0):
        e = loo(k)
        tot = np.nanmedian(np.where(use, np.minimum(e, e0), np.nan))
        if best is None or tot < best[0]:
            best_k, best = k, (tot, e)
    ek = best[1]
    gain = (e0 - ek) * 255
    keep = use & np.isfinite(gain) & (gain >= a0.min_gain)
    full = np.nanmean(D, 0)
    off = {tg.genes[j]: float(best_k * full[j]) for j in np.where(keep)[0]}
    json.dump({"sex": a0.sex, "people": n, "strength": best_k, "offset": off}, open(offset_path(a0.sex), "w"), indent=1)
    # report
    e_all0 = np.nanmedian(np.where(use, e0, np.nan)) * 255
    e_all1 = np.nanmedian(np.where(use, np.where(keep, ek, e0), np.nan)) * 255
    SIDE = ("depth", "forward", "profile", "angle", "roundness")
    side = np.array([any(k in g for k in SIDE) for g in tg.genes]) & use
    front = use & ~side
    split = lambda m: (np.nanmedian(np.where(m, e0, np.nan)) * 255,                      # noqa: E731
                       np.nanmedian(np.where(m, np.where(keep, ek, e0), np.nan)) * 255)
    L = [f"{n} reference {a0.sex} faces (photos + hand-made DNA), run through the normal pipeline (_final), "
         f"bridge share {a0.bridge_weight}.",
         "Error = distance of each slider from the hand-made DNA, 0-255, averaged over the people; leave-one-out.",
         f"Median over the {int(use.sum())} sliders: final {e_all0:.1f}  ->  with the correction {e_all1:.1f}",
         "  front sliders %.1f -> %.1f   side-profile sliders %.1f -> %.1f" % (*split(front), *split(side)),
         f"Corrected sliders: {int(keep.sum())} of {int(use.sum())} (only where leave-one-out helps by "
         f"{a0.min_gain}+); strength {best_k} of the average difference.", "",
         f"{'slider':38s} {'shift':>6s} {'final err':>9s} {'fixed err':>9s}"]
    for j in sorted(np.where(keep)[0], key=lambda j: -gain[j]):
        L.append(f"  {tg.genes[j]:36s} {off[tg.genes[j]] * 255:+6.0f} {e0[j] * 255:9.1f} {ek[j] * 255:9.1f}")
    rep = "\n".join(L)
    sfx = ("" if a0.sex == "female" else f"_{a0.sex}") + ("" if a0.bridge_weight == 0.5 else f"_bw{round(a0.bridge_weight * 100)}")
    open(os.path.join(HERE, "results", f"ref_offset_report{sfx}.txt"), "w").write(rep + "\n")
    print("\n" + rep)
    # side-by-side DNAs for screenshots (the shift here is the full one, learned with this person in it)
    side = os.path.join(HERE, "results", f"refs_{a0.sex}")
    shutil.rmtree(side, ignore_errors=True)
    for pid, (text, dna) in finals.items():
        d = os.path.join(side, pid)
        os.makedirs(d)
        shutil.copy(dna, os.path.join(d, f"{pid}_1_human.txt"))
        open(os.path.join(d, f"{pid}_2_final.txt"), "w", encoding="utf-8").write(text)
        open(os.path.join(d, f"{pid}_3_ref.txt"), "w", encoding="utf-8").write(apply(text, tg, off))
    shutil.rmtree(os.path.join(HERE, "results", work), ignore_errors=True)
    print(f"\nSaved {offset_path(a0.sex)} and results\\ref_offset_report{sfx}.txt.")
    print(f"Side by side (hand-made / final / corrected):  python autoshot.py folders results\\refs_{a0.sex}")


if __name__ == "__main__":
    main()
