"""Beard picker for men: matches the beard (or clean shave) in his photos to the game's beards, like hair.py.

The training screenshots are all clean-shaven, so each allowed beard (beard_list_male.txt, plus clean-shaven) is
first put on the same few training faces and screenshotted once (same faces for every beard, so only the beard
differs between them). The lower face (jaw, chin, mouth, cheeks - no
eyes, no hair) is read in grey (beard colour follows the hair colour) by CLIP two ways: picture numbers, and how
strongly it agrees with plain descriptions (stubble, full, long, moustache, goatee, sideburns ...). His photos are
read the same way; when his photos disagree (shaved in some, bearded in others) the majority wins.

    python beard.py shots                      (no game) one DNA per beard x 6 faces -> random_faces\\beard_gallery
    python autoshot.py folders random_faces\\beard_gallery        (game on Male, ~10 min)
    python beard.py gallery                    (no game, a few min) -> results\\beard_gallery_male.npz
    python beard.py apply --photos "C:\\CK3 Face Generator\\Male Samples" --run male_pilot
        -> results\\male_pilot_beard\\<person>\\<person>_beard1.txt, _beard2.txt, _beard3.txt
           (his _final face with the 1st / 2nd / 3rd best beard), beard_check.jpg and beard_picks.txt
        screenshot:  python autoshot.py folders results\\male_pilot_beard
"""
import argparse
import glob
import os
import sys

import numpy as np

HERE = (os.environ.get("CK3FIT_HOME") or os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import photo_to_dna as P          # noqa: E402
import hair as H                  # noqa: E402  (CLIP, grey, person photo handling)

SHOTS = os.path.join(HERE, "random_faces", "beard_gallery")
GALLERY = os.path.join(HERE, "results", "beard_gallery_male.npz")
NONE = "male_empty"               # clean-shaven
SHAVED_ATTR = 0                   # index of the clean-shaven-vs-beard description below

ATTR_BEARD = [   # (description, its opposite)
    ("a close-up photo of a clean-shaven man's chin and jaw", "a close-up photo of a man's chin and jaw with a beard"),
    ("a photo of a man with light stubble", "a photo of a clean-shaven man"),
    ("a photo of a man with a full thick beard", "a photo of a man with a short trimmed beard"),
    ("a photo of a man with a long beard reaching his chest", "a photo of a man with a short beard"),
    ("a photo of a man with a moustache", "a photo of a man without a moustache"),
    ("a photo of a man with a goatee on his chin only", "a photo of a man with a beard covering his cheeks"),
    ("a photo of a man with long sideburns", "a photo of a man without sideburns"),
    ("a photo of a man with mutton chops", "a photo of a man with a full beard"),
    ("a photo of a man with a braided beard", "a photo of a man with an unbraided beard"),
    ("a photo of a man with a pointed beard", "a photo of a man with a rounded beard"),
    ("a photo of a man with a bushy unkempt beard", "a photo of a man with a neatly groomed beard"),
]


class Clip(H.Clip):
    """hair.py's CLIP reader with the beard descriptions (pictures always in grey)"""
    def __init__(s, dev):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        s.torch, s.dev, s.sex, s.A = torch, dev, "male", ATTR_BEARD
        s.m = CLIPModel.from_pretrained(H.CLIP).to(dev).eval()
        s.p = CLIPProcessor.from_pretrained(H.CLIP)
        with torch.no_grad():
            t = s.p(text=[x for pr in s.A for x in pr], return_tensors="pt", padding=True).to(dev)
            T = s.m.text_projection(s.m.text_model(input_ids=t["input_ids"],
                                                   attention_mask=t["attention_mask"]).pooler_output)
            s.T = T / T.norm(dim=-1, keepdim=True)


def jaw_crop(rgb, L, size=224):
    """the lower face around a face (from its face points): cheeks, mouth, jaw, chin and just below; None if small"""
    import cv2
    le, re_ = L[33, :2], L[263, :2]
    iod = float(np.linalg.norm(re_ - le))
    if iod < 20:
        return None
    c = (le + re_) / 2
    chin = L[152, :2]
    x0, x1 = c[0] - 1.45 * iod, c[0] + 1.45 * iod
    y0, y1 = c[1] + 0.35 * iod, chin[1] + 0.75 * iod
    h, w = rgb.shape[:2]
    pad = int(max(0, -x0, -y0, x1 - w, y1 - h)) + 1
    img = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    crop = img[int(y0) + pad:int(y1) + pad, int(x0) + pad:int(x1) + pad]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def allowed():
    p = os.path.join(HERE, "beard_list_male.txt")
    if not os.path.exists(p):
        sys.exit("beard_list_male.txt not found.")
    names = sorted({l.strip() for l in open(p, encoding="utf-8") if l.strip() and not l.startswith("#")})
    return [NONE] + [b for b in names if b != NONE]


def shots(a):
    import collect_noise as CN
    src = sorted(glob.glob(os.path.join(HERE, "random_faces", "noise_male", "*.txt")))
    if len(src) < 10:
        sys.exit("No men's training faces in random_faces\\noise_male to put the beards on.")
    rng = np.random.default_rng(0)
    beards = allowed()
    pick = rng.permutation(len(src))
    n = 0
    for k, b in enumerate(beards):
        d = os.path.join(SHOTS, b)
        os.makedirs(d, exist_ok=True)
        for f in glob.glob(os.path.join(d, "*.txt")):
            os.remove(f)
        for j in range(a.per_beard):                   # the SAME faces for every beard: only the beard differs
            text = open(src[pick[j]], encoding="utf-8").read()
            text = CN.shave(text) if b == NONE else CN.set_beard(text, b)
            open(os.path.join(d, f"{b}_{j}.txt"), "w", encoding="utf-8").write(text)
            n += 1
    print(f"{n} DNAs ({len(beards)} beards incl. clean-shaven x {a.per_beard} faces) in {SHOTS}")
    print("Screenshot (game on Male):  python autoshot.py folders random_faces\\beard_gallery")


def gallery(a, clip, mesh):
    from PIL import Image
    S, E, A, N = [], [], [], []
    for b in allowed():
        crops = []
        for j in sorted(glob.glob(os.path.join(SHOTS, b, "*.jpg"))):
            rgb = np.asarray(Image.open(j).convert("RGB"))
            front = rgb[:, :int(rgb.shape[1] * 0.66)]              # the big front face on the left
            p, L = mesh(front, full=True)
            if p is not None:
                c = jaw_crop(front, L)
                if c is not None:
                    crops.append(c)
        if len(crops) < 2:
            print(f"  {b}: faces found on only {len(crops)} screenshots - skipped")
            continue
        e, at = H.read(clip, crops)
        S.append(b); E.append(e); A.append(at); N.append(len(crops))
        if b == NONE or len(S) == 2:
            Image.fromarray(np.concatenate([H.grey(c) for c in crops[:6]], 1)).save(
                os.path.join(HERE, "results", f"beard_crop_check_{len(S)}.jpg"))
    if len(S) < 5:
        sys.exit(f"Only {len(S)} beards have screenshots - run the autoshot step first.")
    E, A = np.array(E), np.array(A)
    np.savez(GALLERY, styles=np.array(S), emb=E, attr=A, n=np.array(N), game_mean=E.mean(0))
    print(f"{len(S)} beards (incl. clean-shaven) in {GALLERY}. Crop check: results\\beard_crop_check_*.jpg")


def person_crops(folder, mesh, max_turn=0.6):
    out = []
    for f in sorted(glob.glob(os.path.join(folder, "*"))):
        if not f.lower().endswith(P.EXTS):
            continue
        rgb = P.load_rgb(f)
        p, L = mesh(rgb, full=True)
        if p is None or abs(P.turn(p)) > max_turn:
            continue
        c = jaw_crop(rgb, L)
        if c is not None:
            out.append(c)
    return out


def read_person(clip, crops):
    """the majority look (shaved or bearded) over his photos -> mean picture numbers, mean word scores, kept/all"""
    E, A = clip(crops + [c[:, ::-1].copy() for c in crops])
    n = len(crops)
    shaved = (A[:n, SHAVED_ATTR] + A[n:, SHAVED_ATTR]) / 2 > 0.5
    keep = shaved if shaved.sum() * 2 > n else ~shaved
    if not keep.any():
        keep[:] = True
    k2 = np.concatenate([keep, keep])
    e = E[k2].mean(0)
    is_shaved = keep is shaved
    return e / np.linalg.norm(e), A[k2].mean(0), f"{int(keep.sum())}/{n} {'shaved' if is_shaved else 'bearded'}"


def load_gallery():
    if not os.path.exists(GALLERY):
        return None
    g = dict(np.load(GALLERY, allow_pickle=True))
    g["styles"] = [str(s) for s in g["styles"]]
    return g


def choose(g, e, at, kept, top=3):
    """his photos' vote decides beard or not (moustache-only beards leave the jaw bare and can otherwise beat
    clean-shaven on a shaved man). Shaved -> clean-shaven first, the best beards after it as alternatives;
    bearded -> only real beards compete. -> list of beard names"""
    _, _, sc = H.scores(g, e, at)
    none = g["styles"].index(NONE) if NONE in g["styles"] else None
    order = [int(k) for k in np.argsort(-sc) if int(k) != none]
    if none is not None and kept.endswith("shaved"):
        order = [none] + order
    return [g["styles"][k] for k in order[:top]]


def set_photo_mean(g, reads):
    """photos and game shots differ overall: compare differences from each side's own average. With 5+ men the
    average is worked out from them (and saved for later runs); otherwise the saved one is used."""
    if len(reads) >= 5:
        g["photo_mean"] = np.mean([e for e, _, _ in reads], 0)
        try:
            np.savez(GALLERY, **{k: v for k, v in g.items() if k != "styles"}, styles=np.array(g["styles"]))
        except Exception:
            pass
    return g


def apply(a, clip, mesh):
    import collect_noise as CN
    from PIL import Image, ImageDraw
    g = load_gallery()
    if g is None:
        sys.exit("Run  python beard.py gallery  first.")
    run = os.path.join(HERE, "results", a.run)
    out_root = os.path.join(HERE, "results", a.run + "_beard")
    people = sorted(os.path.basename(d) for d in glob.glob(os.path.join(run, "*")) if os.path.isdir(d))
    got = []
    for nm in people:
        fin = os.path.join(run, nm, f"{nm}_{a.file}.txt")
        folder = next((d for d in (os.path.join(a.photos, nm), os.path.join(a.photos, nm.replace("_", " ")))
                       if os.path.isdir(d)), None)
        if not os.path.exists(fin) or folder is None:
            print(f"  {nm}: no _{a.file}.txt or no photo folder - skipped")
            continue
        crops = person_crops(folder, mesh)
        if not crops:
            print(f"  {nm}: no usable photos - skipped")
            continue
        e, at, kept = read_person(clip, crops)
        got.append((nm, fin, crops, e, at, kept))
    if not got:
        sys.exit("Nobody to do.")
    g = set_photo_mean(g, [(x[3], x[4], x[5]) for x in got])
    lines = []
    for nm, fin, crops, e, at, kept in got:
        picks = choose(g, e, at, kept, a.top)
        top = [g["styles"].index(b) for b in picks]
        text = open(fin, encoding="utf-8").read()
        d = os.path.join(out_root, nm)
        os.makedirs(d, exist_ok=True)
        for f_ in glob.glob(os.path.join(d, "*")):
            os.remove(f_)
        for r, k in enumerate(top):
            b = g["styles"][k]
            t = CN.shave(text) if b == NONE else CN.set_beard(text, b)
            open(os.path.join(d, f"{nm}_beard{r + 1}.txt"), "w", encoding="utf-8").write(t)
        lines.append(f"{nm} (photos {kept}): " + ", ".join(f"{r + 1}. {g['styles'][k]}" for r, k in enumerate(top)))
        print("  " + lines[-1])
        tiles = [H.grey(c) for c in crops[:4]]
        for k in top:                                  # a game shot of each pick, if the gallery shots exist
            j = sorted(glob.glob(os.path.join(SHOTS, g["styles"][k], "*.jpg")))
            if j:
                rgb = np.asarray(Image.open(j[0]).convert("RGB"))
                front = rgb[:, :int(rgb.shape[1] * 0.66)]
                p, L = mesh(front, full=True)
                c = jaw_crop(front, L) if p is not None else None
                if c is not None:
                    tiles.append(H.grey(c))
        sh = Image.fromarray(np.concatenate(tiles, 1))
        dr = ImageDraw.Draw(sh)
        for r, k in enumerate(top):
            dr.text((224 * (min(4, len(crops)) + r) + 4, 4), f"{r + 1}. {g['styles'][k]}", fill=(255, 255, 0))
        sh.save(os.path.join(d, "beard_check.jpg"), quality=85)
    open(os.path.join(out_root, "beard_picks.txt"), "w").write("\n".join(lines) + "\n")
    print(f"\nWrote {len(lines)} men to {out_root}. Screenshot:  python autoshot.py folders results\\{a.run}_beard")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("shots")
    p.add_argument("--per-beard", type=int, default=6)
    sub.add_parser("gallery")
    p = sub.add_parser("apply")
    p.add_argument("--photos", required=True)
    p.add_argument("--run", required=True, help="results folder with <person>_final.txt (e.g. male_pilot)")
    p.add_argument("--file", default="final")
    p.add_argument("--top", type=int, default=3)
    a = ap.parse_args()
    if a.cmd == "shots":
        return shots(a)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    clip, mesh = Clip(dev), P.Mesh()
    {"gallery": gallery, "apply": apply}[a.cmd](a, clip, mesh)


if __name__ == "__main__":
    main()
