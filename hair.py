"""Hairstyle picker: matches the hair in her photos to the game's hairstyles.

How it learns what each hairstyle looks like: your training screenshots (random_faces\\noise_female, real_female)
already have a random hairstyle each, written in their DNA (custom_hair=...). For every allowed hairstyle
(hair_list_female.txt) it takes up to --per-style of those screenshots, crops the head the same way as a photo
(from the face points, with the lower face greyed out so only the hair counts), and reads it two ways with CLIP
(a general picture-and-text model, open weights):
    picture  - CLIP's picture numbers, averaged per hairstyle
    words    - how strongly CLIP agrees with plain descriptions (long / short, down / up, bun, braids, curly,
               wavy, bangs, parting, pulled back, ponytail, bob, ears covered), averaged per hairstyle
Her photos are read the same way and compared with every hairstyle. No game needed for any of this.

    python -m pip install transformers            (already there if celeba_prep.py ran)
    python hair.py gallery                        (once, ~10-20 min) -> results\\hair_gallery.npz
    python hair.py test                           (optional) how often it finds the hairstyle chosen by hand
                                                  for your reference women -> results\\hair_test_report.txt
    python hair.py apply --photos "C:\\CK3 Face Generator\\Samples" --run photos_colfilter
        -> results\\photos_colfilter_hair\\<person>\\<person>_hair1.txt, _hair2.txt, _hair3.txt
           (her _final face with the 1st / 2nd / 3rd best hairstyle) and hair_picks.txt
        screenshot:  python autoshot.py folders results\\photos_colfilter_hair
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

HERE = (os.environ.get("CK3FIT_HOME") or os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import photo_to_dna as P          # noqa: E402

CLIP = "openai/clip-vit-large-patch14"
GALLERY = os.path.join(HERE, "results", "hair_gallery.npz")          # women (the first one made)


def gallery_path(sex="female"):
    return GALLERY if sex == "female" else os.path.join(HERE, "results", f"hair_gallery_{sex}.npz")
HAIR_RE = re.compile(r"custom_hair=(\S+)")
ATTR = [   # (description, its opposite) -> how much CLIP prefers the first, 0-1
    ("a photo of a woman with long hair falling past her shoulders", "a photo of a woman with short hair"),
    ("a photo of a woman with her hair worn down and loose", "a photo of a woman with her hair tied up"),
    ("a photo of a woman with her hair in a bun or an updo", "a photo of a woman with her hair down"),
    ("a photo of a woman with braided hair", "a photo of a woman with unbraided hair"),
    ("a photo of a woman with curly hair", "a photo of a woman with straight hair"),
    ("a photo of a woman with wavy hair", "a photo of a woman with straight hair"),
    ("a photo of a woman with bangs covering her forehead", "a photo of a woman with a bare forehead"),
    ("a photo of a woman with a middle part", "a photo of a woman with a side part"),
    ("a photo of a woman with her hair pulled back tightly", "a photo of a woman with loose, full hair"),
    ("a photo of a woman with a ponytail", "a photo of a woman without a ponytail"),
    ("a photo of a woman with a bob haircut", "a photo of a woman with long hair"),
    ("a photo of a woman with hair covering her ears", "a photo of a woman with her ears showing"),
]


ATTR_MALE = [   # men's hair: length, cut, fringe and hairline matter more than updos and braids
    ("a photo of a man with very short cropped hair", "a photo of a man with medium-length hair"),
    ("a photo of a man with hair down to his shoulders", "a photo of a man with short hair"),
    ("a photo of a man with a shaved head or buzz cut", "a photo of a man with a full head of hair"),
    ("a photo of a bald or balding man", "a photo of a man with a full head of hair"),
    ("a photo of a man with a receding hairline", "a photo of a man with a low hairline"),
    ("a photo of a man with a fringe of hair over his forehead", "a photo of a man with his forehead uncovered"),
    ("a photo of a man with hair combed back off his forehead", "a photo of a man with hair falling forward"),
    ("a photo of a man with a side part", "a photo of a man with no parting"),
    ("a photo of a man with curly hair", "a photo of a man with straight hair"),
    ("a photo of a man with wavy hair", "a photo of a man with straight hair"),
    ("a photo of a man with messy tousled hair", "a photo of a man with neatly styled hair"),
    ("a photo of a man with hair covering his ears", "a photo of a man with his ears showing"),
    ("a photo of a man with hair tied back or braided", "a photo of a man with loose hair"),
    ("a photo of a man with short sides and longer hair on top", "a photo of a man with even-length hair"),
]


def attrs(sex):
    return ATTR if sex == "female" else ATTR_MALE


def grey(c):
    """men: hair judged without its colour (game hair colours are random in the screenshots)"""
    g = (c.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)).astype(np.uint8)
    return np.repeat(g[..., None], 3, 2)


class Clip:
    def __init__(s, dev, sex="female"):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        s.torch, s.dev = torch, dev
        s.m = CLIPModel.from_pretrained(CLIP).to(dev).eval()
        s.p = CLIPProcessor.from_pretrained(CLIP)
        with torch.no_grad():
            s.A = attrs(sex)
            s.sex = sex
            words = [x for pr in s.A for x in pr]
            t = s.p(text=words, return_tensors="pt", padding=True).to(dev)
            # pooled output -> projection, done by hand: works the same on old and new transformers versions
            T = s.m.text_projection(s.m.text_model(input_ids=t["input_ids"],
                                                   attention_mask=t["attention_mask"]).pooler_output)
            s.T = T / T.norm(dim=-1, keepdim=True)

    def __call__(s, imgs):
        """list of HxWx3 uint8 -> (n, D) unit picture numbers, (n, len(ATTR)) word scores"""
        torch = s.torch
        E, A = [], []
        for i in range(0, len(imgs), 32):
            with torch.no_grad():
                batch = imgs[i:i + 32] if s.sex == "female" else [grey(c) for c in imgs[i:i + 32]]
                x = s.p(images=batch, return_tensors="pt").to(s.dev)
                e = s.m.visual_projection(s.m.vision_model(pixel_values=x["pixel_values"]).pooler_output)
                e = e / e.norm(dim=-1, keepdim=True)
                lg = 100 * e @ s.T.T
                pr = lg.view(len(e), len(s.A), 2).softmax(-1)[..., 0]
            E.append(e.float().cpu().numpy())
            A.append(pr.float().cpu().numpy())
        return np.concatenate(E), np.concatenate(A)


def head_crop(rgb, L, size=224):
    """the head and hair around a face (from its face points), lower face greyed out; None if too small"""
    import cv2
    le, re_ = L[33, :2], L[263, :2]
    iod = float(np.linalg.norm(re_ - le))
    if iod < 20:
        return None
    img = rgb.copy()
    brow_y = min(L[105, 1], L[334, 1])
    low = L[L[:, 1] > brow_y, :2].astype(np.int32)          # the face below the brows: not hair
    if len(low) >= 3:
        cv2.fillConvexPoly(img, cv2.convexHull(low), (128, 128, 128))
    c = (le + re_) / 2
    x0, x1 = c[0] - 2.3 * iod, c[0] + 2.3 * iod
    y0, y1 = c[1] - 2.2 * iod, c[1] + 2.4 * iod
    h, w = img.shape[:2]
    pad = int(max(0, -x0, -y0, x1 - w, y1 - h)) + 1
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    crop = img[int(y0) + pad:int(y1) + pad, int(x0) + pad:int(x1) + pad]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def allowed(sex="female"):
    p = os.path.join(HERE, f"hair_list_{sex}.txt")
    if not os.path.exists(p):
        sys.exit(f"hair_list_{sex}.txt not found.")
    return sorted({l.strip() for l in open(p, encoding="utf-8") if l.strip() and not l.startswith("#")})


def read(clip, crops):
    """crops (and their mirror images) -> mean unit picture numbers, mean word scores"""
    both = crops + [c[:, ::-1].copy() for c in crops]
    E, A = clip(both)
    e = E.mean(0)
    return e / np.linalg.norm(e), A.mean(0)


def person_crops(folder, mesh, max_turn=0.6):
    out = []
    for f in sorted(glob.glob(os.path.join(folder, "*"))):
        if not f.lower().endswith(P.EXTS):
            continue
        rgb = P.load_rgb(f)
        p, L = mesh(rgb, full=True)
        if p is None or abs(P.turn(p)) > max_turn:
            continue
        c = head_crop(rgb, L)
        if c is not None:
            out.append(c)
    return out


def gallery(a, clip, mesh):
    styles = allowed(a.sex)
    GALLERY = gallery_path(a.sex)
    by = defaultdict(list)
    for d in a.data:
        d = d if os.path.isabs(d) else os.path.join(HERE, d)
        for t in sorted(glob.glob(os.path.join(d, "*.txt"))):
            j = t[:-4] + ".jpg"
            if not os.path.exists(j):
                continue
            m = HAIR_RE.search(open(t, encoding="utf-8", errors="ignore").read())
            if m and m.group(1) in styles:
                by[m.group(1)].append(j)
    if not by:
        sys.exit(f"No training screenshots with hairstyles found in {a.data}.")
    rng = np.random.default_rng(0)
    from PIL import Image
    S, E, A, N = [], [], [], []
    for k, s in enumerate(styles):
        files = by.get(s, [])
        if len(files) < 5:
            print(f"  {s}: only {len(files)} screenshots - skipped")
            continue
        files = [files[i] for i in rng.permutation(len(files))[:a.per_style]]
        crops = []
        for j in files:
            rgb = np.asarray(Image.open(j).convert("RGB"))
            front = rgb[:, :int(rgb.shape[1] * 0.66)]              # the big front face on the left
            p, L = mesh(front, full=True)
            if p is not None:
                c = head_crop(front, L)
                if c is not None:
                    crops.append(c)
        if len(crops) < 5:
            print(f"  {s}: faces found on only {len(crops)} screenshots - skipped")
            continue
        e, at = read(clip, crops)
        S.append(s); E.append(e); A.append(at); N.append(len(crops))
        print(f"  {k + 1}/{len(styles)} {s}: {len(crops)} screenshots")
        if k == 0:
            Image.fromarray(np.concatenate(crops[:6], 1)).save(os.path.join(HERE, "results", "hair_crop_check.jpg"))
    E, A = np.array(E), np.array(A)
    old = dict(np.load(GALLERY, allow_pickle=True)) if os.path.exists(GALLERY) else {}
    np.savez(GALLERY, styles=np.array(S), emb=E, attr=A, n=np.array(N), game_mean=E.mean(0),
             **{k: old[k] for k in ("photo_mean", "weights") if k in old})
    print(f"\n{len(S)} hairstyles in {GALLERY} (a look at the head crops: results\\hair_crop_check.jpg)")


def load_gallery(sex="female"):
    path = gallery_path(sex)
    if not os.path.exists(path):
        sys.exit(f"Run  python hair.py gallery --sex {sex}  first.")
    g = dict(np.load(path, allow_pickle=True))
    g["styles"] = [str(s) for s in g["styles"]]
    return g


def scores(g, e, at):
    """per hairstyle: picture score, word score, combined (higher = closer)"""
    pm = g.get("photo_mean")
    ep = e - pm if pm is not None else e
    G = g["emb"] - g["game_mean"]
    si = (G @ ep) / (np.linalg.norm(G, axis=1) * np.linalg.norm(ep) + 1e-9)
    sa = -np.linalg.norm(g["attr"] - at, axis=1)
    z = lambda x: (x - x.mean()) / (x.std() + 1e-9)          # noqa: E731
    w = g.get("weights", np.array([0.5, 0.5]))
    return si, sa, w[0] * z(si) + w[1] * z(sa)


def test(a, clip, mesh):
    import calibrate_refs as CR
    g = load_gallery(a.sex)
    GALLERY = gallery_path(a.sex)
    who = "women" if a.sex == "female" else "men"
    refs = a.refs if os.path.isabs(a.refs) else os.path.join(HERE, a.refs)
    rows, embs = [], []
    for d in (a.photos or []):                        # extra people (no hand-picked hairstyle): photo average only
        for f in sorted(x for x in glob.glob(os.path.join(d, "*")) if os.path.isdir(x)):
            crops = person_crops(f, mesh)
            if crops:
                embs.append(read(clip, crops)[0])
    n_extra = len(embs)
    for pid, folder, dna in CR.find_pairs(refs, a.sex):
        m = HAIR_RE.search(open(dna, encoding="utf-8", errors="ignore").read())
        crops = person_crops(folder, mesh)
        if not m or not crops:
            continue
        e, at = read(clip, crops)
        rows.append((pid, m.group(1), e, at))
        embs.append(e)
    if not rows:
        sys.exit(f"No reference {who} with photos found.")
    photo_mean = np.mean(embs, 0)
    L = [f"{len(rows)} reference {who} with photos (+ {n_extra} more people for the photo average). "
         f"Rank of the hairstyle chosen by hand among {len(g['styles'])} "
         "(1 = best); 'leave-one-out' = the photo average is worked out without her.", ""]
    usable = [r for r in rows if r[1] in g["styles"]]
    from collections import Counter
    common = Counter(r[1] for r in usable).most_common(1)
    L.append(f"{len(usable)} have a hand-picked hairstyle in the allowed list"
             + (f" ({common[0][1]} of them {common[0][0]}, the most common)." if common else "."))
    res = {k: [] for k in ("picture", "words", "both 50/50", "picture 75%", "words 75%")}
    W = {"both 50/50": (0.5, 0.5), "picture 75%": (0.75, 0.25), "words 75%": (0.25, 0.75)}
    for i, (pid, h, e, at) in enumerate(usable):
        gi = dict(g)
        gi["photo_mean"] = (photo_mean * len(embs) - e) / (len(embs) - 1)
        si, sa, _ = scores(gi, e, at)
        k = g["styles"].index(h)
        rk = lambda s: int((s > s[k]).sum()) + 1               # noqa: E731
        res["picture"].append(rk(si))
        res["words"].append(rk(sa))
        z = lambda x: (x - x.mean()) / (x.std() + 1e-9)        # noqa: E731
        for nm, (wi, wa) in W.items():
            res[nm].append(rk(wi * z(si) + wa * z(sa)))
    n_ = len(g["styles"])
    L += ["", f"{'method':12s} {'median rank':>11s} {'top 1':>6s} {'top 5':>6s} {'top 10':>7s}   "
              f"(by chance: median {n_ // 2}, top 5 {500 / n_:.0f}%, top 10 {1000 / n_:.0f}%)"]
    for nm, r in res.items():
        r = np.array(r)
        if len(r):
            L.append(f"{nm:12s} {np.median(r):11.0f} {np.mean(r == 1) * 100:5.0f}% {np.mean(r <= 5) * 100:5.0f}% "
                     f"{np.mean(r <= 10) * 100:6.0f}%")
    best = min(W, key=lambda k: np.median(res[k]) if res[k] else 1e9)
    g2 = dict(np.load(GALLERY, allow_pickle=True))
    g2["photo_mean"] = photo_mean
    g2["weights"] = np.array(W[best])
    np.savez(GALLERY, **g2)
    L += ["", f"Saved the photo average and the '{best}' mix for 'apply'.",
          "Caution: this only means something if the hand-picked hairstyles were chosen to look like the photos."]
    rep = "\n".join(L)
    print("\n" + rep)
    open(os.path.join(HERE, "results", "hair_test_report.txt" if a.sex == "female" else f"hair_test_report_{a.sex}.txt"),
         "w").write(rep + "\n")


def preview(style):
    """a screenshot of this hairstyle, if 'collect_noise.py hairs' shots exist"""
    for d in ("hair_shots", os.path.join("random_faces", "hairs_female"), os.path.join("random_faces", "screen_male", "hairs")):
        for ext in (".jpg", ".png"):
            f = os.path.join(HERE, d, style + ext)
            if os.path.exists(f):
                return f
    return None


def apply(a, clip, mesh):
    from PIL import Image, ImageDraw
    g = load_gallery(a.sex)
    run = os.path.join(HERE, "results", a.run)
    out_root = os.path.join(HERE, "results", a.run + "_hair")
    people = sorted(os.path.basename(d) for d in glob.glob(os.path.join(run, "*")) if os.path.isdir(d))
    lines = []
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
        e, at = read(clip, crops)
        _, _, sc = scores(g, e, at)
        top = np.argsort(-sc)[:a.top]
        text = open(fin, encoding="utf-8").read()
        d = os.path.join(out_root, nm)
        os.makedirs(d, exist_ok=True)
        for f_ in glob.glob(os.path.join(d, "*")):
            os.remove(f_)
        for r, k in enumerate(top):
            open(os.path.join(d, f"{nm}_hair{r + 1}.txt"), "w", encoding="utf-8").write(
                HAIR_RE.sub(f"custom_hair={g['styles'][k]}", text))
        lines.append(f"{nm}: " + ", ".join(f"{r + 1}. {g['styles'][k]}" for r, k in enumerate(top)))
        print("  " + lines[-1])
        # sheet: what the picker saw (her head crops) + previews of the picks, if hairstyle shots exist
        tiles = crops[:4] + [np.asarray(Image.open(p).convert("RGB").resize((224, 224)))
                             for p in (preview(g["styles"][k]) for k in top) if p]
        sh = Image.fromarray(np.concatenate(tiles, 1))
        dr = ImageDraw.Draw(sh)
        for r, k in enumerate(top):
            dr.text((224 * (min(4, len(crops)) + r) + 4, 4), f"{r + 1}. {g['styles'][k]}", fill=(255, 255, 0))
        sh.save(os.path.join(d, "hair_check.jpg"), quality=85)
    open(os.path.join(out_root, "hair_picks.txt"), "w").write("\n".join(lines) + "\n")
    print(f"\nWrote {len(lines)} people to {out_root}. Screenshot:  python autoshot.py folders results\\{a.run}_hair")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("gallery")
    p.add_argument("--data", nargs="+", default=None,
                   help="default: random_faces\\noise_female + real_female (women), random_faces\\noise_male (men)")
    p.add_argument("--per-style", type=int, default=150)
    p = sub.add_parser("test")
    p.add_argument("--refs", default="agot_dna2")
    p.add_argument("--photos", nargs="*", default=[],
                   help="extra folders of person folders, used only for the photo average (e.g. your Samples)")
    p = sub.add_parser("apply")
    p.add_argument("--photos", required=True)
    p.add_argument("--run", required=True, help="results folder with <person>_final.txt (e.g. photos_colfilter)")
    p.add_argument("--file", default="final")
    p.add_argument("--top", type=int, default=3)
    for p_ in sub.choices.values():
        p_.add_argument("--sex", choices=["female", "male"], default="female")
    a = ap.parse_args()
    if a.cmd == "gallery" and a.data is None:
        a.data = [os.path.join("random_faces", "noise_male")] if a.sex == "male" else \
            [os.path.join("random_faces", "noise_female"), os.path.join("random_faces", "real_female")]
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    clip, mesh = Clip(dev, a.sex), P.Mesh()
    {"gallery": gallery, "test": test, "apply": apply}[a.cmd](a, clip, mesh)


if __name__ == "__main__":
    main()
