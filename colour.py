"""Skin, eye and hair colour measured straight from the pixels, instead of guessed by the network.

Both game screenshots and photos go through the SAME measuring code:
  - the face points locate cheeks + forehead (skin), the iris (eyes) and the band above the forehead (hair)
  - the whites of the eyes give each photo's colour balance and brightness, which is corrected
  - the conversion from measured colour to the game's colour settings is learned from the training screenshots,
    where the true settings are known (nearest look-alikes among ~45,000 faces)

    python colour.py train --data random_faces\\noise_female random_faces\\real_female
        (needs bridge2.py embed --geo first; writes results\\colour_model.npz and results\\colour_report.txt)
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = (os.environ.get("CK3FIT_HOME") or os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import train_reverse as tr        # noqa: E402

MODEL = os.path.join(HERE, "results", "colour_model.npz")
SKIN_PTS = [50, 280, 101, 330, 151, 108, 337, 205, 425]
EYES = [(33, 133, 159, 145), (362, 263, 386, 374)]        # corner, corner, upper lid, lower lid
NAMES = ("hair", "skin", "eye")                           # order of the game's colour settings (Targets.C)


def _disk(rgb, c, r):
    h, w = rgb.shape[:2]
    r = max(1.5, float(r))
    x0, x1 = int(max(0, c[0] - r)), int(min(w, c[0] + r + 1))
    y0, y1 = int(max(0, c[1] - r)), int(min(h, c[1] + r + 1))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 3), np.float32)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    m = (xx - c[0]) ** 2 + (yy - c[1]) ** 2 <= r * r
    return rgb[y0:y1, x0:x1][m].reshape(-1, 3).astype(np.float32)


def lum(px):
    return px @ np.array([0.299, 0.587, 0.114], np.float32)


LIT_PTS = [50, 280, 101, 330, 205, 425, 117, 346, 118, 347, 123, 352, 151, 108, 337, 9, 69, 299]


def measure(rgb, L, lit=False):
    """rgb: full image (uint8), L: (468,3) face points in that image -> 12 numbers (skin, eye, hair, sclera RGB 0-255;
    NaN where it couldn't be measured). lit=True adds 3 more: the well-lit skin (the brightest 70-92% band of the
    cheeks and forehead, no shadows or shine) - for how pale or dark someone is."""
    out = np.full(15 if lit else 12, np.nan, np.float32)
    iod = np.linalg.norm(L[263, :2] - L[33, :2])
    if iod < 8:
        return out
    # skin: small disks on cheeks and forehead; drop the darkest/brightest 20% (shadows, shine)
    px = np.concatenate([_disk(rgb, L[i, :2], 0.07 * iod) for i in SKIN_PTS])
    if len(px) > 20:
        lu = lum(px)
        lo, hi = np.percentile(lu, [20, 80])
        out[0:3] = np.median(px[(lu >= lo) & (lu <= hi)], 0)
    # eyes: iris disk between the lids; drop pupil/lashes (darkest) and whites/reflections (brightest)
    iris, sclera = [], []
    for a, b, up, lo_ in EYES:
        op = np.linalg.norm(L[up, :2] - L[lo_, :2])
        if op < 1.5:
            continue
        c = (L[up, :2] + L[lo_, :2]) / 2
        iris.append(_disk(rgb, c, 0.38 * op))
        for k in (a, b):                                   # whites: between each corner and the iris
            sclera.append(_disk(rgb, L[k, :2] * 0.55 + c * 0.45, 0.22 * op))
    if iris and sum(len(x) for x in iris) > 6:
        px = np.concatenate(iris)
        lu = lum(px)
        lo, hi = np.percentile(lu, [30, 85])
        sel = px[(lu >= lo) & (lu <= hi)]
        if len(sel):
            out[3:6] = np.median(sel, 0)
    if sclera and sum(len(x) for x in sclera) > 6:
        px = np.concatenate(sclera)
        sat = px.max(1) - px.min(1)
        lu = lum(px)
        keep = (lu >= np.percentile(lu, 60)) & (sat <= np.percentile(sat, 60))
        if keep.sum():
            out[9:12] = np.median(px[keep], 0)
    if lit:
        px = np.concatenate([_disk(rgb, L[i, :2], 0.08 * iod) for i in LIT_PTS])
        if len(px) > 30:
            Lp = lab(px)[:, 0]
            lo, hi = np.percentile(Lp, [70, 92])
            sel = px[(Lp >= lo) & (Lp <= hi)]
            if len(sel):
                out[12:15] = np.median(sel, 0)
    # hair: band above the forehead and the temples, minus anything skin-coloured
    up = L[10, :2] - L[152, :2]
    fh = np.linalg.norm(up)
    if fh > 10:
        u = up / fh
        cs = [L[10, :2] + u * fh * f for f in (0.10, 0.18, 0.26)] + \
             [L[i, :2] + u * fh * 0.12 for i in (54, 284, 103, 332)]
        px = np.concatenate([_disk(rgb, c, 0.06 * fh) for c in cs])
        if len(px) > 20:
            if not np.isnan(out[0]):
                d = np.linalg.norm(lab(px) - lab(out[0:3][None]), axis=1)
                px = px[d > 12]
            if len(px) > 10:
                out[6:9] = np.median(px, 0)
    return out


def lab(rgb):
    """sRGB 0-255 -> CIELAB"""
    c = np.asarray(rgb, np.float32) / 255
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]], np.float32)
    xyz = c @ M.T / np.array([0.9505, 1.0, 1.089], np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def balance(m, target):
    """Correct one measurement's colour balance and brightness so its eye-whites match the game's (capped)."""
    m = m.copy()
    if np.isnan(m[9:12]).any() or target is None:
        return m
    g = np.clip(target / np.maximum(m[9:12], 1.0), 0.6, 1.8)
    for k in range(3):
        m[3 * k:3 * k + 3] = np.clip(m[3 * k:3 * k + 3] * g, 0, 255)
    return m


def features(ms, target):
    """list of measurements (one per photo or screenshot) -> per colour: LAB of the balanced median, or NaN."""
    B = np.array([balance(m, target) for m in ms])
    out = []
    for k in range(3):
        v = B[:, 3 * k:3 * k + 3]
        v = v[~np.isnan(v).any(1)]
        out.append(lab(np.median(v, 0)) if len(v) else np.full(3, np.nan, np.float32))
    return out          # skin, eye, hair


def knn(X, Y, q, k=15):
    d = np.linalg.norm(X - q, axis=1)
    i = np.argpartition(d, k)[:k]
    w = 1 / (d[i] + 1.0)
    return (Y[i] * w[:, None]).sum(0) / w.sum()


def train(a):
    names, M = [], []
    for d in a.data:
        nm = [str(n) for n in np.load(os.path.join(d, tr.cache_name(224, "_names.npy")))]
        done = int(open(os.path.join(d, "_f3_done.txt")).read()) if os.path.exists(os.path.join(d, "_f3_done.txt")) else 0
        if done < len(nm) and not a.limit:
            sys.exit(f"{d}: the face-point embed run isn't finished - run it again to carry on.")
        n = min(done, a.limit) if a.limit else len(nm)
        names += [os.path.join(d, x) for x in nm[:n]]
        M.append(np.load(os.path.join(d, "_f3_col.npy"))[:n])
    M = np.concatenate(M)
    C = np.array([[q for k in ("hair_color", "skin_color", "eye_color") for q in tr.parse(open(n + ".txt", encoding="utf-8").read())[1].get(k, (np.nan, np.nan))]
                  for n in names], np.float32) / 255
    target = np.nanmedian(M[:, 9:12], 0)                 # the game's eye-white colour
    Bm = np.array([balance(m, target) for m in M])
    feats = {"skin": lab(Bm[:, 0:3]), "eye": lab(Bm[:, 3:6]), "hair": lab(Bm[:, 6:9])}
    tgt = {"hair": C[:, 0:2], "skin": C[:, 2:4], "eye": C[:, 4:6]}
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(names))
    te = idx[:max(5, len(idx) // 20)]
    lines = [f"{len(names)} screenshots. Held-out error in the game's colour settings (0-255; lower is better):"]
    save = {"target": target}
    for c in ("skin", "eye", "hair"):
        ok = ~np.isnan(feats[c]).any(1) & ~np.isnan(tgt[c]).any(1)
        tr_ok = np.setdiff1d(np.where(ok)[0], te)
        te_ok = np.intersect1d(np.where(ok)[0], te)
        X, Y = feats[c][tr_ok], tgt[c][tr_ok]
        if len(te_ok) and len(X) > 20:
            P = np.array([knn(X, Y, feats[c][i], min(15, len(X) - 1)) for i in te_ok[:2000]])
            e = np.abs(P - tgt[c][te_ok[:2000]]).mean(0) * 255
            g = np.abs(Y.mean(0) - tgt[c][te_ok[:2000]]).mean(0) * 255
            lines.append(f"  {c:5s} measured {e[0]:5.1f} {e[1]:5.1f}   (guessing {g[0]:5.1f} {g[1]:5.1f};"
                         f" found on {ok.mean():.0%} of faces)")
        save[c + "_X"], save[c + "_Y"] = feats[c][ok], tgt[c][ok]
    out = a.out or MODEL
    np.savez(out, **save)
    lines.append(f"Saved {out}")
    print("\n".join(lines))
    if not a.out:
        open(os.path.join(HERE, "results", "colour_report.txt"), "w").write("\n".join(lines))


CALIB = os.path.join(HERE, "results", "colour_calib.json")
PALETTES = os.path.join(HERE, "palette")                  # skin_palette.png, eye_palette.png, hair_palette.png
SESSION = os.path.join(HERE, "results", "colour_session.npz")   # colour_session.py (optional, cleaner)
SESSION_PARTS = ("skin", "eye")   # which colours take their palette looks from the session (hair: worse there)
REFS = os.path.join(HERE, "agot_dna2")                     # the hand-made DNAs: which palette areas people use
PARTS = {"skin": slice(0, 3), "eye": slice(3, 6), "hair": slice(6, 9)}
SKIN_KEEP = (0.4, 0.2, 0.2)       # share of a person's skin difference from average kept: lightness, red, yellow
# Skin lightness from the well-lit skin (not white-balanced: the eye whites are often shaded by the lids, which
# made pale people read as tan). Your reference people's photos average this lightness (LAB L) - women / men:
SKIN_LIT_CENTRE = {"female": 80.1, "male": 79.6}
SKIN_LIT_SLOPE = 0.8               # game lightness per unit of photo lightness away from that average
SKIN_BANDS = (8, 5.5, 3, 0, -3.5, -8, -14, -21)   # very pale ... dark, as lightness steps from the hand-made average
# Added to every band AFTER it is picked. 2026-11-06: Ryan judged 11 of 29 people too pale in game and 1 not pale
# enough; no photo measure told those people apart, so it's one general shift.
SKIN_SHIFT = -1.5
SKIN_BAND_NAMES = ("very pale", "pale", "fair", "light", "light-medium", "medium / olive", "brown", "dark")
EYE_SHIFT = (4.0, 0.0, -2.0)      # LAB; from your reference people (photo irises vs their hand-made eyes in game)
SKIN_PHOTO_CENTRE = (64.0, 9.8, 15.2)                      # average reference-person photo skin (LAB), if no calib


def robust(ms, target, part, hair_group=12.0, cut=2.5):
    """One colour for a person from all their photos, ignoring the odd ones out (LAB; and 'kept x/y' photos).
    hair: the biggest group of photos with the same hair colour wins (natural vs dyed, odd lighting)
    skin, eyes: photos far from the person's typical colour are dropped (lighting, contacts, odd eyes)"""
    ms, grey = drop_grey(ms)
    B = np.array([balance(np.asarray(m, np.float32), target) for m in ms])[:, PARTS[part]]
    B = B[np.isfinite(B).all(1)]
    if not len(B):
        return np.full(3, np.nan, np.float32), "0/0"
    L = lab(B.astype(np.float32))
    gnote = f", {grey} black-and-white left out" if grey else ""
    if len(L) < 3:
        return np.median(L, 0), f"{len(L)}/{len(L)}{gnote}"
    if part == "hair":
        best = None
        for i in range(len(L)):
            g = np.linalg.norm(L - L[i], axis=1) <= hair_group
            if best is None or g.sum() > best.sum():
                best = g
    else:
        d = np.linalg.norm(L - np.median(L, 0), axis=1)
        best = d <= cut * (np.median(d) + 1e-6)
    return np.median(L[best], 0), f"{int(best.sum())}/{len(L)}{gnote}"


def drop_grey(ms, chroma=4.0):
    """Leave out black-and-white (or nearly colourless) photos: their skin has no colour at all, which read as
    "very pale" skin and greyish hair/eyes. Returns (kept measurements, how many were dropped)."""
    keep = []
    for m in ms:
        sk = np.asarray(m, np.float32)[0:3]
        if np.isfinite(sk).all() and np.hypot(*lab(sk)[1:]) < chroma:
            continue
        keep.append(m)
    if not keep:                       # only black-and-white photos: use them anyway
        return list(ms), 0
    return keep, len(ms) - len(keep)


def lab2rgb(L):
    L = np.asarray(L, np.float64)
    fy = (L[0] + 16) / 116
    f = np.array([fy + L[1] / 500, fy, fy - L[2] / 200])
    xyz = np.where(f > 0.2069, f ** 3, (f - 16 / 116) / 7.787) * np.array([0.9505, 1.0, 1.089])
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    c = np.clip(np.linalg.solve(M, xyz), 0, 1)
    c = np.where(c > 0.0031308, 1.055 * c ** (1 / 2.4) - 0.055, 12.92 * c)
    return np.clip(np.round(c * 255), 0, 255).astype(int)


def hexc(L):
    return "#%02x%02x%02x" % tuple(lab2rgb(L))


def _poly(P):
    Lq, aq, bq = P[:, 0:1] / 50 - 1, P[:, 1:2] / 30, P[:, 2:3] / 30
    return np.hstack([np.ones_like(Lq), Lq, aq, bq, Lq * Lq, aq * aq, bq * bq, Lq * aq, Lq * bq, aq * bq])


class Look:
    """The game's colour palettes, and how each palette colour actually LOOKS once rendered.

    A palette pixel is not what you see in the Ruler Designer: the game's hair shader in particular shows a palette
    colour much lighter (a dark-brown pixel renders as dirty blond). So the look of each palette colour is learned
    from the training screenshots (where the colour setting is known and the rendered colour is measured), as a
    smooth function of the palette colour. Then a person's photo colour is matched to the palette colour whose
    rendered look is nearest, among the palette areas people actually use (near any hand-made DNA's colour)."""
    def __init__(s, data, palettes, refs, radius=15):
        s.ok = False
        files = {c: os.path.join(palettes, f"{c}_palette.png") for c in ("skin", "eye", "hair")}
        if not palettes or not all(os.path.exists(f) for f in files.values()):
            return
        from PIL import Image
        import glob
        real = {c: set() for c in files}
        for f in glob.glob(os.path.join(refs, "**", "*.txt"), recursive=True) if refs and os.path.isdir(refs) else []:
            try:
                cols = tr.parse(open(f, encoding="utf-8", errors="ignore").read())[1]
            except Exception:
                continue
            for c, key in (("hair", "hair_color"), ("skin", "skin_color"), ("eye", "eye_color")):
                if key in cols and None not in cols[key]:
                    real[c].add(tuple(int(x) for x in cols[key]))
        gy, gx = np.mgrid[0:256, 0:256]
        s.grid = np.stack([gx.ravel(), gy.ravel()], 1)
        s.look, s.use, s.fit = {}, {}, {}
        for c, f in files.items():
            pal = lab(np.asarray(Image.open(f).convert("RGB").resize((256, 256), Image.BILINEAR), np.float32))
            X, Y = data[c]
            xy = np.clip(np.round(Y * 255), 0, 255).astype(int)
            A = _poly(pal[xy[:, 1], xy[:, 0]])
            W = np.linalg.lstsq(A, X, rcond=None)[0]
            s.fit[c] = [float(np.corrcoef(A @ W[:, j], X[:, j])[0, 1]) for j in range(3)]
            s.look[c] = _poly(pal.reshape(-1, 3)) @ W
            # the palette pictures have thin stripes / edges that aren't part of the smooth colour field (the eye
            # palette has a grey stripe down the blue side): a pixel unlike its neighbours 6 left and right is skipped
            nb = (np.roll(pal, 6, 1) + np.roll(pal, -6, 1)) / 2
            odd = (np.linalg.norm(pal - nb, axis=2) > 8).ravel() | (s.grid[:, 0] < 6) | (s.grid[:, 0] > 249)
            s.centre = getattr(s, "centre", {})
            if len(real[c]):
                Rc = np.clip(np.array(sorted(real[c])), 0, 255)
                s.centre[c] = np.median(s.look[c][Rc[:, 1] * 256 + Rc[:, 0]], 0)   # typical hand-made look
            if len(real[c]) >= 5:
                R = np.array(sorted(real[c]), np.float32)
                d = np.full(len(s.grid), np.inf, np.float32)
                for i in range(0, len(R), 64):
                    d = np.minimum(d, np.sqrt(((s.grid[:, None, :] - R[None, i:i + 64]) ** 2).sum(2)).min(1))
                s.use[c] = (d <= radius) & ~odd
            else:
                s.use[c] = ~odd
        s.ok = True

    @staticmethod
    def from_file(path):
        """the palette looks, precomputed by make_app.py (plain numbers; no palette images needed)"""
        z = np.load(path)
        s = Look.__new__(Look)
        gy, gx = np.mgrid[0:256, 0:256]
        s.grid = np.stack([gx.ravel(), gy.ravel()], 1)
        s.look, s.use, s.fit, s.centre = {}, {}, {}, {}
        for c in ("skin", "eye", "hair"):
            s.look[c], s.use[c] = z[c + "_look"], z[c + "_use"]
            s.fit[c] = [float(v) for v in z[c + "_fit"]]
            if c + "_centre" in z.files:
                s.centre[c] = z[c + "_centre"]
        s.ok = True
        return s

    def pick(s, c, q, photo_centre=None, light=None):
        """q: the person's colour (LAB). Skin: how light a face measures in photos is mostly the lighting (on your
        reference people it varied 4x as much as their hand-made skin tones), so only part of the difference from
        the average photo is kept (SKIN_KEEP), around the average hand-made skin's look."""
        if c == "skin" and c in getattr(s, "centre", {}):
            pc = np.asarray(photo_centre if photo_centre is not None else SKIN_PHOTO_CENTRE, np.float32)
            q = s.centre[c] + np.array(SKIN_KEEP, np.float32) * (np.asarray(q, np.float32) - pc)
            if light is not None:                    # lightness set by the band instead
                q[0] = s.centre[c][0] + light
        if c == "eye":                              # irises measure darker and yellower in photos than in game
            q = np.asarray(q, np.float32) + np.array(EYE_SHIFT, np.float32)
        d = np.linalg.norm(s.look[c] - q, axis=1)
        d[~s.use[c]] = np.inf
        i = int(np.argmin(d))
        return s.grid[i] / 255.0, s.look[c][i]




class HairDirect:
    """Hair from the colour-session screenshots directly: every session shot is a palette spot whose rendered hair was
    measured. Each spot's look is smoothed over its nearest spots on the palette (measuring noise), and a person gets
    the spot whose look is nearest their hair colour. No formula over the palette pixels: that formula couldn't follow
    the hair shader (fit 0.65 on the session, and blondes came out ginger; the training-screenshot version turned dark
    hair olive / mid-brown)."""
    def __init__(s, X, Y, k=6):
        s.xy = np.asarray(Y, np.float32)
        X = np.asarray(X, np.float32)
        D = np.linalg.norm(s.xy[:, None] - s.xy[None], axis=2)
        nn = np.argsort(D, 1)[:, :k]
        s.look = X[nn].mean(1)
        loo = np.argsort(D + np.eye(len(D)) * 1e9, 1)[:, :k]        # the same, without the spot itself
        P = X[loo].mean(1)
        s.fit = [float(np.corrcoef(P[:, j], X[:, j])[0, 1]) for j in range(3)]

    def pick(s, q):
        i = int(np.argmin(np.linalg.norm(s.look - np.asarray(q, np.float32), axis=1)))
        return s.xy[i], s.look[i]


HAIR_MIN_FIT = 0.6     # below this the session's hair readings for that sex are noise


def session_hair(zs, sex, fallback=True):
    """The hair shader is the same for both sexes, so a sex whose own session readings are noisy uses the other's.
    2026-11-06: female fit 0.97 / 0.94 / 0.97, male 0.30 / 0.26 / 0.19 (the men's short template hair leaves too little
    hair in the measuring band), so men use the women's session shots."""
    X, Y = zs["hair_X"], zs["hair_Y"]
    if "sex" in zs.files and len(zs["sex"]) == len(X):
        m = zs["sex"] == sex
        X, Y = X[m], Y[m]
    h = HairDirect(X, Y) if len(X) >= 100 else None
    if fallback and (h is None or np.mean(h.fit) < HAIR_MIN_FIT):
        other = session_hair(zs, "female" if sex == "male" else "male", fallback=False)
        if other is not None and np.mean(other.fit) >= HAIR_MIN_FIT:
            print(f"  colour: {sex} hair uses the {'female' if sex == 'male' else 'male'} session shots "
                  f"(own fit {np.mean(h.fit) if h else 0:.2f})")
            return other
    return h


class Model:
    """kNN from measured colour (LAB) to the game's colour settings, over the training screenshots.
    With results\\colour_calib.json (colour_calib.py): photo colours are first moved onto the game renders' scale
    (photos are lit and exposed differently from the Ruler Designer), and only screenshots whose colour settings
    are near ones people actually use count as neighbours (no purple eyes or hair from the random training colours)."""
    def __init__(s, path=MODEL, calib=CALIB, palettes=PALETTES, refs=REFS, session=None, use_session=True):
        z = np.load(path)
        s.target = z["target"]
        s.d = {c: (z[c + "_X"], z[c + "_Y"]) for c in ("skin", "eye", "hair")}
        s.last = None
        s.sex = "male" if "male" in os.path.basename(path).replace("female", "") else "female"
        data = s.d
        sp = session or SESSION
        s.session = False
        s.hair = None
        if use_session and sp and os.path.exists(sp):  # colour_session.py: clean colour-only screenshots
            zs = np.load(sp)
            s.hair = session_hair(zs, s.sex)
            if all(len(zs[c + "_X"]) >= 100 for c in ("skin", "eye", "hair")):
                # hair stays on the training screenshots: the session's fixed hairstyle predicted hair looks worse
                # (fit 0.65 vs 0.88 / 0.80) and turned blondes ginger. Skin and eyes improved (0.96), so they use it.
                data = {c: (zs[c + "_X"], zs[c + "_Y"]) if c in SESSION_PARTS else data[c]
                        for c in ("skin", "eye", "hair")}
                s.session = True
        _lf = os.path.join(HERE, "results", f"colour_look_{s.sex}.npz")
        s.look = Look.from_file(_lf) if os.path.exists(_lf) else Look(data, palettes, refs)
        if not s.look.ok:
            s.look = None
        s._calib = json.load(open(calib)) if calib and os.path.exists(calib) else None
        s.cal = None if s.look else (json.load(open(calib)) if calib and os.path.exists(calib) else None)
        if s.cal:
            for c in s.d:
                X, Y = s.d[c]
                real = np.array(s.cal["real"][c], np.float32) / 255
                if len(real):                        # nearest used colour within 0.05 (13 of 255) per neighbour
                    near = np.zeros(len(Y), bool)
                    for i in range(0, len(Y), 4096):
                        dd = np.linalg.norm(Y[i:i + 4096, None, :] - real[None], axis=2).min(1)
                        near[i:i + 4096] = dd <= 0.05
                    if near.sum() > 50:
                        s.d[c] = (X[near], Y[near])

    def to_render(s, c, f):
        """photo LAB -> the same colour as the game would render it (mean / spread matched, per channel)"""
        if not s.cal or c not in s.cal["map"]:
            return f
        m = s.cal["map"][c]
        return (f - np.array(m["mu_p"])) / np.array(m["sd_p"]) * np.array(m["sd_r"]) + np.array(m["mu_r"])

    def predict(s, ms, fallback, parts=("skin", "eye", "hair"), blend=1.0, photo=True):
        """ms: measurements from a person's photos; fallback: the network's 6 colour values (0-1) ->
        6 colour values (hair x,y, skin x,y, eye x,y), measured where possible."""
        out = np.array(fallback, np.float32).copy()
        if s.look is not None and photo:              # palette route: robust colours per person, then the palette
            s.last = {}
            ms, s.grey = drop_grey(ms)
            if s.grey:
                print(f"  colour: {s.grey} black-and-white photo(s) left out")
            for c, o in (("hair", 0), ("skin", 2), ("eye", 4)):
                if c not in parts:
                    continue
                q, kept = robust(ms, s.target, c)
                if np.isnan(q).any():
                    continue
                pc = (s._calib or {}).get("map", {}).get(c, {}).get("mu_p") if c == "skin" else None
                band = None
                lit = np.array([m[12:15] for m in ms if len(m) >= 15], np.float32) if c == "skin" else np.zeros((0, 3))
                lit = lit[np.isfinite(lit).all(1)] if len(lit) else lit
                if len(lit):                          # lightness from the well-lit skin, in bands
                    Ll = lab(lit)[:, 0]
                    d = np.abs(Ll - np.median(Ll))
                    pl = float(np.median(Ll[d <= 2.5 * (np.median(d) + 1e-6)]))
                    off = SKIN_LIT_SLOPE * (pl - SKIN_LIT_CENTRE.get(s.sex, 80.0))
                    b = int(np.argmin([abs(off - x) for x in SKIN_BANDS]))
                    band = f"{SKIN_BAND_NAMES[b]} (well-lit skin lightness {pl:.1f})"
                    xy, look = s.look.pick(c, q, pc, light=SKIN_BANDS[b] + SKIN_SHIFT)
                elif c == "hair" and s.hair is not None:
                    xy, look = s.hair.pick(q)
                    xy = xy.copy()
                else:
                    xy, look = s.look.pick(c, q, pc)
                out[o:o + 2] = (1 - blend) * out[o:o + 2] + blend * xy
                s.last[c] = {"photo": hexc(q), "kept": kept, "coord": [int(round(v * 255)) for v in xy],
                             "game": hexc(look)}
                if band:
                    s.last[c]["band"] = band
            return np.clip(out, 0, 1)
        f = dict(zip(("skin", "eye", "hair"), features(ms, s.target)))
        for c, o in (("hair", 0), ("skin", 2), ("eye", 4)):
            if c not in parts:
                continue
            X, Y = s.d[c]
            q = s.to_render(c, f[c]) if photo else f[c]
            if not np.isnan(q).any() and len(X) > 20:
                out[o:o + 2] = (1 - blend) * out[o:o + 2] + blend * knn(X, Y, q, min(15, len(X) - 1))
        return np.clip(out, 0, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default="")
    a = ap.parse_args()
    train(a)


if __name__ == "__main__":
    main()
