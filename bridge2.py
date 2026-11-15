"""Bridge, second round: more face-recognition information for the bridge.

  arc   - the 512 numbers from ArcFace (InsightFace buffalo_l, w600k_r50): what the current bridge uses
  glint - 512 numbers from a second face-recognition model trained on different faces (InsightFace antelopev2,
          glintr100) - a second, independent description of the same face
  mid   - ArcFace's last detailed layer before it boils the face down to 512 numbers (7x7 grid of 512 channels),
          pooled to 2x2: keeps rough layout, so more shape - and more of the photo's quirks

Both models run on the graphics card through PyTorch (pip install onnx onnx2torch). Free for non-commercial use.

    python bridge2.py embed --data random_faces\\noise_female random_faces\\real_female      (~2 h, no game; resumes)
    python bridge2.py train --data random_faces\\noise_female random_faces\\real_female      (~30 min, 2 bridges)
        -> results\\bridge_arc_glint.pt and results\\bridge_arc_glint_mid.pt, report in results\\bridge2_report.txt
    python photo_to_dna.py --photos ... --bridge results\\bridge_arc_glint.pt

  facenet - FaceNet (VGGFace2), a third face-recognition model; the best automatic judge on the eyes test.
    python bridge2.py embed --feats facenet --data ...    (adds it to an existing embed; resumes)
    then train with e.g.  --variants arc+glint+mid+geo+facenet
"""
import argparse
import os
import sys
import time
import zipfile

import numpy as np

HERE = (os.environ.get("CK3FIT_HOME") or os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import train_reverse as tr        # noqa: E402
import bridge as B1                # noqa: E402  (face crop, photo-like changes, the small network)

ADIR = os.path.join(HERE, "results", "arcface")
GLINT = os.path.join(ADIR, "glintr100.onnx")
GLINT_URL = "https://github.com/deepinsight/insightface/releases/download/model-zoo/antelopev2.zip"
GEO_RAW, GEO_K = 468 * 3, 64                  # face points -> their 64 main ways of varying
DIMS = {"arc": 512, "glint": 512, "mid": 2048, "geo": GEO_K, "facenet": 512}
IMAGE_TYPES = ("arc", "glint", "mid", "facenet")   # picture features (geo comes from face points)
FIRST_ROUND = ("arc", "glint", "mid")             # what plain 'embed' writes; facenet is added with --feats facenet
K = 3                                   # photo-like versions per face (plus the clean one)


class Feats:
    def __init__(s, types=("arc", "glint", "mid"), ck=None):
        import torch
        try:
            import onnx
            import onnx2torch
        except Exception:
            sys.exit("Please run:  python -m pip install onnx onnx2torch")
        s.torch, s.types, s.ck = torch, list(types), ck
        s.dev = "cuda" if torch.cuda.is_available() else "cpu"
        if not os.path.exists(B1.MODEL):              # ArcFace pack (same download as bridge.py)
            B1.ArcFace()
        if "glint" in s.types and not os.path.exists(GLINT):
            import urllib.request
            z = os.path.join(ADIR, "antelopev2.zip")
            print("Downloading InsightFace's antelopev2 pack (~360 MB, once) ...")
            urllib.request.urlretrieve(GLINT_URL, z)
            with zipfile.ZipFile(z) as f:
                name = next(n for n in f.namelist() if n.endswith("glintr100.onnx"))
                open(GLINT, "wb").write(f.read(name))
            os.remove(z)
        s.arc = onnx2torch.convert(onnx.load(B1.MODEL)).to(s.dev).eval()
        s.glint = onnx2torch.convert(onnx.load(GLINT)).to(s.dev).eval() if "glint" in s.types else None
        s.img_types = [t for t in s.types if t in IMAGE_TYPES]
        s.facenet = None
        if "facenet" in s.types:                      # FaceNet (VGGFace2): python -m pip install facenet-pytorch --no-deps
            try:
                from facenet_pytorch import InceptionResnetV1
            except Exception:
                sys.exit("Please run:  python -m pip install facenet-pytorch --no-deps")
            s.facenet = InceptionResnetV1(pretrained="vggface2").to(s.dev).eval()
        s.mid = None
        if "mid" in s.types:                          # ArcFace's layer just before it flattens to 512 numbers
            flat = next(n for n in s.arc.graph.nodes if "Flatten" in str(n.target))
            mod = dict(s.arc.named_modules())[flat.args[0].target]
            mod.register_forward_hook(lambda m, i, o: setattr(s, "_mid", o))

    crop = B1.ArcFace.crop
    five = staticmethod(B1.ArcFace.five)

    def all(s, crops):
        """list of 112x112x3 RGB uint8 -> {type: (n, dim) unit vectors}"""
        torch = s.torch
        x = torch.from_numpy((np.stack(crops).astype(np.float32) - 127.5) / 127.5).permute(0, 3, 1, 2).to(s.dev)
        out = {}
        with torch.no_grad():
            if any(t in s.types for t in ("arc", "mid")):
                out["arc"] = s.arc(x)
            if s.facenet is not None:                 # same aligned face, 160 px, FaceNet's own scaling
                x160 = torch.nn.functional.interpolate(x * 127.5 / 128, size=(160, 160), mode="bilinear",
                                                       align_corners=False)
                out["facenet"] = s.facenet(x160)
            if "mid" in s.types:
                out["mid"] = torch.nn.functional.adaptive_avg_pool2d(s._mid, 2).flatten(1)
            if s.glint is not None:
                out["glint"] = s.glint(x)
        return {k: (v / v.norm(dim=1, keepdim=True).clamp(min=1e-6)).float().cpu().numpy()
                for k, v in out.items() if k in s.types}

    def __call__(s, crops):
        """one vector per crop = this bridge's image feature types side by side"""
        d = s.all(crops)
        return np.concatenate([d[t] for t in s.img_types], 1)

    def photo_vec(s, rgb, p, L, view):
        """one photo -> this bridge's input numbers (image types from the face crop, geo from the face points)"""
        d = s.all([s.crop(rgb, p)]) if s.img_types else {}
        parts = []
        for t in s.types:
            parts.append(geo_feature(L, s.ck, view) if t == "geo" else d[t][0])
        return np.concatenate(parts)


# ---------------------------------------------------------------- face points (geo)
def geo_norm(L):
    """(..., 468, 3) -> centred, unit size"""
    L = L - L.mean(-2, keepdims=True)
    return L / np.maximum(np.sqrt((L ** 2).sum((-1, -2), keepdims=True) / L.shape[-2]), 1e-6)


def geo_align(L, ref):
    """rotate each face's points (any head turn) onto the reference shape -> (..., 468, 3)"""
    X = geo_norm(np.asarray(L, np.float64))
    H = np.einsum("...ki,kj->...ij", X, ref)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.zeros(U.shape)
    D[..., 0, 0] = 1
    D[..., 1, 1] = 1
    D[..., 2, 2] = d
    return X @ (U @ D @ Vt)


def geo_reference(L):
    """average face shape of a set of faces (generalised Procrustes)"""
    ref = geo_norm(np.asarray(L[0], np.float64))
    for _ in range(4):
        ref = geo_norm(geo_align(L, ref).mean(0))
    return ref


def geo_scores(A, mu, comp, psd):
    """aligned points (n, 1404) -> 64 shape numbers each, scaled like the other inputs (about unit length)"""
    return (np.clip((A - mu) @ comp / psd, -6, 6) / np.sqrt(GEO_K)).astype(np.float32)


def geo_feature(L, ck, view):
    """one face's 468 points (photo or screenshot) -> shape numbers, as the bridge was trained"""
    a = geo_align(L, ck["geo_ref"][view]).reshape(1, -1)
    return geo_scores(a, ck["geo_mu"][view], ck["geo_comp"][view], ck["geo_psd"][view])[0]


def embed_geo(a, limit=None):
    """all 468 face points of each screenshot's front and turned face, plus skin/eye/hair colour measurements"""
    from PIL import Image
    import photo_to_dna as P
    import colour
    mesh = P.Mesh()
    for d in a.data:
        names = [str(n) for n in np.load(os.path.join(d, tr.cache_name(224, "_names.npy")))]
        n = len(names)
        donef = os.path.join(d, "_f3_done.txt")
        start = int(open(donef).read()) if os.path.exists(donef) else 0
        mode = "r+" if start else "w+"
        lm = np.lib.format.open_memmap(os.path.join(d, "_f3_lm.npy"), mode=mode, dtype=np.float32, shape=(n, 2, 468, 3))
        col = np.lib.format.open_memmap(os.path.join(d, "_f3_col.npy"), mode=mode, dtype=np.float32, shape=(n, 12))
        okp = os.path.join(d, "_f3_ok.npy")
        ok = np.load(okp) if start else np.zeros((n, 2), bool)
        if not start:
            col[:] = np.nan
        end = min(n, limit) if limit else n
        if start >= end:
            print(f"{d}: face points already done ({start})")
            continue
        print(f"{d}: face points + colours for {n} faces, {start} already done")
        t0 = time.time()
        for i in range(start, end):
            im = np.asarray(Image.open(os.path.join(d, names[i] + ".jpg")).convert("RGB"))
            H, W = im.shape[:2]
            for k, b in enumerate((tr.FRONT, tr.TURNED)):
                x0, y0 = int(W * b[0]), int(H * b[1])
                c = np.ascontiguousarray(im[y0:int(H * b[3]), x0:int(W * b[2])])
                p, L = mesh(c, full=True)
                if L is None:
                    continue
                lm[i, k] = L
                ok[i, k] = True
                if k == 0:                              # colours from the front face, measured on the whole picture
                    col[i] = colour.measure(im, L + np.array([x0, y0, 0], np.float32))
            if (i + 1) % 1000 == 0 or i == end - 1:
                lm.flush()
                col.flush()
                np.save(okp, ok)
                open(donef, "w").write(str(i + 1))
                left = (end - i - 1) * (time.time() - t0) / (i + 1 - start) / 60
                print(f"  {i + 1} / {n}  (front found {ok[:i + 1, 0].mean():.0%}, skin measured "
                      f"{(~np.isnan(col[:i + 1, 0])).mean():.0%}; ~{left:.0f} min left)")
    print("Next: python colour.py train ... and python bridge2.py train ...")


def embed(a):
    if a.geo:
        return embed_geo(a)
    from PIL import Image
    import photo_to_dna as P
    types = list(a.feats or FIRST_ROUND)
    first = set(types) == set(FIRST_ROUND)
    if not first and any(t in FIRST_ROUND for t in types):
        sys.exit("--feats: either the first round (arc glint mid) or extra types only (facenet)")
    tag = "" if first else "_" + "_".join(types)      # extra types get their own progress files
    fx, mesh = Feats(types), P.Mesh()
    V = K + 1
    for d in a.data:
        names = [str(n) for n in np.load(os.path.join(d, tr.cache_name(224, "_names.npy")))]
        n = len(names)
        mm = {}
        donef = os.path.join(d, f"_f2{tag}_done.txt")
        start = int(open(donef).read()) if os.path.exists(donef) else 0
        for t in types:                                # picture features only; face points come from embed --geo
            dim = DIMS[t]
            p = os.path.join(d, f"_f2_{t}.npy")
            mm[t] = np.lib.format.open_memmap(p, mode="r+" if start else "w+", dtype=np.float16, shape=(n, V, 2, dim))
        okp = os.path.join(d, f"_f2{tag}_ok.npy")
        ok = np.load(okp) if start else np.zeros((n, 2), bool)
        print(f"{d}: {n} faces x {V} versions (clean + {K} photo-like), {start} already done")
        t0 = time.time()
        GROUP = 16
        stop = min(n, start + a.limit) if a.limit else n    # --limit: a quick check on a few faces (resumes later)
        for g0 in range(start, stop, GROUP):
            crops, where = [], []
            for i in range(g0, min(n, g0 + GROUP)):
                rng = np.random.default_rng([11, i, n])
                im = np.asarray(Image.open(os.path.join(d, names[i] + ".jpg")).convert("RGB"))
                H, W = im.shape[:2]
                for k, b in enumerate((tr.FRONT, tr.TURNED)):
                    c = np.ascontiguousarray(im[int(H * b[1]):int(H * b[3]), int(W * b[0]):int(W * b[2])])
                    p = mesh(c)
                    if p is None:
                        continue
                    iod = np.linalg.norm(p[3] - p[0])
                    crops.append(fx.crop(c, p))
                    where.append((i, 0, k))
                    for v in range(1, V):
                        pj = p + rng.normal(0, 0.02 * iod, p.shape).astype(np.float32)
                        crops.append(fx.crop(B1.photo_aug(c, rng), pj))
                        where.append((i, v, k))
                    ok[i, k] = True
            if crops:
                f = fx.all(crops)
                idx = tuple(np.array(where).T)
                for t in types:
                    mm[t][idx] = f[t].astype(np.float16)
            done = min(n, g0 + GROUP)
            if done % 1008 < GROUP or done == n or done >= stop:
                for t in types:
                    mm[t].flush()
                np.save(okp, ok)
                open(donef, "w").write(str(done))
                left = (n - done) * (time.time() - t0) / max(1, done - start) / 60
                print(f"  {done} / {n}  (front found {ok[:done, 0].mean():.0%}; ~{left:.0f} min left)")
    print("Next: python bridge2.py train --data " + " ".join(a.data))


def train_one(a, types, data, log):
    import torch
    import torch.nn as nn
    names, Fs, OK, geo = data
    ck = torch.load(a.base, map_location="cpu", weights_only=False)
    tg = tr.Targets("", names, fixed=(ck["genes"], ck["tmpl"], ck["kind"]))
    mu, sd, cmu, csd = (np.asarray(ck[k], np.float32) for k in ("mu", "sd", "cmu", "csd"))
    G = len(tg.genes)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(names))
    te, trn = idx[:len(idx) // 10], idx[len(idx) // 10:]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dim = sum(DIMS[t] for t in types)
    sizes = {j: len(tg.tmpl[tg.genes[j]]) for j in tg.choice}
    X = {t: torch.from_numpy(Fs[t]).to(dev) for t in types}          # n, V, 2, dim (float16)
    hasT = torch.from_numpy(OK[:, 1]).to(dev)
    Vt = torch.from_numpy(((tg.V - mu) / sd).astype(np.float32)).to(dev)
    M = torch.from_numpy(tg.M).to(dev)
    C = torch.from_numpy(((tg.C - cmu) / csd).astype(np.float32)).to(dev)
    Tt = torch.from_numpy(tg.T).to(dev)
    steps = a.epochs * (len(trn) // a.batch)
    ce = nn.CrossEntropyLoss(ignore_index=-1)
    net = None

    def x_of(ids, mode):
        """mode: 'train' (random photo-like subsets, noise, sometimes no turned face), 'clean', 'photo'"""
        B = len(ids)
        rf, rt = torch.rand(B, 1, device=dev), torch.rand(B, 1, device=dev)

        def mask(Vn, r):
            m = torch.zeros(B, Vn, dtype=torch.bool, device=dev)
            if mode == "clean" or Vn == 1:
                m[:, 0] = True
                return m
            m = torch.rand(B, Vn, device=dev) < r
            m[torch.arange(B), torch.randint(0, Vn, (B,), device=dev)] = True
            return m
        parts_f, parts_t = [], []
        for t in types:
            x = X[t][ids].float()                                      # B, V, 2, d
            Vn = x.shape[1]
            mf, mt = mask(Vn, rf), mask(Vn, rt)
            if t == "geo":                                             # shape numbers: averaged, never rescaled
                f = (x[:, :, 0] * mf[..., None]).sum(1) / mf.sum(1, keepdim=True)
                tt = (x[:, :, 1] * mt[..., None]).sum(1) / mt.sum(1, keepdim=True)
                if mode == "train":                                    # face points found a little differently
                    f = f + torch.randn_like(f) * a.geo_noise / GEO_K ** 0.5
                    tt = tt + torch.randn_like(tt) * a.geo_noise / GEO_K ** 0.5
                parts_f.append(f)
                parts_t.append(tt)
                continue
            f = (x[:, :, 0] * mf[..., None]).sum(1)
            tt = (x[:, :, 1] * mt[..., None]).sum(1)
            if mode == "train":
                f = f + torch.randn_like(f) * a.noise * (512 / DIMS[t]) ** 0.5 * f.norm(dim=1, keepdim=True)
                tt = tt + torch.randn_like(tt) * a.noise * (512 / DIMS[t]) ** 0.5 * tt.norm(dim=1, keepdim=True)
            parts_f.append(f / f.norm(dim=1, keepdim=True).clamp(min=1e-6))
            parts_t.append(tt / tt.norm(dim=1, keepdim=True).clamp(min=1e-6))
        has = hasT[ids].float()[:, None]
        if mode == "train":
            has = has * (torch.rand(B, 1, device=dev) >= 0.3).float()
        tpart = torch.cat(parts_t, 1) * has
        return torch.cat([torch.cat(parts_f, 1), tpart, has], 1)

    def predict(ids, mode, front_only=False, nets=None):
        nets = nets or [net]
        for n_ in nets:
            n_.eval()
        out = []
        with torch.no_grad():
            for k in range(0, len(ids), 1024):
                b = torch.from_numpy(ids[k:k + 1024]).to(dev)
                x = x_of(b, mode)
                if front_only:
                    x[:, dim:] = 0
                out.append(torch.stack([n_(x)[0] for n_ in nets]).mean(0).cpu().numpy())
        return np.clip(np.concatenate(out) * sd + mu, 0, 1)

    def score(ids, v):
        reads = []
        for j in range(G):
            m = tg.M[ids, j] > 0
            if m.sum() < 5 or tg.kind[tg.genes[j]] == "choice":
                continue
            err = np.abs(v[m, j] - tg.V[ids][m, j]).mean()
            base = np.abs(tg.V[trn][tg.M[trn, j] > 0, j].mean() - tg.V[ids][m, j]).mean()
            if base * 255 >= 2:
                reads.append(1 - err / base)
        return np.median(reads)

    log(f"\nBridge using: {' + '.join(types)}  ({len(names)} faces; size {a.width} x {a.depth} layers"
        + (f"; a team of {a.team}" if a.team > 1 else "") + ")")
    team = []
    for member in range(a.team):
        torch.manual_seed(100 + member)              # each member starts differently; same faces held out
        net = B1.Net.build(G, sizes, in_dim=2 * dim + 1, width=a.width, depth=a.depth).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
        if a.team > 1:
            log(f" member {member + 1}/{a.team}")
        for ep in range(a.epochs):
            net.train()
            perm = rng.permutation(trn)
            tot = 0.0
            for k in range(0, len(perm) - a.batch + 1, a.batch):
                it = torch.from_numpy(perm[k:k + a.batch]).to(dev)
                v, c, l = net(x_of(it, "train"))
                loss = (((v - Vt[it]) ** 2) * M[it]).sum() / M[it].sum().clamp(min=1) \
                    + 0.5 * ((c - C[it]) ** 2).mean() \
                    + 0.5 * sum(ce(l[str(j)], Tt[it, j]) for j in tg.choice) / max(1, len(tg.choice))
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                tot += loss.item()
            if (ep + 1) % 10 == 0 or ep == a.epochs - 1:
                log(f"  epoch {ep + 1}/{a.epochs}: held-out sliders read {score(te, predict(te, 'clean')) * 100:.0f}% "
                    f"(front face only {score(te, predict(te, 'clean', True)) * 100:.0f}%, "
                    f"photo-like versions {score(te, predict(te, 'photo')) * 100:.0f}%)")
        team.append(net)
    if a.team > 1:
        log(f"  WHOLE TEAM averaged: held-out sliders read {score(te, predict(te, 'clean', nets=team)) * 100:.0f}% "
            f"(front face only {score(te, predict(te, 'clean', True, nets=team)) * 100:.0f}%, "
            f"photo-like versions {score(te, predict(te, 'photo', nets=team)) * 100:.0f}%)")
    out = a.out or os.path.join(HERE, "results", "bridge_" + "_".join(types) + (f"_team{a.team}" if a.team > 1 else "") + ".pt")
    save = {"state": team[0].state_dict(), "team": [n_.state_dict() for n_ in team], "width": a.width, "depth": a.depth,
            "genes": tg.genes, "tmpl": tg.tmpl, "kind": tg.kind, "mu": mu, "sd": sd,
            "cmu": cmu, "csd": csd, "choice": tg.choice, "feats": list(types), "in_dim": 2 * dim + 1}
    if "geo" in types:
        save.update(geo)
    torch.save(save, out)
    log(f"  saved {out}")


def train(a):
    names, OK, Fs, LM, LOK = [], [], {t: [] for t in DIMS}, [], []
    need_geo = any("geo" in v.split("+") for v in a.variants)
    for d in a.data:
        donef = os.path.join(d, "_f2_done.txt")
        nm = [str(n) for n in np.load(os.path.join(d, tr.cache_name(224, "_names.npy")))]
        if not os.path.exists(donef) or int(open(donef).read()) < len(nm):
            sys.exit(f"{d}: the embed run isn't finished - run it again to carry on.")
        n = len(nm)
        for t in IMAGE_TYPES:
            if t not in FIRST_ROUND and any(t in v.split("+") for v in a.variants):
                xd = os.path.join(d, f"_f2_{t}_done.txt")
                if not os.path.exists(xd) or int(open(xd).read()) < (min(len(nm), a.limit) if a.limit else len(nm)):
                    sys.exit(f"{d}: the {t} run (embed --feats {t}) isn't finished - run it again to carry on.")
        if need_geo:
            gdone = os.path.join(d, "_f3_done.txt")
            gd = int(open(gdone).read()) if os.path.exists(gdone) else 0
            if gd < n and not a.limit:
                sys.exit(f"{d}: the face-point run (embed --geo) isn't finished - run it again to carry on.")
            n = min(n, gd)
        if a.limit:
            n = min(n, a.limit)
        ok = np.load(os.path.join(d, "_f2_ok.npy"))[:n]
        keep = ok[:, 0].copy()
        if a.hold_out_refs:                           # faces built on a reference person: left out (fair tests)
            import csv
            mp, refs = os.path.join(d, "meta.csv"), tr.ref_keys()
            drop = {r["shot"] for r in csv.DictReader(open(mp, newline="")) if r.get("anchor") in refs} \
                if os.path.exists(mp) else set()
            out_ = np.array([x in drop for x in nm[:n]])
            keep &= ~out_
            print(f"{d}: {int(out_.sum())} faces built on reference people left out (--hold-out-refs)")
        if need_geo:
            gok = np.load(os.path.join(d, "_f3_ok.npy"))[:n]
            keep &= gok[:, 0]
            LOK.append(gok[keep])
            LM.append(np.load(os.path.join(d, "_f3_lm.npy"), mmap_mode="r")[:n][keep])
        names += [os.path.join(d, x) for x, k in zip(nm[:n], keep) if k]
        OK.append(ok[keep])
        for t in IMAGE_TYPES:
            if any(t in v.split("+") for v in a.variants):
                Fs[t].append(np.load(os.path.join(d, f"_f2_{t}.npy"), mmap_mode="r")[:n][keep])
    OK = np.concatenate(OK)
    Fs = {t: np.concatenate(v) for t, v in Fs.items() if v}
    geo = {}
    if need_geo:                                      # face points -> aligned, standardised shape numbers
        LM, LOK = np.concatenate(LM), np.concatenate(LOK)
        G = np.zeros((len(LM), 1, 2, GEO_K), np.float32)
        geo = {"geo_ref": [], "geo_mu": [], "geo_comp": [], "geo_psd": []}
        for v in (0, 1):
            sel = np.where(LOK[:, v])[0]
            ref = geo_reference(LM[sel[:min(len(sel), 5000)], v])
            A = np.concatenate([geo_align(LM[sel[k:k + 4096], v], ref).reshape(-1, GEO_RAW)
                                for k in range(0, len(sel), 4096)])
            m = A.mean(0)
            sub = A[np.random.default_rng(1).permutation(len(A))[:20000]] - m
            comp = np.linalg.svd(sub, full_matrices=False)[2][:GEO_K].T          # 1404 x 64
            psd = np.maximum(((A - m) @ comp).std(0), 1e-6)
            G[sel, 0, v] = geo_scores(A, m, comp, psd)
            for k_, x_ in (("geo_ref", ref), ("geo_mu", m), ("geo_comp", comp), ("geo_psd", psd)):
                geo[k_].append(x_)
        OK[:, 1] &= LOK[:, 1]
        Fs["geo"] = G
    rep = open(a.report or os.path.join(HERE, "results", "bridge2_report.txt"), "w")

    def log(t):
        print(t)
        rep.write(t + "\n")
        rep.flush()
    log("Held-out game faces; the main network reads 57%, the current bridge (arc only) 41%.")
    for v in a.variants:
        train_one(a, v.split("+"), (names, Fs, OK, geo), log)


class Team:
    """several bridges answering together: sliders averaged, button likelihoods averaged"""
    def __init__(s, nets):
        s.nets = nets

    def __call__(s, x):
        import torch
        outs = [n(x) for n in s.nets]
        v = torch.stack([o[0] for o in outs]).mean(0)
        c = torch.stack([o[1] for o in outs]).mean(0)
        l = {j: torch.log(torch.stack([torch.softmax(o[2][j].float(), 1) for o in outs]).mean(0) + 1e-9)
             for j in outs[0][2]}
        return v, c, l


def load(path, dev):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    states = ck.get("team") or [ck["state"]]
    nets = []
    for st in states:
        net = B1.Net.build(len(ck["genes"]), {j: len(ck["tmpl"][ck["genes"][j]]) for j in ck["choice"]},
                           in_dim=ck["in_dim"], width=ck.get("width", 1024), depth=ck.get("depth", 2))
        net.load_state_dict(st)
        nets.append(net.to(dev).eval())
    return (nets[0] if len(nets) == 1 else Team(nets)), ck


def predict_person_full(net, ck, front, turned, dev):
    """front: (n, dim) rows from Feats.photo_vec; turned: (m, dim) or empty ->
    (slider values 0-1, {button gene index: probabilities})"""
    import torch
    types = ck["feats"]

    def avg(X):
        parts, o = [], 0
        for t in types:
            dm = DIMS[t]
            s = X[:, o:o + dm].mean(0)
            parts.append(s if t == "geo" else s / max(np.linalg.norm(s), 1e-6))
            o += dm
        return np.concatenate(parts)
    f = avg(front)
    t = avg(turned) if len(turned) else np.zeros(sum(DIMS[x] for x in types), np.float32)
    x = np.concatenate([f, t, [1.0 if len(turned) else 0.0]]).astype(np.float32)[None]
    with torch.no_grad():
        v, _, l = net(torch.from_numpy(x).to(dev))
    probs = {int(j): torch.softmax(z[0].float(), 0).cpu().numpy() for j, z in l.items()}
    return np.clip(v.cpu().numpy()[0] * ck["sd"] + ck["mu"], 0, 1), probs


def predict_person(net, ck, front, turned, dev):
    return predict_person_full(net, ck, front, turned, dev)[0]


def selftest(a):
    """Every step of tonight's run on a few faces and one person, in a few minutes, so nothing fails overnight."""
    import photo_to_dna as P
    import colour
    res = os.path.join(HERE, "results")
    print("1/5  face-recognition numbers from the last run are complete?")
    for d in a.data:
        nm = np.load(os.path.join(d, tr.cache_name(224, "_names.npy")))
        donef = os.path.join(d, "_f2_done.txt")
        if not os.path.exists(donef) or int(open(donef).read()) < len(nm):
            sys.exit(f"  NO: {d} - run 'python bridge2.py embed --data ...' first.")
    print("  yes")
    print("2/5  face points + colour measurements on the first 64 faces of each folder")
    embed_geo(a, limit=64)
    print("3/5  colour conversion from those faces")
    colour.train(argparse.Namespace(data=a.data, limit=64, out=os.path.join(res, "_selftest_colour.npz")))
    print("4/5  a 1-pass bridge with face points")
    b = argparse.Namespace(**vars(a))
    b.variants, b.epochs, b.limit, b.batch, b.team = ["arc+glint+mid+geo"], 1, 64, 32, 2
    b.out, b.report = os.path.join(res, "_selftest_bridge.pt"), os.path.join(res, "_selftest_report.txt")
    train(b)
    print("5/5  one person through photo_to_dna with that bridge and colour model")
    argv = sys.argv
    sys.argv = ["photo_to_dna.py", "--photos", a.photos, "--model", a.base, "--bridge", b.out,
                "--bridge-weight", "0.5", "0.6", "--button-weight", "0.5", "1.0", "--colour", os.path.join(res, "_selftest_colour.npz"),
                "--out", "_selftest"]
    P.main()
    sys.argv = argv
    name = os.path.basename(os.path.normpath(a.photos)).replace(" ", "_")
    outd = os.path.join(res, "_selftest", name)
    missing = [f for f in (f"{name}.txt", f"{name}_bridge.txt", f"{name}_s50_b50.txt", f"{name}_s50_b50_col.txt",
                            f"{name}_s60_b100.txt")
               if not os.path.exists(os.path.join(outd, f))]
    if missing:
        sys.exit(f"SELFTEST FAILED: missing {missing} in {outd} - send me this window's text.")
    print("\nSELFTEST OK - everything ran. Start the overnight block.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("embed", "train", "selftest"):
        p = sub.add_parser(name)
        p.add_argument("--data", nargs="+", required=True)
        p.add_argument("--geo", action="store_true", help="embed: face points + colour measurements")
        p.add_argument("--feats", nargs="+", default=None, help="embed: picture features to add (e.g. facenet)")
        p.add_argument("--limit", type=int, default=0, help="train: only the first N faces of each folder")
        p.add_argument("--geo-noise", type=float, default=0.3)
        p.add_argument("--width", type=int, default=1024, help="train: units per layer")
        p.add_argument("--depth", type=int, default=2, help="train: layers")
        p.add_argument("--team", type=int, default=1, help="train: this many bridges, averaged")
        p.add_argument("--out", default="")
        p.add_argument("--report", default="")
        p.add_argument("--photos", default="", help="selftest: one person's photo folder")
        p.add_argument("--base", default=B1.BASE_MODEL)
        p.add_argument("--epochs", type=int, default=40)
        p.add_argument("--batch", type=int, default=256)
        p.add_argument("--noise", type=float, default=0.03)
        p.add_argument("--variants", nargs="+", default=["arc+glint+mid+geo"])
        p.add_argument("--hold-out-refs", action="store_true",
                       help="train: leave out faces built on any reference person in refs.csv (for fair tests)")
    a = ap.parse_args()
    {"embed": embed, "train": train, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
