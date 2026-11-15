"""Route A: a face-recognition bridge from photos to DNA.

A face-recognition network (ArcFace, from InsightFace's free 'buffalo_l' pack - non-commercial use) turns a face into
512 numbers describing WHO it is, and is built to give similar numbers for the same person in very different images.
  1. embed: those numbers for the front and turned face of every training screenshot (exact DNA known)
  2. train: a small network from those numbers to DNA
  3. photo_to_dna.py --bridge results\\bridge.pt: the same numbers from a person's photos -> DNA,
     written as <person>_bridge.txt (bridge alone) and <person>_mix.txt (bridge and the main network averaged)

    python -m pip install onnxruntime
    python bridge.py embed --data random_faces\\noise_female random_faces\\real_female     (~1-2 h, no game; resumes)
    python bridge.py train --data random_faces\\noise_female random_faces\\real_female     (~10 min)
    python bridge.py embed --aug 3 --data ...   photo-like versions too (then train again; ~2-3 h)
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

PACK_URL = "https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip"
MODEL = os.path.join(HERE, "results", "arcface", "w600k_r50.onnx")
BASE_MODEL = os.path.join(HERE, "results", "model_29996_v3_base_photo.pt")
# where ArcFace expects eyes, nose tip and mouth corners in its 112x112 input
DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
               np.float32)


class ArcFace:
    def __init__(s):
        import onnxruntime as ort
        if not os.path.exists(MODEL):
            import urllib.request
            os.makedirs(os.path.dirname(MODEL), exist_ok=True)
            z = os.path.join(os.path.dirname(MODEL), "buffalo_l.zip")
            print("Downloading InsightFace's buffalo_l pack (~280 MB, once) ...")
            urllib.request.urlretrieve(PACK_URL, z)
            with zipfile.ZipFile(z) as f:
                name = next(n for n in f.namelist() if n.endswith("w600k_r50.onnx"))
                open(MODEL, "wb").write(f.read(name))
            os.remove(z)
        prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        so = ort.SessionOptions()
        so.log_severity_level = 3                     # hide a harmless "expected shape {1,512}" warning per batch
        s.sess = ort.InferenceSession(MODEL, sess_options=so, providers=prov)
        s.inp = s.sess.get_inputs()[0].name

    @staticmethod
    def five(p):
        """photo_to_dna's 9 key points -> eye centres, nose tip, mouth corners (as they appear left to right)."""
        return np.array([(p[0] + p[1]) / 2, (p[2] + p[3]) / 2, p[4], p[5], p[6]], np.float32)

    def crop(s, rgb, p):
        import cv2
        M, _ = cv2.estimateAffinePartial2D(s.five(p), DST, method=cv2.LMEDS)
        return cv2.warpAffine(rgb, M, (112, 112), borderMode=cv2.BORDER_REPLICATE)

    def __call__(s, crops):
        """list of 112x112x3 RGB uint8 -> (n,512) unit vectors"""
        x = (np.stack(crops).astype(np.float32) - 127.5) / 127.5
        e = s.sess.run(None, {s.inp: x.transpose(0, 3, 1, 2)})[0]
        return e / np.linalg.norm(e, axis=1, keepdims=True)


def photo_aug(img, rng):
    """Make a game crop look like it came from a random photo: exposure, contrast, colour cast, side light,
    soft focus, low resolution, sensor noise, JPEG compression, sometimes washed-out colour."""
    import cv2
    x = img.astype(np.float32) / 255
    h, w = x.shape[:2]
    m = x.mean()
    x = (x - m) * rng.uniform(0.7, 1.3) + m
    x = x * rng.uniform(0.75, 1.25) * rng.uniform(0.92, 1.08, 3)
    x = np.clip(x, 0, 1) ** rng.uniform(0.8, 1.25)
    d = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[-1:1:h * 1j, -1:1:w * 1j]
    x = x * (1 + (np.cos(d) * xx + np.sin(d) * yy)[..., None] * rng.uniform(0, 0.35))
    if rng.random() < 0.2:
        g = x.mean(2, keepdims=True)
        x = g + (x - g) * rng.uniform(0.3, 0.8)
    x = np.clip(x, 0, 1)
    if rng.random() < 0.6:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.5, 2.0))
    if rng.random() < 0.5:
        f = rng.uniform(0.35, 0.9)
        x = cv2.resize(cv2.resize(x, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA), (w, h))
    x = np.clip(x + rng.normal(0, rng.uniform(0, 0.04), x.shape), 0, 1)
    out = (x * 255).astype(np.uint8)
    if rng.random() < 0.5:
        ok, buf = cv2.imencode(".jpg", out[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(30, 90))])
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]
    return np.ascontiguousarray(out)


def embed_aug(a):
    """K photo-like versions of each screenshot's front and turned face -> <folder>/_arcface_aug.npz"""
    from PIL import Image
    import photo_to_dna as P
    af, mesh = ArcFace(), P.Mesh()
    K = a.aug
    for d in a.data:
        names = [str(n) for n in np.load(os.path.join(d, tr.cache_name(224, "_names.npy")))]
        out = os.path.join(d, "_arcface_aug.npz")
        E = np.zeros((len(names), K, 2, 512), np.float16)
        ok = np.zeros((len(names), 2), bool)
        start = 0
        if os.path.exists(out):
            z = np.load(out)
            if list(z["names"]) == names and z["E"].shape[1] == K:
                E, ok, start = z["E"], z["ok"], int(z["done"])
        print(f"{d}: {len(names)} faces x {K} photo-like versions, {start} already done")
        t0 = time.time()
        for i in range(start, len(names)):
            rng = np.random.default_rng([11, i, len(names)])
            im = np.asarray(Image.open(os.path.join(d, names[i] + ".jpg")).convert("RGB"))
            H, W = im.shape[:2]
            for k, b in enumerate((tr.FRONT, tr.TURNED)):
                c = np.ascontiguousarray(im[int(H * b[1]):int(H * b[3]), int(W * b[0]):int(W * b[2])])
                p = mesh(c)
                if p is None:
                    continue
                iod = np.linalg.norm(p[3] - p[0])
                crops = []
                for _ in range(K):          # photo-like image, and the key points found slightly differently
                    pj = p + rng.normal(0, 0.02 * iod, p.shape).astype(np.float32)
                    crops.append(af.crop(photo_aug(c, rng), pj))
                E[i, :, k] = af(crops).astype(np.float16)
                ok[i, k] = True
            if (i + 1) % 1000 == 0 or i == len(names) - 1:
                np.savez(out, names=np.array(names), E=E, ok=ok, done=i + 1)
                left = (len(names) - i - 1) * (time.time() - t0) / (i + 1 - start) / 60
                print(f"  {i + 1} / {len(names)}  (~{left:.0f} min left)")
    print("Next: python bridge.py train --data " + " ".join(a.data))


def embed(a):
    if a.aug:
        return embed_aug(a)
    from PIL import Image
    import photo_to_dna as P
    af, mesh = ArcFace(), P.Mesh()
    for d in a.data:
        names = [str(n) for n in np.load(os.path.join(d, tr.cache_name(224, "_names.npy")))]
        out = os.path.join(d, "_arcface.npz")
        E = np.zeros((len(names), 2, 512), np.float32)
        ok = np.zeros((len(names), 2), bool)
        start = 0
        if os.path.exists(out):
            z = np.load(out)
            if list(z["names"]) == names:
                E, ok, start = z["E"], z["ok"], int(z["done"])
        print(f"{d}: {len(names)} faces, {start} already done")
        t0 = time.time()
        for i in range(start, len(names)):
            im = np.asarray(Image.open(os.path.join(d, names[i] + ".jpg")).convert("RGB"))
            H, W = im.shape[:2]
            for k, b in enumerate((tr.FRONT, tr.TURNED)):
                c = np.ascontiguousarray(im[int(H * b[1]):int(H * b[3]), int(W * b[0]):int(W * b[2])])
                p = mesh(c)
                if p is not None:
                    E[i, k] = af([af.crop(c, p)])[0]
                    ok[i, k] = True
            if (i + 1) % 1000 == 0 or i == len(names) - 1:
                np.savez(out, names=np.array(names), E=E, ok=ok, done=i + 1)
                left = (len(names) - i - 1) * (time.time() - t0) / (i + 1 - start) / 60
                print(f"  {i + 1} / {len(names)}  (front found {ok[:i + 1, 0].mean():.0%}, turned "
                      f"{ok[:i + 1, 1].mean():.0%}; ~{left:.0f} min left)")
    print("Next: python bridge.py train --data " + " ".join(a.data))


class Net:
    @staticmethod
    def build(G, cls_sizes, in_dim=1025, width=1024, depth=2):
        import torch.nn as nn

        class M(nn.Module):
            def __init__(s):
                super().__init__()
                layers = [nn.Linear(in_dim, width), nn.GELU(), nn.Dropout(0.2)]
                for _ in range(depth - 1):
                    layers += [nn.Linear(width, width), nn.GELU(), nn.Dropout(0.1)]
                s.trunk = nn.Sequential(*layers)
                s.val, s.col = nn.Linear(width, G), nn.Linear(width, 6)
                s.cls = nn.ModuleDict({str(j): nn.Linear(width, n) for j, n in cls_sizes.items()})

            def forward(s, x):
                z = s.trunk(x)
                return s.val(z), s.col(z), {j: c(z) for j, c in s.cls.items()}
        return M()


def inputs(front, turned):
    """front, turned: (n,512) (turned may be all zeros = no turned face) -> (n,1025) network input."""
    has = (np.abs(turned).sum(1, keepdims=True) > 0).astype(np.float32)
    return np.concatenate([front, turned, has], 1).astype(np.float32)


def train(a):
    import torch
    import torch.nn as nn
    ck = torch.load(a.base, map_location="cpu", weights_only=False)
    names, F, T, OK, AUG = [], [], [], [], []
    use_aug = all(os.path.exists(os.path.join(d, "_arcface_aug.npz")) for d in a.data) and not a.no_aug
    for d in a.data:
        z = np.load(os.path.join(d, "_arcface.npz"))
        n = int(z["done"])
        names += [os.path.join(d, x) for x in z["names"][:n]]
        F.append(z["E"][:n, 0]); T.append(z["E"][:n, 1]); OK.append(z["ok"][:n])
        if use_aug:
            za = np.load(os.path.join(d, "_arcface_aug.npz"))
            if int(za["done"]) < n:
                sys.exit(f"{d}: the photo-like embed run isn't finished - run it again to carry on.")
            AUG.append(za["E"][:n].astype(np.float32))
    F, T, OK = np.concatenate(F), np.concatenate(T), np.concatenate(OK)
    keep = OK[:, 0]
    T[~OK[:, 1]] = 0
    names = [n for n, k in zip(names, keep) if k]
    F, T = F[keep], T[keep]
    if use_aug:                                       # views: [clean, photo-like 1..K] for front and turned
        A = np.concatenate(AUG)[keep]
        FV = np.concatenate([F[:, None], A[:, :, 0]], 1)
        TV = np.concatenate([T[:, None], A[:, :, 1]], 1)
        TV[np.abs(T).sum(1) == 0] = 0
    rep = open(os.path.join(HERE, "results", "bridge_report.txt"), "w")

    def log(t):                                       # console and results\bridge_report.txt
        print(t)
        rep.write(t + "\n")
        rep.flush()
    log(f"{len(names)} faces with a front face found" +
        (f"; {A.shape[1]} photo-like versions of each" if use_aug else "; clean screenshots only"))
    tg = tr.Targets("", names, fixed=(ck["genes"], ck["tmpl"], ck["kind"]))
    mu, sd, cmu, csd = (np.asarray(ck[k], np.float32) for k in ("mu", "sd", "cmu", "csd"))
    G = len(tg.genes)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(names))
    te, trn = idx[:len(idx) // 10], idx[len(idx) // 10:]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = Net.build(G, {j: len(tg.tmpl[tg.genes[j]]) for j in tg.choice}).to(dev)
    V = torch.from_numpy(((tg.V - mu) / sd).astype(np.float32)).to(dev)
    M = torch.from_numpy(tg.M).to(dev)
    C = torch.from_numpy(((tg.C - cmu) / csd).astype(np.float32)).to(dev)
    Tt = torch.from_numpy(tg.T).to(dev)
    Fd, Td = torch.from_numpy(F).to(dev), torch.from_numpy(T).to(dev)
    if use_aug:
        FVd, TVd = torch.from_numpy(FV).to(dev), torch.from_numpy(TV).to(dev)

    def views(X, ids):
        """average of a random subset of a face's versions, like averaging over a person's photos"""
        v = X[ids]                                    # B, K+1, 512
        m = torch.rand(v.shape[:2], device=dev) < torch.rand(len(ids), 1, device=dev)
        m[torch.arange(len(ids)), torch.randint(0, v.shape[1], (len(ids),), device=dev)] = True
        s = (v * m[..., None]).sum(1)
        return s / s.norm(dim=1, keepdim=True).clamp(min=1e-6)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)
    steps = a.epochs * (len(trn) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
    ce = nn.CrossEntropyLoss(ignore_index=-1)

    def x_of(ids, train_mode, photo_like=False):
        f, t = Fd[ids], Td[ids]
        if use_aug and (train_mode or photo_like):
            f, t = views(FVd, ids), views(TVd, ids)
        if train_mode:                          # photo-like wobble in the numbers, and sometimes no turned face
            f = f + torch.randn_like(f) * a.noise
            t = t + torch.randn_like(t) * a.noise
            f = f / f.norm(dim=1, keepdim=True)
            t = t / t.norm(dim=1, keepdim=True).clamp(min=1e-6)
            drop = torch.rand(len(ids), 1, device=dev) < 0.3
            t = torch.where(drop, torch.zeros_like(t), t)
        has = (t.abs().sum(1, keepdim=True) > 0).float()
        return torch.cat([f, t, has], 1)

    def predict(ids, front_only=False, photo_like=False):
        net.eval()
        with torch.no_grad():
            out = []
            for k in range(0, len(ids), 1024):
                b = torch.from_numpy(ids[k:k + 1024]).to(dev)
                x = x_of(b, False, photo_like)
                if front_only:
                    x[:, 512:] = 0
                out.append(net(x)[0].cpu().numpy())
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

    for ep in range(a.epochs):
        net.train()
        perm = rng.permutation(trn)
        tot = 0.0
        for k in range(0, len(perm) - a.batch + 1, a.batch):
            it = torch.from_numpy(perm[k:k + a.batch]).to(dev)
            v, c, l = net(x_of(it, True))
            loss = (((v - V[it]) ** 2) * M[it]).sum() / M[it].sum().clamp(min=1) + 0.5 * ((c - C[it]) ** 2).mean() \
                + 0.5 * sum(ce(l[str(j)], Tt[it, j]) for j in tg.choice) / max(1, len(tg.choice))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        if (ep + 1) % 5 == 0 or ep == a.epochs - 1:
            log(f"  epoch {ep + 1}/{a.epochs}: loss {tot / (len(perm) // a.batch):.3f}, held-out sliders read "
                  f"{score(te, predict(te)) * 100:.0f}% (front face only: {score(te, predict(te, True)) * 100:.0f}%"
                  + (f"; photo-like versions: {score(te, predict(te, photo_like=True)) * 100:.0f}%)" if use_aug else ")"))
    out = os.path.join(HERE, "results", "bridge.pt")
    torch.save({"state": net.state_dict(), "genes": tg.genes, "tmpl": tg.tmpl, "kind": tg.kind, "mu": mu, "sd": sd,
                "cmu": cmu, "csd": csd, "choice": tg.choice}, out)
    log(f"Saved {out}.  For comparison, the main network reads 57% from the screenshots themselves.")


def load(path, dev):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = Net.build(len(ck["genes"]), {j: len(ck["tmpl"][ck["genes"][j]]) for j in ck["choice"]})
    net.load_state_dict(ck["state"])
    return net.to(dev).eval(), ck


def predict_person(net, ck, front, turned, dev):
    """front: (n,512), turned: (m,512) or empty -> slider values 0-1 (one face)."""
    import torch
    f = front.mean(0)
    f = f / np.linalg.norm(f)
    t = np.zeros(512, np.float32)
    if len(turned):
        t = turned.mean(0)
        t = t / np.linalg.norm(t)
    x = torch.from_numpy(inputs(f[None], t[None])).to(dev)
    with torch.no_grad():
        v = net(x)[0].cpu().numpy()[0]
    return np.clip(v * ck["sd"] + ck["mu"], 0, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("embed", "train"):
        p = sub.add_parser(name)
        p.add_argument("--data", nargs="+", required=True)
        p.add_argument("--base", default=BASE_MODEL, help="main model: its settings list and scaling are reused")
        p.add_argument("--epochs", type=int, default=40)
        p.add_argument("--batch", type=int, default=256)
        p.add_argument("--noise", type=float, default=0.03)
        p.add_argument("--aug", type=int, default=0, help="embed: make this many photo-like versions of each face")
        p.add_argument("--no-aug", action="store_true", help="train: ignore the photo-like versions")
    a = ap.parse_args()
    {"embed": embed, "train": train}[a.cmd](a)


if __name__ == "__main__":
    main()
