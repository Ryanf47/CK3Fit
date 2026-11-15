"""Specialist readers: one small network per face region (eyes first), looking at a close, high-resolution,
precisely lined-up crop of just that region, trained only on that region's settings.

Step 1 (this file) is the game-side test: does the eye specialist read the eye and brow sliders clearly better
than the current readers, on game faces none of them trained on? If not, it can't help on photos either.

    python region.py prepare --region eyes      (once, ~30 min: cuts the lined-up crops; ~30 GB of disk)
    python region.py train   --region eyes      (hours; no game; saves after every epoch and carries on if restarted)
        -> results\\region_eyes.pt and results\\region_eyes_report.txt

Both commands can simply be run again after an interruption: they carry on where they stopped.
No game needed for any of it.

How it's trained:
  - crops are lined up on the face points (eye centres always in the same place, same size, level), front and turned
  - fake makeup is painted on during training (eyeliner, lower liner, eyeshadow, filled-in brows), plus the usual
    photo-like changes (lighting, colour, blur, noise, slight misalignment), so it learns to read through them
  - it's tested on the bridge's own held-out faces, so all readers are compared on faces they never saw
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = (os.environ.get("CK3FIT_HOME") or os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import train_reverse as tr        # noqa: E402

DATA = [os.path.join("random_faces", "noise_female"), os.path.join("random_faces", "real_female")]

# MediaPipe face points used for painting makeup (in each crop's own coordinates)
R_UP, R_LO = [33, 246, 161, 160, 159, 158, 157, 173, 133], [33, 7, 163, 144, 145, 153, 154, 155, 133]
L_UP, L_LO = [263, 466, 388, 387, 386, 385, 384, 398, 362], [263, 249, 390, 373, 374, 380, 381, 382, 362]
R_BROW_U, R_BROW_L = [70, 63, 105, 66, 107], [46, 53, 52, 65, 55]
L_BROW_U, L_BROW_L = [300, 293, 334, 296, 336], [276, 283, 282, 295, 285]
MK = R_UP + R_LO + L_UP + L_LO + R_BROW_U + R_BROW_L + L_BROW_U + L_BROW_L          # 56 points
S_RUP, S_RLO, S_LUP, S_LLO = slice(0, 9), slice(9, 18), slice(18, 27), slice(27, 36)
S_RBU, S_RBL, S_LBU, S_LBL = slice(36, 41), slice(41, 46), slice(46, 51), slice(51, 56)

# lips: outer outline (corner to corner along the top, then back along the bottom), and the inner outline
LIP_OUT = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
LIP_IN = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
NOSE_PTS = [1, 4, 5, 195, 197, 98, 327, 2, 64, 294]

REGIONS = {
    # out: crop size (w, h); x/y: crop span in eye-centre distances d, around the eye-centre midpoint (y down);
    # mk: face points kept with each crop (for painting makeup); paint: which fake makeup is painted on
    "eyes": {"out": (448, 224), "x": (-1.3, 1.3), "y": (-0.8, 0.5), "mk": MK, "paint": "eyes",
             "genes": lambda g: "eye" in g or "brow" in g, "eye_colour": True},
    "nose": {"out": (378, 294), "x": (-0.7, 0.8), "y": (-0.15, 1.0), "mk": NOSE_PTS, "paint": None,
             "genes": lambda g: "nose" in g, "eye_colour": False},
    "mouth": {"out": (420, 266), "x": (-0.95, 1.1), "y": (0.75, 2.05), "mk": LIP_OUT + LIP_IN, "paint": "lips",
              "genes": lambda g: any(k in g for k in ("mouth", "lip", "chin")), "eye_colour": False},
}
MARGIN = 16


def region_of(a):
    return REGIONS[a.region]


def cache_paths(d, region):
    return (os.path.join(d, f"_reg_{region}.npy"), os.path.join(d, f"_reg_{region}_pts.npy"),
            os.path.join(d, f"_reg_{region}_done.txt"))


def eye_frame(L):
    """eye-centre midpoint, eye-centre distance, angle of the line through the eyes"""
    er = (L[33, :2] + L[133, :2]) / 2
    el = (L[362, :2] + L[263, :2]) / 2
    v = el - er
    return (er + el) / 2, float(np.linalg.norm(v)), float(np.arctan2(v[1], v[0]))


def region_matrix(L, R, frontal_d=None):
    """2x3 matrix: picture -> crop (with MARGIN on each side). frontal_d: use this eye distance instead (turned view:
    the eyes look closer together when turned, so the size comes from the face height there)."""
    c, d, ang = eye_frame(L)
    if frontal_d is not None:
        d = frontal_d
    W, H = R["out"]
    sx = W / ((R["x"][1] - R["x"][0]) * d)
    ca, sa = np.cos(-ang), np.sin(-ang)
    Rm = np.array([[ca, -sa], [sa, ca]]) * sx
    # crop position of the eye-centre midpoint
    ox = MARGIN + (-R["x"][0]) * d * sx
    oy = MARGIN + (-R["y"][0]) * d * sx
    t = np.array([ox, oy]) - Rm @ c
    return np.hstack([Rm, t[:, None]]).astype(np.float32)


def face_scale(L):
    """a size measure that doesn't shrink when the head turns: eye line to chin"""
    c, _, _ = eye_frame(L)
    return float(np.linalg.norm(L[152, :2] - c))


def cut(img, L, R, ratio=None):
    """img: full picture (H,W,3); L: 468 points in it -> crop (h+2M, w+2M, 3) uint8 and the makeup points in it"""
    import cv2
    fd = face_scale(L) / ratio if ratio else None
    M = region_matrix(L, R, fd)
    W, H = R["out"]
    crop = cv2.warpAffine(img, M, (W + 2 * MARGIN, H + 2 * MARGIN), flags=cv2.INTER_AREA,
                          borderMode=cv2.BORDER_REPLICATE)
    pts = (L[R["mk"], :2] @ M[:, :2].T + M[:, 2]).astype(np.float32)
    return crop, pts


# ---------------------------------------------------------------- prepare
def all_rows(data):
    rows = []
    for d in data:
        d = os.path.join(HERE, d) if not os.path.isabs(d) else d
        nm = [str(n) for n in np.load(os.path.join(d, tr.cache_name(224, "_names.npy")))]
        rows.append((d, nm))
    return rows


def prepare(a):
    from PIL import Image
    R = region_of(a)
    W, H = R["out"]
    for d, nm in all_rows(a.data):
        n = len(nm)
        done_f = int(open(os.path.join(d, "_f3_done.txt")).read())
        lm = np.load(os.path.join(d, "_f3_lm.npy"), mmap_mode="r")
        ok = np.load(os.path.join(d, "_f3_ok.npy"))
        cp, pp, dp = cache_paths(d, a.region)
        start = int(open(dp).read()) if os.path.exists(dp) else 0
        mode = "r+" if start else "w+"
        arr = np.lib.format.open_memmap(cp, mode=mode, dtype=np.uint8, shape=(n, 2, H + 2 * MARGIN, W + 2 * MARGIN, 3))
        pts = np.lib.format.open_memmap(pp, mode=mode, dtype=np.float32, shape=(n, 2, len(R["mk"]), 2))
        okp = os.path.join(d, f"_reg_{a.region}_ok.npy")
        rok = np.load(okp) if start else np.zeros((n, 2), bool)
        # turned view: typical (eye-line-to-chin / eye distance) of front faces, to size the turned crop the same way
        sample = [i for i in range(0, min(n, done_f), max(1, min(n, done_f) // 500)) if ok[i, 0]]
        ratio = float(np.median([face_scale(np.asarray(lm[i, 0])) / eye_frame(np.asarray(lm[i, 0]))[1]
                                 for i in sample]))
        np.save(os.path.join(d, f"_reg_{a.region}_ratio.npy"), np.array([ratio]))
        end = min(n, done_f)
        print(f"{d}: {n} faces, {start} already cut; face points available for {done_f}")
        t0 = time.time()
        for i in range(start, end):
            if ok[i, 0]:
                im = np.asarray(Image.open(os.path.join(d, nm[i] + ".jpg")).convert("RGB"))
                Hh, Ww = im.shape[:2]
                for k, b in enumerate((tr.FRONT, tr.TURNED)):
                    if not ok[i, k]:
                        continue
                    off = np.array([int(Ww * b[0]), int(Hh * b[1])], np.float32)
                    L = np.asarray(lm[i, k], np.float32).copy()
                    L[:, :2] += off
                    arr[i, k], pts[i, k] = cut(im, L, R, ratio if k == 1 else None)
                    rok[i, k] = True
            if (i + 1) % 1000 == 0 or i == end - 1:
                arr.flush()
                pts.flush()
                np.save(okp, rok)
                open(dp, "w").write(str(i + 1))
                left = (end - i - 1) * (time.time() - t0) / (i + 1 - start) / 60
                print(f"  {i + 1} / {end}  (~{left:.0f} min left)")
        if start >= end:
            print("  already done")
        # a picture of the first few, to check the line-up
        tiles = [arr[i, k] for i in range(min(n, 40)) for k in (0, 1) if rok[i, k]][:12]
        if tiles:
            sheet = np.concatenate([np.concatenate(tiles[r * 2:r * 2 + 2], 1) for r in range(len(tiles) // 2)], 0)
            Image.fromarray(sheet).save(os.path.join(HERE, "results", f"region_{a.region}_check.jpg"), quality=88)
    print(f"Done. Check results\\region_{a.region}_check.jpg (front and turned crops), then: "
          f"python region.py train --region {a.region}")


# ---------------------------------------------------------------- makeup + photo-like changes (on the graphics card)
def polyline(P, steps=6):
    """B,K,2 points -> B,(K-1)*steps+1,2 densely sampled along the line"""
    import torch
    t = torch.linspace(0, 1, steps + 1, device=P.device)[:-1]
    seg = P[:, :-1, None, :] * (1 - t[None, None, :, None]) + P[:, 1:, None, :] * t[None, None, :, None]
    return torch.cat([seg.flatten(1, 2), P[:, -1:]], 1)


def dist_to(P, H, W, scale=2):
    """distance (in full-res pixels) from every pixel to the nearest of the points P (B,N,2), computed at 1/scale"""
    import torch
    import torch.nn.functional as F
    h, w = H // scale, W // scale
    yy, xx = torch.meshgrid(torch.arange(h, device=P.device), torch.arange(w, device=P.device), indexing="ij")
    g = torch.stack([xx, yy], -1).float().view(1, -1, 2) * scale + scale / 2
    d = torch.cdist(g.expand(P.shape[0], -1, -1), P.float()).min(2).values.view(-1, 1, h, w)
    return F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False)


def makeup(x, pts, p=0.6, brows=True):
    """x: B,3,H,W (0-1), pts: B,56,2 crop coordinates -> x with random makeup painted on (each kind independently)"""
    import torch
    B, _, H, W = x.shape
    dev = x.device
    r = lambda *s: torch.rand(*s, device=dev)
    who = r(B) < p
    if not who.any():
        return x
    x = x.clone()
    xs, P = x[who], pts[who]
    n = xs.shape[0]
    d_eye = (((P[:, S_RUP][:, 4] - P[:, S_LUP][:, 4]) ** 2).sum(1).sqrt()).view(n, 1, 1, 1)   # px between eyes

    def paint(mask, colour, alpha):
        nonlocal xs
        a_ = (mask * alpha.view(n, 1, 1, 1)).clamp(0, 1)
        xs = xs * (1 - a_) + colour.view(n, 3, 1, 1) * a_

    def wing(up):   # extend the upper lid line a little past the outer corner, slightly upwards
        o, nx = up[:, 0], up[:, 1]
        v = o - nx
        v = v / v.norm(dim=1, keepdim=True).clamp(min=1e-3)
        tip = o + v * (d_eye.view(n, 1) * 0.08 * r(n, 1)) + torch.stack([torch.zeros(n, device=dev),
                                                                        -d_eye.view(n) * 0.03 * r(n)], 1)
        return torch.cat([tip[:, None], up], 1)
    # eyeliner on the upper lids
    on = (r(n) < 0.55).float()
    if on.any():
        line = torch.cat([polyline(wing(P[:, S_RUP])), polyline(wing(P[:, S_LUP]))], 1)
        th = d_eye * (0.008 + 0.025 * r(n, 1, 1, 1))
        m = torch.sigmoid((th - dist_to(line, H, W)) / 0.8)
        paint(m, 0.03 + 0.1 * r(n, 1).expand(n, 3), on * (0.5 + 0.45 * r(n)))
    # lower liner / mascara shadow
    on = (r(n) < 0.3).float()
    if on.any():
        line = torch.cat([polyline(P[:, S_RLO]), polyline(P[:, S_LLO])], 1)
        th = d_eye * (0.005 + 0.012 * r(n, 1, 1, 1))
        m = torch.sigmoid((th - dist_to(line, H, W)) / 0.8)
        paint(m, 0.05 + 0.15 * r(n, 1).expand(n, 3), on * (0.3 + 0.5 * r(n)))
    # eyeshadow: soft tint between lid and brow
    on = (r(n) < 0.4).float()
    if on.any():
        sh = []
        for up, bl in ((S_RUP, S_RBL), (S_LUP, S_LBL)):
            bc = P[:, bl].mean(1, keepdim=True)
            sh.append(P[:, up] * 0.65 + bc * 0.35)
        line = polyline(torch.cat(sh, 1)[:, :9]), polyline(torch.cat(sh, 1)[:, 9:])
        dd = torch.minimum(dist_to(line[0], H, W), dist_to(line[1], H, W))
        sig = d_eye * (0.04 + 0.05 * r(n, 1, 1, 1))
        m = torch.exp(-dd ** 2 / (2 * sig ** 2))
        base = torch.stack([0.45 + 0.3 * r(n), 0.3 + 0.2 * r(n), 0.25 + 0.25 * r(n)], 1)      # browns/pinks/greys
        paint(m, base, on * (0.1 + 0.25 * r(n)))
    # brows filled in / darkened (off with --no-brow-makeup: a filled-in brow is part of how she looks)
    on = (r(n) < (0.4 if brows else 0.0)).float()
    if on.any():
        mid = torch.cat([polyline((P[:, S_RBU] + P[:, S_RBL]) / 2), polyline((P[:, S_LBU] + P[:, S_LBL]) / 2)], 1)
        th = (P[:, S_RBU] - P[:, S_RBL]).norm(dim=2).mean(1).view(n, 1, 1, 1) * (0.12 + 0.18 * r(n, 1, 1, 1))
        m = torch.sigmoid((th - dist_to(mid, H, W)) / 2.0)
        paint(m, 0.1 + 0.15 * r(n, 1).expand(n, 3), on * (0.15 + 0.3 * r(n)))
    x[who] = xs.clamp(0, 1)
    return x


def lipstick(x, pts, p=0.6):
    """x: B,3,H,W (0-1), pts: B,40,2 (outer then inner lip outline) -> random lipstick / lip liner, sometimes drawn a
    little outside the natural line (overlining), sometimes glossy"""
    import cv2
    import torch
    B, _, H, W = x.shape
    dev = x.device
    who = np.where(np.random.rand(B) < p)[0]
    if not len(who):
        return x
    P = pts[who].cpu().numpy()
    masks = np.zeros((len(who), H, W), np.float32)
    for k, q in enumerate(P):
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [np.round(q[:20]).astype(np.int32)], 255)
        width = np.linalg.norm(q[0] - q[10])
        grow = int(round(np.random.rand() * 0.04 * width)) if np.random.rand() < 0.4 else 0       # overlining
        if grow:
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
        if np.random.rand() < 0.5:                                                            # mouth gap left bare
            cv2.fillPoly(m, [np.round(q[20:]).astype(np.int32)], 0)
        masks[k] = cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), 1.0 + np.random.rand() * 1.5)
    m = torch.from_numpy(masks).to(dev)[:, None]
    n = len(who)
    r = lambda *s: torch.rand(*s, device=dev)
    kind = r(n)
    red = torch.stack([0.55 + 0.35 * r(n), 0.05 + 0.15 * r(n), 0.1 + 0.15 * r(n)], 1)            # reds
    pink = torch.stack([0.7 + 0.25 * r(n), 0.35 + 0.2 * r(n), 0.45 + 0.2 * r(n)], 1)             # pinks
    nude = torch.stack([0.6 + 0.2 * r(n), 0.4 + 0.15 * r(n), 0.35 + 0.15 * r(n)], 1)            # nudes/browns
    col = torch.where((kind < 0.35)[:, None], red, torch.where((kind < 0.65)[:, None], pink, nude))
    alpha = (0.3 + 0.5 * r(n)).view(n, 1, 1, 1)
    w_ = torch.from_numpy(who).to(dev)
    xs = x[w_]
    lum = xs.mean(1, keepdim=True)
    tinted = col.view(n, 3, 1, 1) * (0.6 + 0.8 * lum)                     # keep the lips' own shading
    xs = xs * (1 - m * alpha) + tinted * m * alpha
    gloss = (r(n) < 0.3).float().view(n, 1, 1, 1)                          # a soft highlight on the lower lip
    xs = xs + gloss * m * (lum > lum.flatten(1).quantile(0.9, dim=1).view(n, 1, 1, 1)).float() * 0.15
    x = x.clone()
    x[w_] = xs.clamp(0, 1)
    return x


def paint_region(x, pts, R, p, brows=True):
    if R["paint"] == "eyes":
        return makeup(x, pts, p, brows)
    if R["paint"] == "lips":
        return lipstick(x, pts, p)
    return x


def photo_like(x, out_w, out_h, train=True):
    """B,3,H+2M,W+2M (0-1) -> B,3,out_h,out_w: slight misalignment, side lighting, brightness/contrast/colour, blur,
    noise. train=False: plain centre crop."""
    import torch
    import torch.nn.functional as F
    n, _, Hc, Wc = x.shape
    dev = x.device
    if not train:
        return x[:, :, MARGIN:MARGIN + out_h, MARGIN:MARGIN + out_w]
    r = lambda *s: torch.rand(*s, device=dev)
    ang = (r(n) - 0.5) * 2 * np.radians(3)
    sc = 1 + (r(n) - 0.5) * 0.1
    kx, ky = out_w / Wc / sc, out_h / Hc / sc
    th = torch.stack([torch.stack([kx * torch.cos(ang), -ky * torch.sin(ang) * Hc / Wc, (r(n) - 0.5) * 0.05], 1),
                      torch.stack([kx * torch.sin(ang) * Wc / Hc, ky * torch.cos(ang), (r(n) - 0.5) * 0.06], 1)], 1)
    grid = F.affine_grid(th, (n, 3, out_h, out_w), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    d = r(n) * 2 * np.pi
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, out_h, device=dev), torch.linspace(-1, 1, out_w, device=dev),
                            indexing="ij")
    ramp = torch.cos(d).view(n, 1, 1) * xx + torch.sin(d).view(n, 1, 1) * yy
    x = x * (1 + ramp.unsqueeze(1) * (r(n) * 0.3).view(n, 1, 1, 1))
    m = x.mean((1, 2, 3), keepdim=True)
    x = (x - m) * (1 + (r(n, 1, 1, 1) - 0.5) * 0.5) + m
    x = x * (1 + (r(n, 1, 1, 1) - 0.5) * 0.4) * (1 + (r(n, 3, 1, 1) - 0.5) * 0.12)
    blur = r(n) < 0.5
    if blur.any():
        sig = (r(int(blur.sum())) * 2.0).clamp(min=0.3)
        t = torch.arange(-4, 5, device=dev).float()
        k = torch.exp(-(t.view(1, -1) ** 2) / (2 * sig.view(-1, 1) ** 2))
        k = k / k.sum(1, keepdim=True)
        xb = x[blur]
        nb = xb.shape[0]
        xb = xb.reshape(1, nb * 3, out_h, out_w)
        xb = F.conv2d(F.pad(xb, (4, 4, 0, 0), mode="replicate"), k.view(nb, 1, 1, 9).repeat_interleave(3, 0), groups=nb * 3)
        xb = F.conv2d(F.pad(xb, (0, 0, 4, 4), mode="replicate"), k.view(nb, 1, 9, 1).repeat_interleave(3, 0), groups=nb * 3)
        x[blur] = xb.view(nb, 3, out_h, out_w)
    x = x + torch.randn_like(x) * (r(n) * 0.03).view(n, 1, 1, 1)
    return x.clamp(0, 1)


# ---------------------------------------------------------------- network
def build_net(model, G, nt, eye_colour):
    import torch
    import torch.nn as nn
    from transformers import AutoModel

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.bb = AutoModel.from_pretrained(model)
            d = s.bb.config.hidden_size * 4
            s.trunk = nn.Sequential(nn.Linear(d, 1024), nn.GELU(), nn.Dropout(0.1), nn.Linear(1024, 1024), nn.GELU())
            s.val = nn.Linear(1024, G)
            s.col = nn.Linear(1024, 2) if eye_colour else None
            s.cls = nn.ModuleDict({str(j): nn.Linear(1024, k) for j, k in nt.items()})

        def forward(s, x):                       # x: B,2,3,H,W (front, turned)
            B = x.shape[0]
            h = s.bb(pixel_values=x.flatten(0, 1)).last_hidden_state
            f = torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).view(B, -1)
            z = s.trunk(f)
            return s.val(z), (s.col(z) if s.col is not None else None), {j: c(z) for j, c in s.cls.items()}
    return Net()


# ---------------------------------------------------------------- train + test
def train(a):
    import torch
    import torch.nn as nn
    import probe3d
    R = region_of(a)
    W, H = R["out"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using", torch.cuda.get_device_name(0) if dev == "cuda" else "the processor (very slow - stop and tell me)")
    # the bridge's held-out faces are the test faces; everything else with lined-up crops is training
    rows, te_ids = probe3d.held_out(a.data)
    folders = {}
    for d, nm in all_rows(a.data):
        cp, pp, dp = cache_paths(d, a.region)
        if not os.path.exists(dp) or int(open(dp).read()) < min(len(nm), int(open(os.path.join(d, "_f3_done.txt")).read())):
            sys.exit(f"{d}: run  python region.py prepare --region {a.region}  first (or let it finish).")
        folders[d] = {"img": np.load(cp, mmap_mode="r"), "pts": np.load(pp, mmap_mode="r"),
                      "ok": np.load(os.path.join(d, f"_reg_{a.region}_ok.npy"))}
    rows = [r_ for r_ in rows if folders[r_[0]]["ok"][r_[1], 0]]
    te_set = set(int(g) for g in te_ids)
    # held_out() numbers faces in its own list; rebuild that numbering for the rows kept here
    allrows, _ = probe3d.held_out(a.data)
    key = {(r_[0], r_[1]): g for g, r_ in enumerate(allrows)}
    gid = np.array([key[(r_[0], r_[1])] for r_ in rows])
    is_te = np.array([g in te_set for g in gid])
    if a.limit:
        keep = np.where(~is_te)[0][:a.limit]
        sel = np.sort(np.concatenate([keep, np.where(is_te)[0][:max(50, a.limit // 10)]]))
        rows, gid, is_te = [rows[i] for i in sel], gid[sel], is_te[sel]
    ck = torch.load(os.path.join(HERE, a.model_dna) if not os.path.isabs(a.model_dna) else a.model_dna,
                    map_location="cpu", weights_only=False)
    names = [os.path.join(r_[0], r_[2]) for r_ in rows]
    tg = tr.Targets("", names, fixed=(ck["genes"], ck["tmpl"], ck["kind"]))
    gsel = [j for j, g in enumerate(tg.genes) if R["genes"](g)]
    if not gsel:
        sys.exit("No settings matched this region.")
    sl = [j for j in gsel if tg.kind[tg.genes[j]] != "choice"]
    chs = [j for j in gsel if tg.kind[tg.genes[j]] == "choice"]
    print(f"{len(rows)} faces ({(~is_te).sum()} training, {is_te.sum()} held out). Region '{a.region}': "
          f"{len(sl)} sliders, {len(chs)} button settings" + (", eye colour" if R.get("eye_colour") else ""))
    print("  " + ", ".join(tg.genes[j] for j in gsel))
    trn, te = np.where(~is_te)[0], np.where(is_te)[0]
    Vall, Mall, Tall = tg.V[:, gsel], tg.M[:, gsel], tg.T[:, gsel]
    mu = np.array([Vall[trn][Mall[trn, k] > 0, k].mean() if (Mall[trn, k] > 0).any() else 0.5 for k in range(len(gsel))], np.float32)
    sd = np.array([max(Vall[trn][Mall[trn, k] > 0, k].std(), 0.02) if (Mall[trn, k] > 0).any() else 1.0
                   for k in range(len(gsel))], np.float32)
    ecol = tg.C[:, 4:6]
    cmu, csd = ecol[trn].mean(0), np.maximum(ecol[trn].std(0), 0.02)
    nt = {k: len(tg.tmpl[tg.genes[j]]) for k, j in enumerate(gsel) if j in chs}
    net = build_net(a.model, len(gsel), nt, R.get("eye_colour")).to(dev)
    tag = f"region_{a.region}" + (f"_{a.name}" if a.name else "")
    res = os.path.join(HERE, "results")
    os.makedirs(res, exist_ok=True)
    ckpt = os.path.join(res, tag + "_progress.pt")
    opt = torch.optim.AdamW([{"params": net.bb.parameters(), "lr": a.lr_backbone},
                             {"params": [p for n_, p in net.named_parameters() if not n_.startswith("bb.")], "lr": a.lr_head}],
                            weight_decay=0.02)
    steps = a.epochs * (len(trn) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr_backbone, a.lr_head], total_steps=max(steps, 1), pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    start_ep = 0
    if os.path.exists(ckpt) and not a.restart:
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        if st.get("n") == len(rows) and st.get("epochs") == a.epochs and st.get("model") == a.model:
            net.load_state_dict(st["state"])
            opt.load_state_dict(st["opt"])
            sched.load_state_dict(st["sched"])
            start_ep = st["epoch"]
            print(f"Carrying on from epoch {start_ep} of {a.epochs} (saved progress).")
        else:
            print("Saved progress is from different settings - starting over (use the same settings to carry on).")
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    Vn = torch.from_numpy(((Vall - mu) / sd).astype(np.float32)).to(dev)
    Mt = torch.from_numpy(Mall.astype(np.float32)).to(dev)
    Tt = torch.from_numpy(Tall).to(dev)
    Ct = torch.from_numpy(((ecol - cmu) / csd).astype(np.float32)).to(dev)
    ce = nn.CrossEntropyLoss(ignore_index=-1)
    sl_k = [k for k, j in enumerate(gsel) if j in sl]

    def load(ids):
        img = np.stack([folders[rows[i][0]]["img"][rows[i][1]] for i in ids])         # B,2,h,w,3
        pts = np.stack([folders[rows[i][0]]["pts"][rows[i][1]] for i in ids])         # B,2,56,2
        okt = np.array([folders[rows[i][0]]["ok"][rows[i][1], 1] for i in ids])
        img[~okt, 1] = img[~okt, 0]                                                   # no turned face: repeat front
        pts[~okt, 1] = pts[~okt, 0]
        return img, pts

    def prep(ids, mode, seed=None):
        """mode: 'train' (makeup + photo-like), 'clean', 'photo' (fixed-seed makeup + photo-like, for testing)"""
        img, pts = load(ids)
        g = None
        if seed is not None:
            g = torch.get_rng_state(), (torch.cuda.get_rng_state() if dev == "cuda" else None)
            torch.manual_seed(seed)
        x = torch.from_numpy(img).to(dev).permute(0, 1, 4, 2, 3).float().flatten(0, 1) / 255     # 2B,3,h,w
        p = torch.from_numpy(pts).to(dev).flatten(0, 1)
        if mode in ("train", "photo"):
            with torch.autocast(dev, enabled=False):
                x = paint_region(x, p, R, a.makeup, not a.no_brow_makeup)
                x = photo_like(x, W, H, True)
        else:
            x = photo_like(x, W, H, False)
        if g is not None:
            torch.set_rng_state(g[0])
            if g[1] is not None:
                torch.cuda.set_rng_state(g[1])
        x = (x - mean) / std
        return x.view(len(ids), 2, 3, H, W)

    def predict(ids, mode):
        net.eval()
        vs, cs, ps = [], [], {k: [] for k in nt}
        with torch.no_grad(), torch.autocast(dev, enabled=dev == "cuda"):
            for k in range(0, len(ids), a.batch):
                v, c, l = net(prep(ids[k:k + a.batch], mode, seed=1000 + k if mode == "photo" else None))
                vs.append(v.float().cpu())
                if c is not None:
                    cs.append(c.float().cpu())
                for kk in nt:
                    ps[kk].append(torch.softmax(l[str(kk)].float(), 1).cpu())
        v = np.clip(torch.cat(vs).numpy() * sd + mu, 0, 1)
        c = np.clip(torch.cat(cs).numpy() * csd + cmu, 0, 1) if cs else None
        return v, c, {kk: torch.cat(x).numpy() for kk, x in ps.items()}

    def read(ids, v, k):
        m = Mall[ids, k] > 0
        if m.sum() < 10:
            return np.nan
        base = np.abs(Vall[trn][Mall[trn, k] > 0, k].mean() - Vall[ids][m, k]).mean()
        if base * 255 < 2:
            return np.nan
        return 1 - np.abs(v[m, k] - Vall[ids][m, k]).mean() / base

    t0 = time.time()
    for ep in range(start_ep, a.epochs):
        net.train()
        perm = np.random.default_rng(ep).permutation(trn)
        tot, nb = 0.0, 0
        for k in range(0, len(perm) - a.batch + 1, a.batch):
            ids = perm[k:k + a.batch]
            it = torch.from_numpy(ids).to(dev)
            x = prep(ids, "train")
            with torch.autocast(dev, enabled=dev == "cuda"):
                v, c, l = net(x)
            v = v.float()
            loss = (((v - Vn[it]) ** 2) * Mt[it]).sum() / Mt[it].sum().clamp(min=1)
            if c is not None:
                loss = loss + 0.5 * ((c.float() - Ct[it]) ** 2).mean()
            if nt:
                loss = loss + 0.5 * sum(ce(l[str(kk)].float(), Tt[it, kk]) for kk in nt) / len(nt)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if sched.last_epoch < sched.total_steps - 1:
                sched.step()
            tot += loss.item()
            nb += 1
            if nb % 200 == 0:
                print(f"    epoch {ep + 1}: {nb * a.batch} / {len(perm)} faces, {(time.time() - t0) / 60:.0f} min")
        chk = te[:min(len(te), 1000)]
        vt, _, _ = predict(chk, "clean")
        rd = np.nanmedian([read(chk, vt, k) for k in sl_k])
        print(f"  epoch {ep + 1}/{a.epochs}: loss {tot / max(1, nb):.4f}, held-out region sliders read {rd * 100:.0f}% "
              f"(median), {(time.time() - t0) / 60:.0f} min")
        torch.save({"state": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep + 1,
                    "n": len(rows), "epochs": a.epochs, "model": a.model}, ckpt)
    torch.save({"state": net.state_dict(), "region": a.region, "genes_idx": gsel, "genes": [tg.genes[j] for j in gsel],
                "nt": nt, "mu": mu, "sd": sd, "cmu": cmu, "csd": csd, "model": a.model, "out": R["out"],
                "x": R["x"], "y": R["y"], "eye_colour": R.get("eye_colour"),
                "ratio": float(np.mean([np.load(os.path.join(d, f"_reg_{a.region}_ratio.npy"))[0] for d in folders]))},
               os.path.join(res, tag + ".pt"))
    report(a, tg, rows, te, gsel, sl, nt, predict, read, Vall, Mall, Tall, ecol, res, tag)


def report(a, tg, rows, te, gsel, sl, nt, predict, read, Vall, Mall, Tall, ecol, res, tag):
    """the specialist against the current readers, on held-out faces none of them trained on"""
    import torch
    import bridge2 as B2
    import photo_to_dna as P
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("Comparing with the current readers on the held-out faces ...")
    vs_c, cs_c, ps_c = predict(te, "clean")
    vs_p, _, ps_p = predict(te, "photo")
    # the main network: only on faces it never trained on (its own held-out tenth, and all of the second folder)
    ckm_path = os.path.join(HERE, a.model_dna) if not os.path.isabs(a.model_dna) else a.model_dna
    mnet, mck, _ = P.load_model(ckm_path, dev)
    fair = np.zeros(len(te), bool)
    main_v = np.zeros((len(te), len(tg.genes)), np.float32)
    main_p = {}
    first = os.path.join(HERE, a.data[0]) if not os.path.isabs(a.data[0]) else a.data[0]
    n_first = len(np.load(os.path.join(first, tr.cache_name(224, "_names.npy"))))
    main_te = set(np.random.default_rng(0).permutation(n_first)[:max(24, n_first // 10)].tolist())
    crops = {}
    pairs, where = [], []
    for k, i in enumerate(te):
        d, fi, _ = rows[i]
        if d == first and fi not in main_te:
            continue
        fair[k] = True
        if d not in crops:
            crops[d] = np.load(os.path.join(d, tr.cache_name(224, ".npy")), mmap_mode="r")
        c = crops[d][fi]
        o = (c.shape[1] - 224) // 2
        pairs.append((np.ascontiguousarray(c[0, o:o + 224, o:o + 224]), np.ascontiguousarray(c[1, o:o + 224, o:o + 224])))
        where.append(k)
    if pairs:
        mv, _, mp = P.run(mnet, mck, pairs, dev)
        main_v[where] = mv
        for j, pr in mp.items():
            main_p.setdefault(j, np.zeros((len(te), pr.shape[1]), np.float32))[where] = pr
    del mnet
    # the bridge (clean, and read like photos from its photo-like versions)
    bnet, bck = B2.load(os.path.join(HERE, a.bridge) if not os.path.isabs(a.bridge) else a.bridge, dev)
    types = bck["feats"]
    mm = {}

    def bridge_read(i, versions):
        d, fi, _ = rows[i]
        if d not in mm:
            mm[d] = {t: np.load(os.path.join(d, f"_f2_{t}.npy"), mmap_mode="r") for t in B2.IMAGE_TYPES if t in types}
            mm[d]["ok"] = np.load(os.path.join(d, "_f2_ok.npy")) & np.load(os.path.join(d, "_f3_ok.npy"))
            mm[d]["lm"] = np.load(os.path.join(d, "_f3_lm.npy"), mmap_mode="r")
        m = mm[d]

        def rows_for(view):
            geo = B2.geo_feature(np.asarray(m["lm"][fi, view], np.float64), bck, view) if "geo" in types else None
            return np.array([np.concatenate([geo if t == "geo" else np.asarray(m[t][fi, v, view], np.float32)
                                             for t in types]) for v in versions])
        f = rows_for(0)
        t = rows_for(1) if m["ok"][fi, 1] else np.zeros((0, f.shape[1]), np.float32)
        return B2.predict_person_full(bnet, bck, f, t, dev)
    br_c = [bridge_read(i, [0]) for i in te]
    nv = next(iter(mm.values()))[[t for t in types if t != "geo"][0]].shape[1]
    br_p = [bridge_read(i, list(range(1, nv))) for i in te]
    bv_c = np.array([b[0] for b in br_c])[:, gsel]
    bv_p = np.array([b[0] for b in br_p])[:, gsel]
    mv_s = main_v[:, gsel]
    mix = 0.5 * mv_s + 0.5 * bv_c
    te_f = np.where(fair)[0]
    L = [f"Region '{a.region}' specialist - {len(te)} held-out game faces ({fair.sum()} that the main network also "
         f"never saw; the 'main' and 'mix50' columns use only those).",
         "read = how much better than guessing the average (higher is better; the training reports use the same measure)",
         "'photo' columns: the same faces with fake makeup and photo-like lighting/blur, read like photos.", ""]
    sl_k = [k for k, j in enumerate(gsel) if j in sl]
    # best blend of mix50 and the specialist per slider, chosen on one half and scored on the other
    half = np.zeros(len(te_f), bool)
    half[np.random.default_rng(5).permutation(len(te_f))[:len(te_f) // 2]] = True
    blend = np.zeros((len(te_f), len(gsel)), np.float32)
    wsel = {}
    for k in sl_k:
        for h in (True, False):
            fit, use = te_f[half == h], te_f[half != h]
            best = max((0.0, 0.25, 0.5, 0.75, 1.0), key=lambda w: np.nan_to_num(
                read(te[fit], (1 - w) * mix[fit] + w * vs_c[fit], k), nan=-9))
            blend[half != h, k] = ((1 - best) * mix[use] + best * vs_c[use])[:, k]
            wsel.setdefault(k, []).append(best)
    L.append(f"{'setting':38s} {'mix50':>6s} {'main':>6s} {'bridge':>6s} {'SPEC':>6s} {'blend':>6s}   "
             f"{'bridge photo':>12s} {'SPEC photo':>10s}  blend share")
    cols = {"mix": [], "main": [], "bridge": [], "spec": [], "blend": [], "bp": [], "sp": []}
    tf = te[te_f]
    for k in sl_k:
        r_ = {"mix": read(tf, mix[te_f], k), "main": read(tf, mv_s[te_f], k), "bridge": read(te, bv_c, k),
              "spec": read(te, vs_c, k), "blend": read(tf, blend, k), "bp": read(te, bv_p, k), "sp": read(te, vs_p, k)}
        if np.isnan(r_["spec"]):
            continue
        for c_, v_ in r_.items():
            cols[c_].append(v_)
        f_ = lambda v_: f"{v_ * 100:5.0f}%" if np.isfinite(v_) else "    - "
        L.append(f"  {tg.genes[gsel[k]]:36s} {f_(r_['mix'])} {f_(r_['main'])} {f_(r_['bridge'])} {f_(r_['spec'])} "
                 f"{f_(r_['blend'])}   {f_(r_['bp']):>12s} {f_(r_['sp']):>10s}  {np.mean(wsel[k]):.2f}")
    med = {c_: np.nanmedian(v_) * 100 for c_, v_ in cols.items()}
    L.insert(4, f"MEDIAN over {len(cols['spec'])} sliders:  mix50 {med['mix']:.0f}%   main {med['main']:.0f}%   "
                f"bridge {med['bridge']:.0f}%   SPECIALIST {med['spec']:.0f}%   best blend {med['blend']:.0f}%\n"
                f"  read like photos (makeup etc.):  bridge {med['bp']:.0f}%   SPECIALIST {med['sp']:.0f}%\n"
                f"  specialist beats mix50 by 5+ points on {sum(s - m > 0.05 for s, m in zip(cols['spec'], cols['mix']))}"
                f" of {len(cols['spec'])} sliders; worse by 5+ on {sum(m - s > 0.05 for s, m in zip(cols['spec'], cols['mix']))}\n")
    # buttons and eye colour
    if nt:
        L += ["", "Buttons (right button chosen, held-out):  specialist / bridge / main (main on its unseen faces)"]
        for kk in nt:
            j = gsel[kk]
            tt = Tall[te, kk]
            ok = tt >= 0
            acc_s = (ps_c[kk].argmax(1) == tt)[ok].mean()
            acc_b = (np.array([b[1][j].argmax() if j in b[1] else -1 for b in br_c]) == tt)[ok].mean()
            okf = ok & fair
            acc_m = (main_p[j].argmax(1) == tt)[okf].mean() if j in main_p and okf.any() else np.nan
            L.append(f"  {tg.genes[j]:36s} {acc_s * 100:4.0f}% / {acc_b * 100:4.0f}% / {acc_m * 100:4.0f}%")
    if cs_c is not None:
        e = np.abs(cs_c - ecol[te]).mean(0) * 255
        g = np.abs(ecol[te].mean(0) - ecol[te]).mean(0) * 255
        L += ["", f"Eye colour error (0-255): specialist {e[0]:.1f} / {e[1]:.1f}   (guessing the average {g[0]:.1f} / {g[1]:.1f})"]
    L += ["", "How to read this: the specialist is worth taking to photos if it (or the best blend) beats mix50 by",
          "several points overall and clearly on the eye/brow sliders, AND its 'photo' column doesn't collapse."]
    rep = "\n".join(L)
    print("\n" + rep)
    open(os.path.join(res, tag + "_report.txt"), "w").write(rep + "\n")
    print(f"\nSaved results\\{tag}.pt and results\\{tag}_report.txt")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("prepare", "train"):
        p = sub.add_parser(c)
        p.add_argument("--region", default="eyes", choices=list(REGIONS))
        p.add_argument("--data", nargs="+", default=DATA)
        if c == "train":
            p.add_argument("--epochs", type=int, default=12)
            p.add_argument("--batch", type=int, default=32)
            p.add_argument("--model", default="facebook/dinov2-small")
            p.add_argument("--lr-backbone", type=float, default=3e-5)
            p.add_argument("--lr-head", type=float, default=1e-3)
            p.add_argument("--makeup", type=float, default=0.6, help="share of training crops with fake makeup")
            p.add_argument("--no-brow-makeup", action="store_true",
                           help="eyes: don't paint filled-in brows (so it reads brows the way they look)")
            p.add_argument("--limit", type=int, default=0, help="quick test: only this many training faces")
            p.add_argument("--name", default="")
            p.add_argument("--restart", action="store_true", help="ignore saved progress and start over")
            p.add_argument("--model-dna", default=os.path.join("results", "model_29996_v3_base_photo.pt"))
            p.add_argument("--bridge", default=os.path.join("results", "bridge.pt"))
    a = ap.parse_args()
    prepare(a) if a.cmd == "prepare" else train(a)


if __name__ == "__main__":
    main()
