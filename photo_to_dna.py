"""Real photos -> CK3 DNA, using the reverse network trained on game faces (train_reverse.py).

    python photo_to_dna.py --photos "C:\\CK3 Face Generator\\Samples\\Sample 1"
    python photo_to_dna.py --photos "C:\\CK3 Face Generator\\Samples"          (every sub-folder = one person)

What it does, per person:
  1. finds the face in every photo (MediaPipe face mesh) and measures how far the head is turned;
  2. lines each photo up so its eye corners, nose, mouth and chin land where they sit in the game's front or turned
     crop (measured once from your game screenshots and saved in results\\photo_calib.json), then applies the same
     squeeze-to-square the training crops had - so a photo is framed exactly like a game face;
  3. front = photos facing the camera; turned = photos turned about as far as the game's turned view (photos turned
     the other way are mirrored);
  4. runs every front + turned pairing through the model and averages them into one DNA;
  5. writes, in results\\photos\\<person>\\:
        <person>.txt               the averaged DNA (on the female template: hair/clothes from the template)
        single\\<photo>.txt         one DNA per front photo (each averaged over the turned photos) - for the
                                   consistency check: do different photos give the same face?
        aligned_check.jpg          what the network actually sees, front and turned - check this first
        consistency.log            how much the photos disagree, per setting, next to how much faces vary in training
Screenshot them:  python autoshot.py folder results\\photos\\<person>   and   ...\\<person>\\single
No measuring targets, no score climbing: the photos go straight into the network, and that's all.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = (os.environ.get("CK3FIT_HOME") or os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import train_reverse as tr      # noqa: E402  (crop boxes, DNA reading/writing)

LMK = [33, 133, 362, 263, 1, 61, 291, 152, 10]      # eye corners, nose tip, mouth corners, chin, forehead
EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


# ---------------------------------------------------------------- face landmarks
class Mesh:
    TASK_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/"
                "face_landmarker.task")

    def __init__(s):
        try:
            import mediapipe as mp
        except Exception as e:
            sys.exit(f"MediaPipe isn't installed ({e}): python -m pip install mediapipe")
        s.mp, s.fm, s.tl = mp, None, None
        if hasattr(mp, "solutions"):                         # older MediaPipe
            s.fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=False,
                                                    min_detection_confidence=0.3)
        else:                                                # newer MediaPipe: Tasks API + small model file
            path = os.path.join(HERE, "face_landmarker.task")
            if not os.path.exists(path):
                import urllib.request
                print("Downloading MediaPipe's face landmark model (~4 MB, once) ...")
                urllib.request.urlretrieve(s.TASK_URL, path)
            v = mp.tasks.vision
            s.tl = v.FaceLandmarker.create_from_options(v.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=path), num_faces=1,
                min_face_detection_confidence=0.3))

    def full(s, rgb):
        """-> (468,3) all face points in pixels (x, y, and depth z on the x scale), or None."""
        h, w = rgb.shape[:2]
        if s.fm is not None:
            r = s.fm.process(rgb)
            if not r.multi_face_landmarks:
                return None
            lm = r.multi_face_landmarks[0].landmark
        else:
            r = s.tl.detect(s.mp.Image(image_format=s.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
            if not r.face_landmarks:
                return None
            lm = r.face_landmarks[0]
        return np.array([[lm[i].x * w, lm[i].y * h, lm[i].z * w] for i in range(468)], np.float32)

    def __call__(s, rgb, full=False):
        """-> (9,2) pixel coords of the key points, or None. full=True: (9,2) points and all 468 points."""
        L = s.full(rgb)
        if L is None:
            return (None, None) if full else None
        p = np.ascontiguousarray(L[LMK, :2])
        return (p, L) if full else p


def turn(p):
    """Head turn, same measure for game shots and photos: nose offset from the eye-corner midpoint, in eye-widths.
    0 = facing the camera; positive = face turned towards the picture's right (like the game's turned view)."""
    mid = (p[0] + p[3]) / 2
    iod = np.linalg.norm(p[3] - p[0]) + 1e-6
    return float((p[4][0] - mid[0]) / iod)


def load_rgb(path, max_side=1600):
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if max(im.size) > max_side:
        k = max_side / max(im.size)
        im = im.resize((round(im.size[0] * k), round(im.size[1] * k)))
    return np.asarray(im)


# ---------------------------------------------------------------- calibration on game screenshots
def calibrate(data, mesh, n=300):
    from PIL import Image
    files = sorted(glob.glob(os.path.join(data, "*.jpg")))[::max(1, len(glob.glob(os.path.join(data, "*.jpg"))) // n)][:n]
    out = {}
    for name, box in (("front", tr.FRONT), ("turned", tr.TURNED)):
        pts, turns, size = [], [], None
        for f in files:
            im = Image.open(f).convert("RGB")
            W, H = im.size
            c = np.asarray(im.crop((int(W * box[0]), int(H * box[1]), int(W * box[2]), int(H * box[3]))))
            size = c.shape[1], c.shape[0]
            p = mesh(c)
            if p is not None:
                pts.append(p)
                turns.append(turn(p))
        if len(pts) < 20:
            sys.exit(f"Could only find the {name} face in {len(pts)} of {len(files)} game screenshots - send me this.")
        out[name] = {"points": np.mean(pts, 0).tolist(), "box": size, "turn": float(np.median(turns)),
                     "found": f"{len(pts)}/{len(files)}"}
        print(f"  game {name} view: face found in {len(pts)}/{len(files)}, turn {np.median(turns):+.2f}")
    return out


def align_cache(rgb, p, cal, out_size):
    """Similarity-transform the photo so its key points match the game crop, at the crop's own size, then squeeze to
    the training cache size (out_size + 16), exactly like the game screenshots in the training cache."""
    import cv2
    from PIL import Image
    dst = np.array(cal["points"], np.float32)
    M, _ = cv2.estimateAffinePartial2D(p, dst, method=cv2.LMEDS)
    w, h = cal["box"]
    warped = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
    cache = out_size + 16
    return np.asarray(Image.fromarray(warped).resize((cache, cache)))


def align(rgb, p, cal, out_size):
    """align_cache, then the same centre crop the model was evaluated on."""
    sq = align_cache(rgb, p, cal, out_size)
    o = (sq.shape[0] - out_size) // 2
    return sq[o:o + out_size, o:o + out_size]


# ---------------------------------------------------------------- model
def load_model(path, dev):
    import torch
    import torch.nn as nn
    from transformers import AutoModel
    ck = torch.load(path, map_location="cpu", weights_only=False)
    genes, tmpl, kind = ck["genes"], ck["tmpl"], ck["kind"]
    choice = [j for j, g in enumerate(genes) if kind[g] == "choice"]
    G = len(genes)

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.face = ck["model"] in tr.FACE_BACKBONES
            if s.face:                                # face-recognition backbone (train_reverse --model arcface)
                s.bb = tr.face_backbone(ck["model"])
                d = s.bb.out_dim * 2
            else:
                _cfg = os.path.join(HERE, "dinov2_config", ck["model"].replace("/", "_"))
                if os.path.isdir(_cfg):               # app: built from a config file, weights come from the .pt
                    from transformers import AutoConfig
                    s.bb = AutoModel.from_config(AutoConfig.from_pretrained(_cfg))
                else:
                    s.bb = AutoModel.from_pretrained(ck["model"])
                d = s.bb.config.hidden_size * 4
            s.trunk = nn.Sequential(nn.Linear(d, 1024), nn.GELU(), nn.Dropout(0.1), nn.Linear(1024, 1024), nn.GELU())
            s.val, s.col = nn.Linear(1024, G), nn.Linear(1024, 6)
            s.cls = nn.ModuleDict({str(j): nn.Linear(1024, len(tmpl[genes[j]])) for j in choice})
            s.collect, s.coral = None, None           # coral.py: photo features saved / photo features re-shaped

        def forward(s, x):
            B = x.shape[0]
            if s.face:
                f = s.bb(x.flatten(0, 1)).view(B, -1)
            else:
                h = s.bb(pixel_values=x.flatten(0, 1)).last_hidden_state
                f = torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).view(B, -1)
            if s.collect is not None:
                s.collect.append(f.float().cpu().numpy())
            if s.coral is not None:                   # move photo features onto the screenshots' pattern
                mp, W, ms, k = s.coral
                ff = f.float()
                f = (ff + k * (((ff - mp) @ W + ms) - ff)).to(f.dtype)
            z = s.trunk(f)
            return s.val(z), s.col(z), {j: c(z) for j, c in s.cls.items()}

    net = Net()
    net.load_state_dict(ck["state"])
    net.to(dev).eval()
    # a Targets-like object so train_reverse's DNA writer can be reused
    tg = tr.Targets.__new__(tr.Targets)
    tg.genes, tg.tmpl, tg.kind, tg.choice = genes, tmpl, kind, choice
    return net, ck, tg


def run(net, ck, pairs, dev):
    """pairs: list of (front 224x224x3, turned 224x224x3) uint8. -> per-pair values (0-1), colours, class probs."""
    import torch
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 1, 3, 1, 1)
    V, C, P = [], [], {}
    with torch.no_grad(), torch.autocast(dev, enabled=dev == "cuda"):
        for k in range(0, len(pairs), 32):
            xb = np.stack([np.stack(p) for p in pairs[k:k + 32]])
            if ck.get("view") == "strip":             # model trained on stripped pictures (strip.py): convert first
                if getattr(net, "_stripper", None) is None:
                    import strip
                    net._stripper = strip.Stripper(dev)
                xb = net._stripper(xb.reshape((-1,) + xb.shape[2:])).reshape(xb.shape)
            x = torch.from_numpy(xb).to(dev)
            x = x.permute(0, 1, 4, 2, 3).float() / 255
            if getattr(net, "fda", None) is not None:
                x = fda_apply(x, *net.fda)
            x = (x - mean) / std
            v, c, l = net(x)
            V.append(np.clip(v.float().cpu().numpy() * ck["sd"] + ck["mu"], 0, 1))
            C.append(np.clip(c.float().cpu().numpy() * ck["csd"] + ck["cmu"], 0, 1))
            for j, t in l.items():
                P.setdefault(int(j), []).append(torch.softmax(t.float(), 1).cpu().numpy())
    return np.concatenate(V), np.concatenate(C), {j: np.concatenate(x) for j, x in P.items()}


def fda_apply(x, ref_amp, b, strength):
    """Fourier Domain Adaptation (Yang & Soatto 2020): the photo's broad patterns (overall light, colour cast,
    contrast = the lowest frequencies) are swapped for the average screenshot's; edges and shapes stay the photo's.
    x: B,2,3,H,W in 0-1; ref_amp: 2,3,H,W (centred amplitude spectrum per view)."""
    import torch
    with torch.autocast(x.device.type, enabled=False):
        F_ = torch.fft.fftshift(torch.fft.fft2(x.float()), dim=(-2, -1))
        amp, pha = F_.abs(), F_.angle()
        H, W = x.shape[-2:]
        cy, cx = H // 2, W // 2
        lo = amp[..., cy - b:cy + b + 1, cx - b:cx + b + 1]
        rf = ref_amp[None, :, :, cy - b:cy + b + 1, cx - b:cx + b + 1]
        amp = amp.clone()
        amp[..., cy - b:cy + b + 1, cx - b:cx + b + 1] = lo + strength * (rf - lo)
        y = torch.fft.ifft2(torch.fft.ifftshift(torch.polar(amp, pha), dim=(-2, -1))).real
    return y.clamp(0, 1)


CORR_PATH = os.path.join(HERE, "results", "photo_correction.json")


def load_correction(tg):
    if not os.path.exists(CORR_PATH):
        return None
    c = json.load(open(CORR_PATH))
    gi = {g: j for j, g in enumerate(c["genes"])}
    G = len(tg.genes)
    a, b = np.zeros(G), np.ones(G)
    for j, g in enumerate(tg.genes):
        if g in gi:
            a[j], b[j] = c["a"][gi[g]], c["b"][gi[g]]
    return {"method": c["method"], "a": a, "b": b, "ca": np.array(c["col_a"]), "cb": np.array(c["col_b"])}


def apply_correction(corr, v, c):
    return np.clip(corr["a"] + corr["b"] * v, 0, 1), np.clip(corr["ca"] + corr["cb"] * c, 0, 1)


# ---------------------------------------------------------------- region specialists (region.py)
def load_regions(names, dev):
    """[(name, net, ck, geometry, ratio)] for each results\\region_<name>.pt that exists"""
    import torch
    import region as RG
    out = []
    for nm in names:
        path = os.path.join(HERE, "results", f"region_{nm}.pt")
        if not os.path.exists(path):
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        base = RG.REGIONS[ck["region"]]
        geo = {"out": tuple(ck.get("out", base["out"])), "x": tuple(ck.get("x", base["x"])),
               "y": tuple(ck.get("y", base["y"])), "mk": base["mk"]}
        ratio = ck.get("ratio")
        if ratio is None:                             # older file: the ratio saved next to the training crops
            rp = os.path.join(HERE, "random_faces", "noise_female", f"_reg_{ck['region']}_ratio.npy")
            ratio = float(np.load(rp)[0]) if os.path.exists(rp) else None
        if ratio is None:
            print(f"(region_{nm}.pt: can't find its turned-view size setting - skipped)")
            continue
        net = RG.build_net(ck["model"], len(ck["genes"]), ck["nt"],
                           ck.get("eye_colour", ck["region"] == "eyes"))
        net.load_state_dict(ck["state"])
        out.append((nm, net.to(dev).eval(), ck, geo, ratio))
    return out


def region_crop(geo, rgb, L, ratio=None):
    import region as RG
    c, _ = RG.cut(rgb, np.asarray(L, np.float32), geo, ratio)
    W, H = geo["out"]
    return c[RG.MARGIN:RG.MARGIN + H, RG.MARGIN:RG.MARGIN + W]


def region_read(spec, fronts, turneds, dev):
    """front crops x turned crops -> averaged reading: values (0-1, the region's settings), button
    probabilities, eye colour (or None)"""
    import torch
    nm, net, ck, geo, _ = spec
    W, H = geo["out"]
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 1, 3, 1, 1)
    pairs = [(f, t) for f in fronts for t in (turneds or fronts[:1])]
    V, C, Pb = [], [], {}
    with torch.no_grad(), torch.autocast(dev, enabled=dev == "cuda"):
        for k in range(0, len(pairs), 16):
            x = torch.from_numpy(np.stack([np.stack(p_) for p_ in pairs[k:k + 16]])).to(dev)
            x = (x.permute(0, 1, 4, 2, 3).float() / 255 - mean) / std
            v, c, l = net(x)
            V.append(v.float().cpu().numpy())
            if c is not None:
                C.append(c.float().cpu().numpy())
            for kk, t in l.items():
                Pb.setdefault(int(kk), []).append(torch.softmax(t.float(), 1).cpu().numpy())
    v = np.clip(np.concatenate(V).mean(0) * ck["sd"] + ck["mu"], 0, 1)
    c = np.clip(np.concatenate(C).mean(0) * ck["csd"] + ck["cmu"], 0, 1) if C else None
    return v, {kk: np.concatenate(x).mean(0) for kk, x in Pb.items()}, c


def smile_measures(L):
    """(mouth opening, corner lift) relative to eye distance, in the face's own upright frame.
    corner lift > 0: the mouth corners sit higher than the middle of the mouth (a smile)."""
    er, el = (L[33, :2] + L[133, :2]) / 2, (L[362, :2] + L[263, :2]) / 2
    v = el - er
    iod = max(float(np.linalg.norm(v)), 1.0)
    ang = np.arctan2(v[1], v[0])
    R = np.array([[np.cos(-ang), -np.sin(-ang)], [np.sin(-ang), np.cos(-ang)]])
    P = (L[:, :2] - er) @ R.T
    opening = float(np.linalg.norm(P[13] - P[14]) / iod)
    lift = float(((P[13, 1] + P[14, 1]) / 2 - (P[61, 1] + P[291, 1]) / 2) / iod)
    return opening, lift


def colour_skip(found, pics, limit=10.0, keep_min=3):
    """photos to leave out of the COLOUR reading (they still count for face shape):
      - black-and-white photos, and any file with 'nocolour' / 'nocolor' in its name
      - photos whose skin is much darker, redder, more orange or more washed-out than the same woman's other photos
        (film stills, dim or tinted lighting): more than `limit` CIELAB units from her median
    Only judged against her own photos, so a woman photographed consistently loses none. Keeps at least keep_min."""
    import colour
    lab, out = {}, set()
    for f, (p, L) in found.items():
        nm = os.path.basename(f).lower()
        if "nocolour" in nm or "nocolor" in nm:
            out.add(f)
            continue
        if p is None:
            continue
        rgb = pics[f]
        if float(np.mean(rgb.max(2).astype(np.float32) - rgb.min(2))) < 4:     # no colour in the photo at all
            out.add(f)
            continue
        sk = colour.measure(rgb, L)[0:3]
        if np.isfinite(sk).all():
            lab[f] = colour.lab(sk[None].astype(np.float32))[0]
    if len(lab) - keep_min < 1:
        return out
    M = np.median(np.array(list(lab.values())), 0)
    off = {f: max(M[0] - x[0], x[1] - M[1], abs(x[2] - M[2])) for f, x in lab.items()}   # darker / redder / tint
    bad = sorted((f for f in off if off[f] > limit), key=lambda f: -off[f])
    while bad and len(lab) - len(bad) < keep_min:
        bad.pop()                                     # keep the least-off ones if too few would remain
    return out | set(bad)


def smile_skip(found, front_max, keep_min=5, keep_target=10):
    """found: {file: (p, L)} -> set of front photos to leave out because the mouth is open or smiling broadly,
    always keeping at least keep_min front photos, and only trimming smiles while more than keep_target remain"""
    fr = {f: smile_measures(L) for f, (p, L) in found.items() if p is not None and abs(turn(p)) <= front_max}
    if len(fr) <= keep_min:
        return set()
    order = sorted(fr, key=lambda f: fr[f][0] * 2 + max(fr[f][1], 0))        # least open / smiling first
    skip = {f for f in fr if fr[f][0] > 0.07}
    if len(fr) - len(skip) < keep_min:
        skip = set(order[keep_min:]) & skip
    rest = [f for f in order if f not in skip]
    kept = len(rest)
    if kept > keep_target:
        med = np.median([fr[f][1] for f in rest])
        for f in reversed(rest):                                   # broadest smiles first
            if kept <= keep_target:
                break
            if fr[f][1] > med + 0.03:
                skip.add(f)
                kept -= 1
    return skip


def pitch_of(p):
    """head tipped up/down, from the 9 key points (eye corners 0/3, nose 4, chin 7): where the nose sits between the
    eye line and the chin. Same measure as nuisance.py."""
    p = np.asarray(p, np.float64)
    eye_mid = (p[0] + p[3]) / 2
    return float((p[4][1] - eye_mid[1]) / max(p[7][1] - eye_mid[1], 1.0))


def pitch_skip(found, front_max, game, keep, already, keep_min=4):
    """--pitch-keep: of the usable front photos, keep the share 'keep' whose tilt is nearest the game camera's
    (at least keep_min); the rest are left out"""
    fr = {f: abs(pitch_of(p) - game) for f, (p, L) in found.items()
          if p is not None and abs(turn(p)) <= front_max and f not in already}
    n = max(keep_min, int(round(keep * len(fr))))
    return set(sorted(fr, key=fr.get)[n:]) if len(fr) > n else set()


BROKEN_EYES = ("normal_eyes_no_shadow", "normal_eyes_asian")      # these blank out the iris in the game


def finish_dna(text, lashes, sex="female"):
    """last touches on every DNA written: never the eye options that blank out the iris; dark eyelashes (the
    closest the game gets to mascara) unless --lashes read"""
    import re
    text = re.sub(r'\beye_accessory=\{\s*"(' + "|".join(BROKEN_EYES) + r')"\s+(\d+)\s+"[^"]+"\s+(\d+)\s*\}',
                  lambda m: f'eye_accessory={{ "normal_eyes" {m.group(2)} "normal_eyes" {m.group(3)} }}', text)
    if lashes == "dark":
        text = re.sub(r'\beyelashes_accessory=\{\s*"[^"]+"\s+(\d+)\s+"[^"]+"\s+(\d+)\s*\}',
                      lambda m: f'eyelashes_accessory={{ "dark_eyelashes" {m.group(1)} "dark_eyelashes" {m.group(2)} }}',
                      text)
    if sex == "male":                                 # clean-shaven, like the training faces (beards: not picked yet)
        text = re.sub(r'\bbeards=\{[^}]*\}', 'beards={ "no_beard" 0 "no_beard" 0 }', text)
        if re.search(r"custom_beards=\S+", text):
            text = re.sub(r"custom_beards=\S+", "custom_beards=male_empty", text)
        else:
            text = re.sub(r"(custom_hair=\S+)", lambda m: m.group(1) + "\n\t\t\tcustom_beards=male_empty", text, count=1)
    return text


def region_gene_ok(g, a):
    """should a region specialist set this setting? --region-only (if given) decides; otherwise --region-skip"""
    if a.region_only:
        return any(x in g for x in a.region_only)
    return not any(x in g for x in a.region_skip)


def apply_colfix(cf, c, eyecol=None, c_col=None):
    """cf: colour_fix.json -> the 6 colour values (hair x,y, skin x,y, eye x,y) corrected. Sources: 'net' the
    network's colour, 'meas' the colour measured from the photos' pixels, 'blend' their average, 'spec' the eye
    specialist"""
    out = np.array(c, np.float64).copy()
    for k, d in enumerate(cf["dims"]):
        x = c[k]
        if d["source"] == "spec":
            if eyecol is None:
                continue
            x = eyecol[k - 4]
        elif d["source"] in ("meas", "blend"):
            if c_col is None or not np.isfinite(c_col[k]):
                continue
            x = c_col[k] if d["source"] == "meas" else 0.5 * (c[k] + c_col[k])
        out[k] = np.clip(d["a"] + d["b"] * x, 8 / 255, 247 / 255)   # the palettes' edges hold odd colours (purple
    return out                                                       # hair at the very bottom of the hair palette)


def eye_open(L):
    w = (np.linalg.norm(L[33, :2] - L[133, :2]) + np.linalg.norm(L[362, :2] - L[263, :2])) / 2
    return float((np.linalg.norm(L[159, :2] - L[145, :2]) + np.linalg.norm(L[386, :2] - L[374, :2])) / 2 / max(w, 1))


def mouth_open(L):
    iod = np.linalg.norm(L[263, :2] - L[33, :2])
    return float(np.linalg.norm(L[13, :2] - L[14, :2]) / max(iod, 1))


# ---------------------------------------------------------------- test variants (--variants)
def write_variants(a, name, out, fronts, V, P, owner, bf, bt, bown, slide, cls, c, BR, tg, template, dev):
    """Extra versions of the mix50 face, all in one folder so one screenshot run compares them:
      _x125, _x150   every slider pushed 25% / 50% further from the average face
      _d50, _d100    each slider pushed out by half / all of the flattening measured on game faces (deshrink.py)
      _med           each front photo read on its own, then the middle value per slider (an odd photo can't drag it)
      _n5, _n10, _n15  built from only 5 / 10 / 15 of her front photos (same random pick every run)
    Buttons, strengths and colours stay as in mix50, except _n* (from the same photos) and _med (mix50's)."""
    w = a.bridge_weight[0]
    mu = np.asarray(BR[1]["mu"], np.float64)

    def mixed(sel):
        sel = np.asarray(sorted(sel))
        idx = np.where(np.isin(owner, sel))[0]
        rows = np.where(np.isin(bown, sel))[0]
        if not len(idx) or not len(rows):
            return None, None
        v_s = V[idx].mean(0)
        pm_s = {j: P[j][idx].mean(0) for j in P}
        vb_s, pb_s = BR[3].predict_person_full(BR[0], BR[1], bf[rows], bt, dev)
        vm = (1 - w) * v_s + w * vb_s
        cl = {j: int(((1 - w) * pm_s[j] + w * pb_s[j]).argmax()) if j in pb_s else int(pm_s[j].argmax()) for j in cls}
        return vm, cl

    def save(tag, vv, cl):
        open(os.path.join(out, f"{name}{tag}.txt"), "w", encoding="utf-8").write(
            tg.write(template, np.clip(vv, 0, 1), cl, c))
    for f in glob.glob(os.path.join(out, f"{name}_x[0-9][0-9]*.txt")) + \
            [os.path.join(out, f"{name}{t}.txt") for t in ("_d50", "_d100", "_med", "_n5", "_n10", "_n15")]:
        if os.path.exists(f):
            os.remove(f)
    allf = range(len(fronts))
    base, bcls = mixed(allf)
    if base is None:
        return
    ks = getattr(a, "mix_stretch", None) or [1.25, 1.5]
    for k in ks:
        save(f"_x{round(k * 100)}", np.where(slide, mu + k * (base - mu), base), bcls)
    if getattr(a, "mix_stretch", None) and not a.variants:
        return
    dp = os.path.join(HERE, "results", "deshrink.npz")
    if os.path.exists(dp):
        z = np.load(dp)
        fac = dict(zip([str(g) for g in z["genes"]], z["factor"]))
        F = np.array([fac.get(g, 1.0) for g in tg.genes])
        for s_, tag in ((0.5, "_d50"), (1.0, "_d100")):
            save(tag, np.where(slide, mu + F ** s_ * (base - mu), base), bcls)
    else:
        print("  (no results\\deshrink.npz - run python deshrink.py first for the _d50/_d100 versions)")
    per = [mixed([i])[0] for i in allf]
    per = np.array([p_ for p_ in per if p_ is not None])
    if len(per) >= 3:
        save("_med", np.where(slide, np.median(per, 0), base), bcls)
    rng = np.random.default_rng(len(fronts))
    order = rng.permutation(len(fronts))
    for n in (5, 10, 15):
        if len(fronts) > n:
            vv, cl = mixed(order[:n])
            if vv is not None:
                save(f"_n{n}", vv, cl)


# ---------------------------------------------------------------- per person
def person(folder, name, a, mesh, cal, net, ck, tg, dev, template):
    from PIL import Image
    files = sorted(f for f in glob.glob(os.path.join(folder, "*")) if f.lower().endswith(EXTS))
    size = int(ck.get("size", 224))
    gt = cal["turned"]["turn"]
    fronts, turneds, skipped, status, mirrors = [], [], [], [], {}
    audit = []
    BR = getattr(a, "_bridge", None)                  # (net, ck, feature reader, module) when --bridge is given
    RS = getattr(a, "_regions", None) or []           # region specialists
    rfront, rturn = [], {}                            # per front photo / per turned photo: {region: crop}, plus checks
    bfront, bturn, cms = [], [], []
    bowner = []                                       # which front photo each bridge front row came from

    def bvec(img, pp, LL, view):
        """bridge input numbers for one photo (None if its face points are needed but missing)"""
        if hasattr(BR[2], "photo_vec"):
            if "geo" in BR[1]["feats"] and LL is None:
                return None
            return BR[2].photo_vec(img, pp, LL, view)
        return BR[2]([BR[2].crop(img, pp)])[0]
    found, pics = {}, {}
    for f in files:                                   # find the face points once (also used to spot smiles)
        pics[f] = load_rgb(f)
        found[f] = mesh(pics[f], full=True)
    no_smile = set() if (a.keep_smiles or a.check) else smile_skip(found, a.front_max)
    game_pitch = pitch_of(cal["front"]["points"])
    no_pitch = set() if (a.check or not a.pitch_keep) else \
        pitch_skip(found, a.front_max, game_pitch, a.pitch_keep, no_smile)
    if no_pitch:
        print(f"\n{name}: left out {len(no_pitch)} front photo(s) tilted furthest from the game camera (--pitch-keep)")
    no_smile = no_smile | no_pitch
    pitches = {os.path.basename(f): pitch_of(p) for f, (p, L) in found.items() if p is not None}
    no_col = set() if getattr(a, "colour_all", False) else colour_skip(found, pics, getattr(a, "colour_limit", 10.0))
    if no_col:
        print(f"\n{name}: colours read without {len(no_col)} photo(s) (black-and-white or off lighting; still used "
              f"for face shape): " + ", ".join(sorted(os.path.basename(f) for f in no_col)))
    no_col = {os.path.basename(f) for f in no_col}
    for f in files:
        rgb = pics.pop(f)
        p, L = found[f]
        if f in no_smile:
            status.append(f"  {os.path.basename(f):40s} {'tilted - not used' if f in no_pitch else 'open mouth / broad smile - not used'}")
            continue
        if p is None:
            skipped.append(os.path.basename(f))
            status.append(f"  {os.path.basename(f):40s} NO FACE FOUND")
            continue
        t = turn(p)
        st = ("front" if abs(t) <= a.front_max else
              "turned (usable)" if 0.5 * gt <= abs(t) <= 1.7 * gt else
              "too turned - not used" if abs(t) > 1.7 * gt else "slightly turned - not used")
        status.append(f"  {os.path.basename(f):40s} turn {abs(t):.2f}  {st}")
        if getattr(a, "_audit", None) is not None:    # --check: per-photo quality and agreement numbers
            import cv2
            iod = float(np.linalg.norm(L[263, :2] - L[33, :2]))
            x0, x1 = int(max(0, L[:, 0].min())), int(L[:, 0].max())
            y0, y1 = int(max(0, L[:, 1].min())), int(L[:, 1].max())
            g = cv2.resize(cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY), (160, 160))
            audit.append({"i": len(status) - 1, "kind": st, "iod": iod,
                          "sharp": float(cv2.Laplacian(g, cv2.CV_64F).var()),
                          "open": float(np.linalg.norm(L[13, :2] - L[14, :2]) / max(iod, 1)),
                          "e": a._audit([a._audit.crop(rgb, p)])[0]})
        if abs(t) <= a.front_max:
            fronts.append((os.path.basename(f), align(rgb, p, cal["front"], size), t))
            if RS:
                crops = {s_[0]: [region_crop(s_[3], rgb, L)] for s_ in RS}
                if not getattr(a, "no_mirror", True):
                    pmm, Lmm = mesh(rgb[:, ::-1].copy(), full=True)
                    if pmm is not None:
                        for s_ in RS:
                            crops[s_[0]].append(region_crop(s_[3], rgb[:, ::-1].copy(), Lmm))
                rfront.append({"crops": crops, "eye": eye_open(L), "mouth": mouth_open(L)})
            if getattr(a, "_colour", None) is not None and os.path.basename(f) not in no_col:
                import colour
                cms.append(colour.measure(rgb, L, lit=True))
            if BR:
                bv = bvec(rgb, p, L, 0)
                if bv is not None:
                    bfront.append(bv)
                    bowner.append(len(fronts) - 1)
            if not getattr(a, "no_mirror", True):     # the same photo flipped left-right (points swapped to match)
                rm = rgb[:, ::-1].copy()
                pm = p.copy()
                pm[:, 0] = rgb.shape[1] - 1 - pm[:, 0]
                mirrors[len(fronts) - 1] = align(rm, pm[[3, 2, 1, 0, 4, 6, 5, 7, 8]], cal["front"], size)
                if BR:
                    if hasattr(BR[2], "photo_vec"):           # face points found again on the flipped photo
                        pm2, Lm = mesh(rm, full=True)
                        bv = bvec(rm, pm2, Lm, 0) if pm2 is not None else None
                    else:
                        bv = bvec(rm, pm[[3, 2, 1, 0, 4, 6, 5, 7, 8]], None, 0)
                    if bv is not None:
                        bfront.append(bv)
                        bowner.append(len(fronts) - 1)
        if 0.5 * gt <= abs(t) <= 1.7 * gt:
            if t < 0:                                         # turned the other way: mirror it
                rgb = rgb[:, ::-1].copy()
                p = p.copy()
                p[:, 0] = rgb.shape[1] - 1 - p[:, 0]
                p = p[[3, 2, 1, 0, 4, 6, 5, 7, 8]]           # left/right points swap when mirrored
            turneds.append((os.path.basename(f), align(rgb, p, cal["turned"], size), abs(t)))
            if RS:
                Lt = L
                if t < 0:                                     # face points found again on the flipped photo
                    _, Lt = mesh(rgb, full=True)
                if Lt is not None:
                    rturn[os.path.basename(f)] = {s_[0]: region_crop(s_[3], rgb, Lt, s_[4]) for s_ in RS}
            if BR:
                if t < 0 and hasattr(BR[2], "photo_vec"):  # face points found again on the flipped photo
                    p2, L2 = mesh(rgb, full=True)
                    bv = bvec(rgb, p2, L2, 1) if p2 is not None else None
                else:
                    bv = bvec(rgb, p, L, 1)
                if bv is not None:
                    bturn.append(bv)
    if no_smile:
        print(f"\n{name}: left out {len(no_smile)} front photo(s) with an open mouth or broad smile")
    print(f"\n{name}: {len(files)} photos -> {len(fronts)} front, {len(turneds)} turned "
          f"(game turned view = {gt:.2f}), {len(skipped)} no face found")
    if a.check:
        out = os.path.join(HERE, "results", "photo_check", name)
        os.makedirs(out, exist_ok=True)
        ok = "OK" if len(fronts) >= 4 and len(turneds) >= 1 else "NEEDS MORE"
        need = []
        if len(fronts) < 4: need.append(f"{4 - len(fronts)} more front")
        if len(turneds) < 1: need.append("1 more turned")
        flagged = []
        # agreement is judged among FRONT photos only: face recognition always rates turned faces lower, so turned
        # and in-between photos would be flagged for their angle, not for looking unlike the person
        audit_all = audit
        audit = [x for x in audit_all if x["kind"] == "front"]
        for x in audit_all:
            if x["kind"] != "front":
                notes = (["small face"] if x["iod"] < 45 else []) + (["blurry or smoothed"] if x["sharp"] < 40 else [])
                if notes:
                    status[x["i"]] += f"   <- {', '.join(notes)}"
                    if "not used" not in x["kind"]:
                        flagged.append(status[x["i"]].split()[0])
        if len(audit) >= 4:                           # agreement: how much each photo looks like the person's others
            E = np.array([x["e"] for x in audit])
            E = E / np.linalg.norm(E, axis=1, keepdims=True)
            agree = []
            for k in range(len(E)):
                o = np.delete(E, k, 0).mean(0)
                agree.append(float(E[k] @ (o / np.linalg.norm(o))))
            agree = np.array(agree)
            med = np.median(agree)
            mad = 1.4826 * np.median(np.abs(agree - med)) + 1e-6
            for x, ag in zip(audit, agree):
                z = (ag - med) / mad
                notes = []
                if z < -2.5:
                    notes.append("DISAGREES strongly - remove")
                elif z < -1.5:
                    notes.append("disagrees - consider removing")
                if x["iod"] < 45:
                    notes.append("small face")
                if x["sharp"] < 40:
                    notes.append("blurry or smoothed")
                if x["open"] > 0.12:
                    notes.append("open mouth")
                status[x["i"]] += f"   agree {ag:.2f} ({z:+.1f})" + (f"   <- {', '.join(notes)}" if notes else "")
                if notes and "not used" not in x["kind"]:
                    flagged.append(status[x["i"]].split()[0])
        lines = [f"{name}: {ok}" + (f" ({', '.join(need)})" if need else "")
                 + (f"; check these photos: {', '.join(flagged)}" if flagged else ""),
                 f"game turned view = {gt:.2f}; turned photos must be {0.5 * gt:.2f}-{1.7 * gt:.2f}, front under {a.front_max:.2f}", ""] + status
        open(os.path.join(out, "check.log"), "w").write("\n".join(lines))
        tiles = [x[1] for x in fronts] + [x[1] for x in turneds]
        if tiles:
            cols = 6
            sheet = Image.new("RGB", (size * cols, size * ((len(tiles) + cols - 1) // cols)))
            for i, t in enumerate(tiles):
                sheet.paste(Image.fromarray(t), ((i % cols) * size, (i // cols) * size))
            sheet.save(os.path.join(out, "aligned_check.jpg"), quality=88)
        print("  " + lines[0])
        return ok
    if not fronts:
        print("  No front-facing photo - skipped.")
        return
    if not turneds:
        print("  No photo turned like the game's turned view - using the most-turned front photo instead "
              "(weaker: the network expects a turned face there).")
        best = max(fronts, key=lambda x: abs(x[2]))
        turneds = [best]
    turneds = sorted(turneds, key=lambda x: abs(x[2] - gt))[:a.max_turned]

    out = os.path.join(HERE, "results", getattr(a, "out_dir", "photos"), name)
    os.makedirs(os.path.join(out, "single"), exist_ok=True)
    for f in glob.glob(os.path.join(out, "single", "*")):
        os.remove(f)
    # what the network sees
    tiles = [x[1] for x in fronts[:12]] + [x[1] for x in turneds[:6]]
    cols = 6
    sheet = Image.new("RGB", (size * cols, size * ((len(tiles) + cols - 1) // cols)))
    for i, t in enumerate(tiles):
        sheet.paste(Image.fromarray(t), ((i % cols) * size, (i // cols) * size))
    sheet.save(os.path.join(out, "aligned_check.jpg"), quality=88)

    pairs, owner, col_ok = [], [], []
    for fi, (fn, fimg, _) in enumerate(fronts):
        for tn, timg, _ in turneds:
            for img in [fimg] + ([mirrors[fi]] if fi in mirrors else []):   # mirrored photo counts for this one too
                pairs.append((img, timg))
                owner.append(fi)
                col_ok.append((fn not in no_col, tn not in no_col))
    V, C, P = run(net, ck, pairs, dev)
    owner = np.array(owner)
    if getattr(a, "_pitch_slopes", None) is not None:  # --pitch-fix: each photo's push from its own tilt taken off
        dp = np.array([pitches.get(fronts[o][0], game_pitch) - game_pitch for o in owner])
        V = np.clip(V - a.pitch_fix * dp[:, None] * a._pitch_slopes[None, :], 0, 1)
        print(f"  tilt correction: her front photos sit {np.mean([pitches.get(x[0], game_pitch) for x in fronts]) - game_pitch:+.3f} "
              "from the game camera on average")

    def combine(idx):
        v, c = V[idx].mean(0), C[idx].mean(0)
        cls = {j: int(P[j][idx].mean(0).argmax()) for j in P}
        return v, c, cls

    v, c, cls = combine(np.arange(len(pairs)))
    if no_col:                                        # colours only from the well-lit colour photos
        ok = np.array([f_ and t_ for f_, t_ in col_ok])
        if not ok.any():                              # every turned photo is off: judge by the front photo alone
            ok = np.array([f_ for f_, _ in col_ok])
        if ok.any():
            c = C[ok].mean(0)
    open(os.path.join(out, f"{name}.txt"), "w", encoding="utf-8").write(tg.write(template, v, cls, c))
    c_col = None
    if getattr(a, "_colour", None) is not None and cms:   # colours measured from the photos' pixels
        c_col = a._colour.predict(cms, c, a.colour_parts, a.colour_blend)
        if getattr(a._colour, "last", None):          # the palette route: what it measured and picked, as hex
            L_ = [f"{name}: colours from {len(cms)} photos (photo = measured from the photos, white-balanced on the eye "
                  "whites; game = how the picked palette colour looks in the Ruler Designer)"]
            for k_, d_ in a._colour.last.items():
                L_.append(f"  {k_:5s} photo {d_['photo']}  (kept {d_['kept']} photos)  -> palette ({d_['coord'][0]}, "
                          f"{d_['coord'][1]})  game {d_['game']}" + (f"   band: {d_['band']}" if d_.get("band") else ""))
            open(os.path.join(out, "colours.txt"), "w").write("\n".join(L_) + "\n")
            try:
                from PIL import Image as _I, ImageDraw as _D
                sw = _I.new("RGB", (360, 100 * len(a._colour.last)), "white")
                dr = _D.Draw(sw)
                for i_, (k_, d_) in enumerate(a._colour.last.items()):
                    dr.rectangle((0, 100 * i_, 179, 100 * i_ + 95), fill=d_["photo"])
                    dr.rectangle((180, 100 * i_, 359, 100 * i_ + 95), fill=d_["game"])
                    dr.text((4, 100 * i_ + 4), f"{k_} photo {d_['photo']}", fill="white")
                    dr.text((184, 100 * i_ + 4), f"game {d_['game']}", fill="white")
                sw.save(os.path.join(out, "colours.png"))
            except Exception:
                pass
    if BR and bfront:                                 # face-recognition bridge: alone, and mixed with the main reading
        pm = {j: P[j].mean(0) for j in P}             # the main network's button probabilities
        if hasattr(BR[3], "predict_person_full"):
            vb, pb = BR[3].predict_person_full(BR[0], BR[1], np.array(bfront), np.array(bturn), dev)
        else:
            vb, pb = BR[3].predict_person(BR[0], BR[1], np.array(bfront), np.array(bturn), dev), {}
        slide = np.array([tg.kind[g] != "choice" for g in tg.genes])
        vb_full = np.where(slide, vb, v)
        cls_b = {j: int(pb[j].argmax()) if j in pb else cls[j] for j in cls}
        open(os.path.join(out, f"{name}_bridge.txt"), "w", encoding="utf-8").write(tg.write(template, vb_full, cls_b, c))
        for f_ in glob.glob(os.path.join(out, f"{name}_mix*.*")) + glob.glob(os.path.join(out, f"{name}_s*_b*.*")):
            os.remove(f_)
        grid = a.button_weight is not None
        combos = [(w, bw) for w in a.bridge_weight for bw in (a.button_weight if grid else [w])]
        col_pair = min(combos, key=lambda q: abs(q[0] - 0.5) + abs(q[1] - 0.5))   # measured colours: on this one
        for w in a.bridge_weight:                     # w: bridge's share of the sliders
            for bw in (a.button_weight if grid else [w]):   # bw: its share of buttons and button strengths
                vm = np.where(slide, (1 - w) * v + w * vb, (1 - bw) * v + bw * vb)
                cls_m = {j: int(((1 - bw) * pm[j] + bw * pb[j]).argmax()) if j in pb else cls[j] for j in cls}
                tag = f"_s{round(w * 100)}_b{round(bw * 100)}" if grid else f"_mix{round(w * 100)}"
                open(os.path.join(out, f"{name}{tag}.txt"), "w", encoding="utf-8").write(tg.write(template, vm, cls_m, c))
                if c_col is not None and (w, bw) == col_pair:
                    open(os.path.join(out, f"{name}{tag}_col.txt"), "w", encoding="utf-8").write(
                        tg.write(template, vm, cls_m, c_col))
        for f_ in glob.glob(os.path.join(out, f"{name}_reg*.*")) + glob.glob(os.path.join(out, f"{name}_rc*.*")) + \
                glob.glob(os.path.join(out, f"{name}_final.*")) + glob.glob(os.path.join(out, f"{name}_fb50.*")):
            os.remove(f_)
        CT = getattr(a, "_centre", None)
        if CT is not None:                            # move the centre from the training average to the realistic one
            w = a.bridge_weight[0]
            vm = (1 - w) * v + w * vb
            cl_ = {j: int(((1 - w) * pm[j] + w * pb[j]).argmax()) if j in pb else cls[j] for j in cls}
            use = CT["use"] & slide
            sh, gs = CT["shift"], 1 - CT["r2"]
            vA = np.where(use, vm + gs * sh, vm)
            vB = np.where(use, vm + sh, vm)
            vS = np.where(use, CT["human"] + a.recentre_spread * (vA - CT["human"]), vm)
            outs_ = (("_rcA", vA), ("_rcB", vB), ("_rcS", vS)) if a.recentre_test else ()
            for tag, vv in outs_:
                open(os.path.join(out, f"{name}{tag}.txt"), "w", encoding="utf-8").write(
                    tg.write(template, np.clip(vv, 0, 1), cl_, c))
            def buttons(alpha):                       # buttons: the training faces' habits swapped for realistic ones
                cl_f = dict(cl_)
                if CT.get("bt") and alpha > 0:
                    for j, (tf_, hf_) in CT["bt"].items():
                        pr = (1 - w) * pm[j] + w * pb[j] if j in pb else pm[j]
                        if len(pr) == len(tf_):
                            cl_f[j] = int(np.argmax(pr * (hf_ / tf_) ** alpha))
                return cl_f
            cl_f = buttons(a.button_recentre)
            final_state = (np.clip(vS, 0, 1), cl_f)
            open(os.path.join(out, f"{name}_final.txt"), "w", encoding="utf-8").write(
                tg.write(template, np.clip(vS, 0, 1), cl_f, c))
            if a.recentre_test:
                open(os.path.join(out, f"{name}_fb50.txt"), "w", encoding="utf-8").write(
                    tg.write(template, np.clip(vS, 0, 1), buttons(0.5 * a.button_recentre), c))
            nchg = sum(cl_f[j] != cl_[j] for j in cl_)
            print(f"  final: realistic centre; {nchg} button(s) changed from the mix")
        if RS and rfront:                             # region specialists: their settings replace (most of) the mix's
            w = a.bridge_weight[0]
            vr = np.where(slide, (1 - w) * v + w * vb, (1 - w) * v + w * vb)
            probs = {j: (1 - w) * pm[j] + w * pb[j] if j in pb else pm[j] for j in cls}
            reg_info = {"vmix": vr.copy(), "pmix": {j: p_.copy() for j, p_ in probs.items()}, "reads": [], "c": c}
            cr = np.array(c, np.float64).copy()
            used = []
            gi = {g: j for j, g in enumerate(tg.genes)}
            tsel = [x[0] for x in turneds]
            for spec in RS:
                rn = spec[0]
                kind = spec[2].get("region", rn)          # eyes / nose / mouth (the file may have a longer name)
                keep = list(range(len(rfront)))
                if kind == "eyes" and len(keep) >= 3:  # leave out squinting / half-closed eyes
                    med = np.median([r_["eye"] for r_ in rfront])
                    k2 = [i for i in keep if rfront[i]["eye"] >= 0.75 * med]
                    keep = k2 if len(k2) >= 2 else keep
                if kind == "mouth" and len(keep) >= 3:  # leave out open mouths
                    k2 = [i for i in keep if rfront[i]["mouth"] <= 0.08]
                    keep = k2 if len(k2) >= 2 else keep
                fc = [cimg for i in keep for cimg in rfront[i]["crops"][rn]]
                tc = [rturn[t_][rn] for t_ in tsel if t_ in rturn]
                sv, sp, sc = region_read(spec, fc, tc, dev)
                if sc is not None and (rn == "eyes" or "eyecol" not in reg_info):
                    reg_info["eyecol"] = sc               # eye colour: the plain 'eyes' specialist when there is one
                if a.region_use and rn not in a.region_use:
                    used.append(f"{rn} (colour only)")
                    continue
                reg_info["reads"].append((spec[2]["genes"], sv, sp))
                rw, rbw = a.region_weight, a.region_button_weight
                for k, g in enumerate(spec[2]["genes"]):
                    j = gi.get(g)
                    if j is None or not region_gene_ok(g, a):
                        continue
                    vr[j] = (1 - rw) * vr[j] + rw * sv[k]
                    if j in probs and k in sp and len(sp[k]) == len(probs[j]):
                        probs[j] = (1 - rbw) * probs[j] + rbw * sp[k]
                if sc is not None and "eye_color" not in a.region_skip:
                    cr[4:6] = sc
                used.append(f"{rn} ({len(keep)}/{len(rfront)} front photos)")
            cls_r = {j: int(probs[j].argmax()) for j in cls}
            open(os.path.join(out, f"{name}_reg.txt"), "w", encoding="utf-8").write(tg.write(template, vr, cls_r, c))
            if any(s_[2].get("eye_colour", s_[2]["region"] == "eyes") for s_ in RS):
                open(os.path.join(out, f"{name}_reg_eyecol.txt"), "w", encoding="utf-8").write(
                    tg.write(template, vr, cls_r, cr))
            print("  region specialists: " + ", ".join(used) + f" -> {name}_reg.txt")
    # exaggerated versions: push every slider away from the average training face by a factor
    # (counters the pull toward an average face; buttons, strengths and colours stay as they are)
    for f in glob.glob(os.path.join(out, f"{name}_x*.txt")) + glob.glob(os.path.join(out, f"{name}_x*.jpg")):
        os.remove(f)
    slide = np.array([tg.kind[g] != "choice" for g in tg.genes])
    for k in a.stretch:
        vs = np.where(slide, np.clip(np.asarray(ck["mu"]) + k * (v - np.asarray(ck["mu"])), 0, 1), v)
        open(os.path.join(out, f"{name}_x{k:g}.txt"), "w", encoding="utf-8").write(tg.write(template, vs, cls, c))
    if (getattr(a, "variants", False) or getattr(a, "mix_stretch", None)) and BR and bfront and hasattr(BR[3], "predict_person_full"):
        write_variants(a, name, out, fronts, V, P, owner, np.array(bfront), np.array(bturn), np.array(bowner),
                       slide, cls, c, BR, tg, template, dev)
    per = []
    for fi, (fn, _, _) in enumerate(fronts):
        vi, ci, cli = combine(np.where(owner == fi)[0])
        per.append(vi)
        open(os.path.join(out, "single", os.path.splitext(fn)[0] + ".txt"), "w", encoding="utf-8").write(
            tg.write(template, vi, cli, ci))

    # consistency: spread across single photos vs how much faces vary in the training data
    lines = [f"{name}: {len(fronts)} front photos, {len(turneds)} turned photos, {len(pairs)} pairings.",
             "Spread across the single-photo predictions, as a share of how much training faces vary.",
             "Low (under ~40%) = the photos agree on this setting; near 100% = it's as if each photo were a random face.",
             ""]
    if len(per) >= 2:
        per = np.array(per)
        ratios = []
        for j, g in enumerate(tg.genes):
            if ck["sd"][j] * 255 < 2 or tg.kind[g] == "choice":
                continue
            r = per[:, j].std() / ck["sd"][j]
            ratios.append((r, g))
        ratios.sort()
        lines.insert(3, f"Median over {len(ratios)} sliders: {np.median([r for r, _ in ratios]) * 100:.0f}%\n")
        lines += [f"  {g:36s} {r * 100:4.0f}%" for r, g in ratios]
    else:
        lines.append("Only one front photo: no consistency check possible.")
    open(os.path.join(out, "consistency.log"), "w").write("\n".join(lines))
    print("  " + "\n  ".join(lines[:4]))
    print(f"  Saved in {out}")
    return {"name": name, "out": out, "v": v, "cls": cls, "c": c, "c_col": c_col, "reg": locals().get("reg_info"),
            "final": locals().get("final_state"), "bfront": np.array(bfront), "bturn": np.array(bturn)}


def main(argv=None, setup_only=False):
    """setup_only: parse argv, load everything and return (a, mesh, cal, net, ck, tg, dev, template) without running
    (used by colour_fix.py)"""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos", required=True, help="a person's photo folder, or a folder of person folders")
    ap.add_argument("--sex", choices=["female", "male"], default="female",
                    help="male: uses the men's networks and files (model_male.pt, bridge_male.pt, colour_model_male.npz, "
                         "template_male.txt, realistic_centre_male.npz, hair_gallery_male.npz); clean-shaven for now")
    ap.add_argument("--model", default=os.path.join("results", "model_29996_v3_base_photo.pt"))
    ap.add_argument("--data", default=os.path.join("random_faces", "noise_female"),
                    help="game screenshots, for the one-off framing calibration")
    ap.add_argument("--template", default="template_hair17.txt")
    ap.add_argument("--front-max", type=float, default=0.20, help="max turn for a 'front' photo (eye-widths)")
    ap.add_argument("--max-turned", type=int, default=6)
    ap.add_argument("--recalibrate", action="store_true")
    ap.add_argument("--pitch-keep", type=float, default=0.0,
                    help="keep only this share of front photos, those tilted (up/down) nearest the game camera (e.g. 0.6)")
    ap.add_argument("--pitch-fix", type=float, default=0.0,
                    help="take each photo's tilt push off the main network's reading (1 = full; needs "
                         "results\\pitch_slopes[_male].npz from nuisance.py)")
    ap.add_argument("--coral", default="", help="coral.py file: re-shape photo features onto the screenshots' pattern")
    ap.add_argument("--coral-strength", type=float, default=1.0, help="0 = off, 1 = full re-shaping")
    ap.add_argument("--fda", default="", help="fda.py file: swap photos' broad light/colour patterns for the screenshots'")
    ap.add_argument("--fda-beta", type=float, default=0.03, help="how much of the low end to swap (0.01-0.09)")
    ap.add_argument("--fda-strength", type=float, default=1.0)
    ap.add_argument("--coral-collect", default="",
                    help="save the main network's photo features to this .npz (for coral.py fit)")
    ap.add_argument("--centre-stretch", type=float, nargs="*", default=[],
                    help="with 12+ people: write shift-removed DNA at these strengths")
    ap.add_argument("--out", default="photos", help="folder inside results to write to")
    ap.add_argument("--bridge", default=os.path.join("results", "bridge.pt"),
                    help="face-recognition bridge (bridge.py train) -> _bridge and _mix files; 'none' to skip")
    ap.add_argument("--colour", default=os.path.join("results", "colour_model.npz"),
                    help="colour model (colour.py train) -> <person>_mix50_col.txt; 'none' to skip")
    ap.add_argument("--colour-parts", nargs="+", default=["skin", "eye", "hair"], choices=["skin", "eye", "hair"],
                    help="which colours to measure from the photos (the rest come from the network)")
    ap.add_argument("--colour-blend", type=float, default=None,
                    help="measured colour's share, averaged with the network's colour (1 = measured only)")
    ap.add_argument("--button-weight", type=float, nargs="+", default=None,
                    help="bridge's share of the buttons, separately from the sliders (files <person>_s50_b75 etc.)")
    ap.add_argument("--bridge-weight", type=float, nargs="+", default=[0.5],
                    help="bridge's share in the mix: one file per value, <person>_mix50.txt for 0.5 etc.")
    ap.add_argument("--no-mirror", action="store_true", help="don't also read each front photo flipped left-right")
    ap.add_argument("--correction", action="store_true", help="also write the old _fixed version")
    ap.add_argument("--check", action="store_true",
                    help="only check the photos (faces found, front/turned counts); no DNA written")
    ap.add_argument("--regions", nargs="*", default=[],
                    help="region specialists to use, e.g. --regions eyes nose mouth (results\\region_<name>.pt; "
                         "-> <person>_reg.txt). Off by default: they didn't help on photos.")
    ap.add_argument("--region-weight", type=float, default=0.8,
                    help="specialist's share of its region's sliders in _reg (the rest from the mix)")
    ap.add_argument("--region-button-weight", type=float, default=0.75,
                    help="specialist's share of its region's buttons in _reg")
    ap.add_argument("--region-skip", nargs="*", default=["brow"],
                    help="settings whose names contain any of these keep the mix's reading (default: brows)")
    ap.add_argument("--lashes", choices=["dark", "read"], default=None,
                    help="dark: dark eyelashes (like mascara; the default for women); read: as the networks read them "
                         "(the default for men)")
    ap.add_argument("--colour-fix", action="store_true", help="(kept for old commands; now on whenever "
                    "results\\colour_fix.json / colour_fix_male.json exists)")
    ap.add_argument("--no-colour-fix", action="store_true",
                    help="don't correct the final colours with colour_fix.py's correction")
    ap.add_argument("--region-final", action="store_true",
                    help="put the region specialists' settings (shared shift removed, as in _regc) into _final.txt, "
                         "so a blind comparison of _final shows them; needs 8+ people in the run")
    ap.add_argument("--region-use", nargs="*", default=[],
                    help="region specialists whose SETTINGS are used (default: all loaded); others only give eye colour")
    ap.add_argument("--region-only", nargs="*", default=[],
                    help="region specialists: use only settings whose names contain one of these (e.g. brow)")
    ap.add_argument("--hair", type=int, default=3,
                    help="pick her hairstyle from the photos: the best goes into _final.txt, the next ones into "
                         "_hair2.txt ... (this many in all; 0 = keep the template's hairstyle). Needs hair.py gallery")
    ap.add_argument("--beard", type=int, default=3,
                    help="men: pick his beard from the photos: the best goes into _final.txt, the next ones into "
                         "_beard2.txt ... (this many in all; 0 = clean-shaven). Needs beard.py shots + gallery")
    ap.add_argument("--colour-limit", type=float, default=10.0,
                    help="how far a photo's skin may be from her usual before it's left out of the colour reading "
                         "(lower = stricter; default 10)")
    ap.add_argument("--colour-all", action="store_true",
                    help="read colours from every photo (by default black-and-white photos, and photos whose skin "
                         "looks much darker / more orange than her others, are used for face shape only)")
    ap.add_argument("--keep-smiles", action="store_true",
                    help="use open-mouth and broadly smiling front photos too (by default they're left out while "
                         "enough others remain)")
    ap.add_argument("--no-recentre", dest="recentre", action="store_false",
                    help="don't write <person>_final.txt (the mix recentred on the realistic average, differences "
                         "widened 25%%; needs results\\realistic_centre.npz from check_center.py)")
    ap.add_argument("--recentre-test", action="store_true", help="also write the _rcA/_rcB/_rcS test versions")
    ap.add_argument("--corrected", action="store_true",
                    help="men: also apply the reference correction + proportion fit to _final (off by default since "
                         "2026-11-07; they were tuned on the old model_male.pt)")
    ap.add_argument("--raw-final", action="store_true",
                    help="men: leave _final.txt uncorrected (the reference correction / proportion fit only as _ref)")
    ap.add_argument("--button-recentre", type=float, default=None,
                    help="_final: how strongly buttons are moved from the training faces' habits to the hand-made "
                         "people's (0 = not at all, 1 = fully; default 0 for women, 1 for men)")
    ap.add_argument("--recentre-spread", type=float, default=1.25, help="_rcS: widening around the realistic centre")
    ap.add_argument("--recentre-skip", nargs="*", default=["mouth_open", "neck"],
                    help="settings left out of the recentring")
    ap.add_argument("--mix-stretch", type=float, nargs="+", default=None,
                    help="write the mix50 face exaggerated by these factors: <person>_x150 for 1.5 etc.")
    ap.add_argument("--variants", action="store_true",
                    help="also write the test versions: _x125 _x150 _d50 _d100 _med _n5 _n10 _n15 (see write_variants)")
    ap.add_argument("--stretch", type=float, nargs="*", default=[],
                    help="also write exaggerated versions of each person's DNA (sliders pushed away from the average face)")
    a = ap.parse_args(argv)
    a.out_dir = a.out
    a.sfx = "_male" if a.sex == "male" else ""
    if a.sex == "male":                               # the men's files, unless one was given by hand
        for k, fem, mal in (("model", os.path.join("results", "model_29996_v3_base_photo.pt"),
                             os.path.join("results", "model_male_pilot_lr20.pt")),   # 2026-11-07: preferred by eye
                            ("data", os.path.join("random_faces", "noise_female"), os.path.join("random_faces", "noise_male")),
                            ("template", "template_hair17.txt", "template_male.txt"),
                            ("bridge", os.path.join("results", "bridge.pt"), os.path.join("results", "bridge_male.pt")),
                            ("colour", os.path.join("results", "colour_model.npz"),
                             os.path.join("results", "colour_model_male.npz"))):
            if getattr(a, k) == fem:
                setattr(a, k, mal)
    if a.lashes is None:
        a.lashes = "read" if a.sex == "male" else "dark"
    # men: the reference correction + narrow-only proportion fit were tuned on the old model_male.pt and changed
    # little by eye; with the pilot model they're off unless --corrected is given
    if a.sex == "male" and not a.corrected:
        a.raw_final = True
    if a.button_recentre is None:
        a.button_recentre = 1.0 if a.sex == "male" else 0.0

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mesh = Mesh()
    cp = os.path.join(HERE, "results", f"photo_calib{a.sfx}.json")
    if a.recalibrate or not os.path.exists(cp):
        print("Measuring where faces sit in the game crops (once) ...")
        cal = calibrate(os.path.join(HERE, a.data) if not os.path.isabs(a.data) else a.data, mesh)
        os.makedirs(os.path.dirname(cp), exist_ok=True)
        json.dump(cal, open(cp, "w"), indent=1)
    cal = json.load(open(cp))
    if a.check:
        a._audit = None
        try:                                          # per-photo agreement uses the two face-recognition models
            import bridge2
            a._audit = bridge2.Feats(("arc", "glint"))
        except SystemExit:
            print("(onnx / onnx2torch not installed: skipping the per-photo agreement check)")
        net, tg, template = None, None, None
        ck = {"size": 224}
        if os.path.exists(os.path.join(HERE, a.model) if not os.path.isabs(a.model) else a.model):
            ck = {"size": int(torch.load(os.path.join(HERE, a.model) if not os.path.isabs(a.model) else a.model,
                                         map_location="cpu", weights_only=False).get("size", 224))}
        subs = sorted(d for d in glob.glob(os.path.join(a.photos, "*")) if os.path.isdir(d))
        res = [(os.path.basename(os.path.normpath(f)), person(f, os.path.basename(os.path.normpath(f)).replace(" ", "_"),
                                                            a, mesh, cal, net, ck, tg, dev, template))
               for f in (subs or [a.photos])]
        print(f"\n{sum(r == 'OK' for _, r in res)} of {len(res)} people OK. Details per photo: results\\photo_check\\<person>\\check.log")
        return
    net, ck, tg = load_model(os.path.join(HERE, a.model) if not os.path.isabs(a.model) else a.model, dev)
    if a.coral:
        zc = np.load(os.path.join(HERE, a.coral) if not os.path.isabs(a.coral) else a.coral)
        if int(zc["dim"]) != net.trunk[0].in_features or str(zc["model"]) != os.path.basename(a.model):
            sys.exit(f"{a.coral} was made for another model ({zc['model']}).")
        net.coral = tuple(torch.from_numpy(zc[k].astype(np.float32)).to(dev) for k in ("mp", "W", "ms")) + \
            (float(a.coral_strength),)
        print(f"Photo features re-shaped onto the screenshots' pattern ({a.coral}, strength {a.coral_strength:g})")
    if a.coral_collect:
        net.collect = []
    a._pitch_slopes = None
    if a.pitch_fix:
        zp = os.path.join(HERE, "results", f"pitch_slopes{a.sfx}.npz")
        if not os.path.exists(zp):
            sys.exit(f"{zp} missing: run nuisance.py first.")
        z = np.load(zp, allow_pickle=True)
        sl = dict(zip([str(g) for g in z["genes"]], z["slope"]))
        a._pitch_slopes = np.array([sl.get(g, 0.0) * (0.5 if tg.kind[g] == "signed" else 1.0)
                                    if tg.kind[g] != "choice" else 0.0 for g in tg.genes])
        print(f"Tilt correction on {int((a._pitch_slopes != 0).sum())} sliders (strength {a.pitch_fix:g})")
    if a.fda:
        zf = np.load(os.path.join(HERE, a.fda) if not os.path.isabs(a.fda) else a.fda)
        size = int(ck.get("size", 224))
        if zf["amp"].shape[-1] != size:
            sys.exit(f"{a.fda} was made for size {zf['amp'].shape[-1]}, the model uses {size}.")
        b = max(1, int(round(a.fda_beta * size)))
        net.fda = (torch.from_numpy(zf["amp"].astype(np.float32)).to(dev), b, float(a.fda_strength))
        print(f"Photos' broad light and colour patterns swapped for the screenshots' ({a.fda}, "
              f"{2 * b + 1}x{2 * b + 1} lowest frequencies, strength {a.fda_strength:g})")
    _write = tg.write
    tg.write = lambda text, v, cls, col: finish_dna(_write(text, v, cls, col), a.lashes, a.sex)
    bpath = os.path.join(HERE, a.bridge) if a.bridge and not os.path.isabs(a.bridge) else a.bridge
    if a.bridge.lower() == "none":
        a.bridge = ""
    elif not os.path.exists(bpath):
        if a.bridge in (os.path.join("results", "bridge.pt"), os.path.join("results", "bridge_male.pt")):
            print(f"(No {a.bridge} yet - writing the main network's version only.)")
            a.bridge = ""
        else:
            sys.exit(f"Bridge file not found: {bpath}  - did 'bridge2.py train' finish? Its output says.")
    if a.bridge:
        import torch as _t
        bp = os.path.join(HERE, a.bridge) if not os.path.isabs(a.bridge) else a.bridge
        if "feats" in _t.load(bp, map_location="cpu", weights_only=False):     # second-round bridge (bridge2.py)
            import bridge2 as bmod
            bnet, bck = bmod.load(bp, dev)
            fx = bmod.Feats(bck["feats"])
        else:
            import bridge as bmod
            bnet, bck = bmod.load(bp, dev)
            fx = bmod.ArcFace()
        if list(bck["genes"]) != list(tg.genes):
            sys.exit("The bridge was trained for a different settings list than this --model.")
        if "feats" in bck:
            fx.ck = bck
        a._bridge = (bnet, bck, fx, bmod)
    a._colour = None
    a._regions = load_regions(a.regions, dev) if a.bridge else []
    a._centre = None
    if a.recentre:
        cp_ = os.path.join(HERE, "results", f"realistic_centre{a.sfx}.npz")
        z = np.load(cp_, allow_pickle=True) if os.path.exists(cp_) else None
    if a.recentre and z is None:
        print(f"(No results\\realistic_centre{a.sfx}.npz - run  python check_center.py --sex {a.sex}  once to get "
              "<person>_final.txt.)")
    elif a.recentre:
        cg = {str(g): k for k, g in enumerate(z["genes"])}
        G_ = len(tg.genes)
        human, train, r2, use = np.zeros(G_), np.zeros(G_), np.ones(G_), np.zeros(G_, bool)
        for j, g in enumerate(tg.genes):
            k = cg.get(g)
            if k is None or not z["slide"][k] or z["n"][k] < 5 or not np.isfinite(z["human"][k]) \
                    or any(x in g for x in a.recentre_skip):
                continue
            human[j], train[j], r2[j], use[j] = z["human"][k], z["train"][k], z["r2"][k], True
        bt = {}
        if "choice" in z.files:                       # realistic button habits (check_center.py, newer version)
            zg = [str(g) for g in z["genes"]]
            for k_, tf_, hf_ in zip(z["choice"], z["train_freq"], z["human_freq"]):
                g = zg[int(k_)]
                if g in tg.genes and not any(x in g for x in a.recentre_skip):
                    bt[tg.genes.index(g)] = (np.asarray(tf_, float), np.asarray(hf_, float))
        else:
            print("(realistic_centre.npz is from the older check_center.py - rerun it to recentre buttons too.)")
        a._centre = {"human": human, "shift": human - train, "r2": r2, "use": use, "bt": bt}
        print(f"Recentring {use.sum()} sliders on the realistic average -> <person>_final.txt"
              + (" (+ _rcA, _rcB, _rcS)" if a.recentre_test else ""))
    if a._regions:
        print("Region specialists: " + ", ".join(s_[0] for s_ in a._regions))
    cpath = os.path.join(HERE, a.colour) if a.colour and not os.path.isabs(a.colour) else a.colour
    if a.colour.lower() != "none" and os.path.exists(cpath):
        import colour
        a._colour = colour.Model(cpath)
    calibrated = bool(getattr(a._colour, "cal", None)) or bool(getattr(a._colour, "look", None))
    if getattr(a._colour, "look", None):
        print("Colours: from the game's palettes (palette\\*.png) - per person, odd photos dropped; see colours.txt"
              + (" (palette looks from colour_session.npz)" if getattr(a._colour, "session", False) else ""))
    elif getattr(a, "_colour", None) is not None:
        print("Colours: palette\\skin_palette.png etc. not found - using the older colour lookup")
    if a.colour_blend is None:
        a.colour_blend = 1.0 if calibrated else 0.5
    template = open(os.path.join(HERE, a.template), encoding="utf-8").read()
    a._colfix = None
    cfp = os.path.join(HERE, "results", f"colour_fix{a.sfx}.json")
    # final colours: measured from the photos on the game's scale (colour_calib.py) when that exists; the older
    # reference correction (colour_fix.py) only when asked for with --colour-fix, or when there's no calibration
    a._measured_final = calibrated and not a.colour_fix and not a.no_colour_fix
    if not a.no_colour_fix and not a._measured_final and os.path.exists(cfp):
        a._colfix = json.load(open(cfp))
        print("Colour correction from your reference people -> <person>_fcol.txt")
    if setup_only:
        return a, mesh, cal, net, ck, tg, dev, template

    subs = sorted(d for d in glob.glob(os.path.join(a.photos, "*")) if os.path.isdir(d))
    folders = subs if subs else [a.photos]
    done = []
    for f in folders:
        r = person(f, os.path.basename(os.path.normpath(f)).replace(" ", "_"), a, mesh, cal, net, ck, tg, dev, template)
        if r:
            r["folder"] = f
            done.append(r)
    if a.coral_collect and net.collect:
        cp_ = os.path.join(HERE, a.coral_collect) if not os.path.isabs(a.coral_collect) else a.coral_collect
        np.savez(cp_, feats=np.concatenate(net.collect).astype(np.float32), model=os.path.basename(a.model))
        print(f"\nPhoto features saved: {cp_} ({sum(len(x) for x in net.collect)} photo pairs)")
        net.collect = None
    # Remove the shift every photo shares ("this is a photo, not a screenshot"): measure each person's sliders
    # from the average of the OTHER people instead of from the game's average face, then add the game's average back.
    if len(done) >= 12 and a.centre_stretch:
        slide = np.array([tg.kind[g] != "choice" for g in tg.genes])
        mu = np.asarray(ck["mu"])
        allv = np.array([r["v"] for r in done])
        for i, r in enumerate(done):
            others = np.delete(allv, i, 0).mean(0)
            for k in a.centre_stretch:
                vc = np.where(slide, np.clip(mu + k * (r["v"] - others), 0, 1), r["v"])
                tag = "_c" if k == 1 else f"_c_x{k:g}"
                open(os.path.join(r["out"], f"{r['name']}{tag}.txt"), "w", encoding="utf-8").write(
                    tg.write(template, vc, r["cls"], r["c"]))
        print(f"\nShared photo shift removed using {len(done)} people: <person>_c.txt "
              f"(and {', '.join(f'_c_x{k:g}' for k in a.centre_stretch if k != 1)}) in each person's folder.")
    elif len(done) > 1 and a.centre_stretch:
        print(f"\nOnly {len(done)} people: need 12+ to remove the shared photo shift, so no _c files written.")
    # region specialists, with the shift they give EVERY photo removed: each setting's average difference from the
    # mix over all the people (a photo-vs-game bias, not anything about one person) is taken off, so the specialist
    # only decides how each person differs from the others. Saved, so a single person later gets the same correction.
    regs = [r for r in done if r.get("reg")]
    if regs:
        offp = os.path.join(HERE, "results", "region_offset.npz")
        gi = {g: j for j, g in enumerate(tg.genes)}
        if len(regs) >= 8:
            dv, dp = {}, {}
            for r in regs:
                for genes_, sv, sp in r["reg"]["reads"]:
                    for k, g in enumerate(genes_):
                        j = gi.get(g)
                        if j is None:
                            continue
                        dv.setdefault(g, []).append(sv[k] - r["reg"]["vmix"][j])
                        if k in sp and j in r["reg"]["pmix"] and len(sp[k]) == len(r["reg"]["pmix"][j]):
                            dp.setdefault(g, []).append(sp[k] - r["reg"]["pmix"][j])
            off_v = {g: float(np.mean(x)) for g, x in dv.items()}
            off_p = {g: np.mean(x, 0) for g, x in dp.items()}
            np.savez(offp, gv=np.array(list(off_v)), v=np.array(list(off_v.values())),
                     gp=np.array(list(off_p), dtype=object), p=np.array(list(off_p.values()), dtype=object),
                     people=len(regs))
            print(f"\nRegion specialists' shared shift measured on {len(regs)} people (saved results\\region_offset.npz).")
        elif os.path.exists(offp):
            z = np.load(offp, allow_pickle=True)
            off_v = dict(zip([str(g) for g in z["gv"]], z["v"]))
            off_p = dict(zip([str(g) for g in z["gp"]], z["p"]))
            print(f"\nRegion specialists' shared shift from results\\region_offset.npz ({int(z['people'])} people).")
        else:
            off_v = off_p = None
            print("\n(Need 8+ people once to measure the region specialists' shared shift - no _regc files.)")
        if off_v is not None:
            rw, rbw = a.region_weight, a.region_button_weight
            for r in regs:
                vr = r["reg"]["vmix"].copy()
                probs = {j: p_.copy() for j, p_ in r["reg"]["pmix"].items()}
                for genes_, sv, sp in r["reg"]["reads"]:
                    for k, g in enumerate(genes_):
                        j = gi.get(g)
                        if j is None or not region_gene_ok(g, a):
                            continue
                        vr[j] = (1 - rw) * vr[j] + rw * np.clip(sv[k] - off_v.get(g, 0.0), 0, 1)
                        if j in probs and k in sp and g in off_p and len(sp[k]) == len(probs[j]):
                            q = np.clip(sp[k] - off_p[g], 0, None)
                            q = q / max(q.sum(), 1e-6)
                            probs[j] = (1 - rbw) * probs[j] + rbw * q
                cls_r = {j: int(probs[j].argmax()) for j in r["cls"]}
                regc_text = tg.write(template, vr, cls_r, r["reg"]["c"])
                open(os.path.join(r["out"], f"{r['name']}_regc.txt"), "w", encoding="utf-8").write(regc_text)
                fin = os.path.join(r["out"], f"{r['name']}_final.txt")
                if a.region_final and os.path.exists(fin):
                    # the specialists' settings (shared shift removed) replace those settings in _final; everything
                    # else in _final (recentring, colours, hair, beard added later) stays as it is
                    import re as _re
                    text = open(fin, encoding="utf-8").read()
                    nrep = 0
                    for genes_, _sv, _sp in r["reg"]["reads"]:
                        for g in genes_:
                            if g not in gi or not region_gene_ok(g, a):
                                continue
                            m = _re.search(r"\b" + _re.escape(g) + r"=\{[^}]*\}", regc_text)
                            if m and _re.search(r"\b" + _re.escape(g) + r"=\{[^}]*\}", text):
                                text = _re.sub(r"\b" + _re.escape(g) + r"=\{[^}]*\}", lambda _m: m.group(0), text)
                                nrep += 1
                    open(fin, "w", encoding="utf-8").write(text)
                    r["reg_final_n"] = nrep
            print("  <person>_regc.txt: the specialists without their shared shift.")
            if a.region_final:
                ns = sorted({r.get("reg_final_n", 0) for r in regs})
                print(f"  --region-final: the specialists' settings are now in _final.txt ({'/'.join(map(str, ns))} settings per person).")
        elif a.region_final:
            print("  --region-final: no shared shift yet (need 8+ people in one run), so _final.txt is unchanged.")
    # final colours: measured (colour_calib.py) or corrected from the reference people (colour_fix.py)
    if a._colfix or a._measured_final:
        import re as _re
        n_ = 0
        for r in done:
            fin = os.path.join(r["out"], f"{r['name']}_final.txt")
            if not os.path.exists(fin):
                continue
            if a._colfix:
                cc = apply_colfix(a._colfix, r["c"], (r.get("reg") or {}).get("eyecol"), r.get("c_col"))
            elif r.get("c_col") is not None:
                cc = np.clip(r["c_col"], 8 / 255, 247 / 255)
            else:
                continue
            q = [int(round(float(np.clip(x, 0, 1)) * 255)) for x in cc]
            text = open(fin, encoding="utf-8").read()
            for k, key in enumerate(("hair_color", "skin_color", "eye_color")):
                x, y = q[2 * k], q[2 * k + 1]
                text = _re.sub(r"\b" + key + r"=\{[^}]*\}", f"{key}={{ {x} {y} {x} {y} }}", text)
            open(fin, "w", encoding="utf-8").write(text)
            n_ += 1
        print(f"\nColours in _final.txt for {n_} people: " + (f"corrected from the reference people (colour_fix{a.sfx}.json)"
              if a._colfix else ("matched to the game's palettes (colours.txt / colours.png in each folder)"
                                 if getattr(a._colour, "look", None) else
                                 "measured from the photos on the game's colour scale (colour_calib.json)")))
    # correction learned from the hand-made reference characters (calibrate_refs.py)
    corr = load_correction(tg)
    if corr is not None and a.correction:
        for r in done:
            vf, cf = apply_correction(corr, r["v"], r["c"])
            open(os.path.join(r["out"], f"{r['name']}_fixed.txt"), "w", encoding="utf-8").write(
                tg.write(template, vf, r["cls"], cf))
        print(f"\nReference correction applied ({corr['method']}): <person>_fixed.txt in each person's folder.")
    # hairstyle from her photos (hair.py's gallery): best one into _final.txt, the next ones as _hair2, _hair3
    hair_lines = []
    if a.hair and done:
        try:
            import hair as H
        except ImportError:
            H = None
        if H is None or not os.path.exists(H.gallery_path(a.sex)):
            print(f"\n(No hairstyle picking: run  python hair.py gallery --sex {a.sex}  once to set it up.)")
        else:
            g = H.load_gallery(a.sex)
            clip = H.Clip(dev, a.sex)
            for r in done:
                fin = os.path.join(r["out"], f"{r['name']}_final.txt")
                if not os.path.exists(fin):
                    continue
                crops = H.person_crops(r["folder"], mesh)
                if not crops:
                    continue
                e, at = H.read(clip, crops)
                top = np.argsort(-H.scores(g, e, at)[2])[:max(1, a.hair)]
                text = open(fin, encoding="utf-8").read()
                for k_, j in enumerate(top):
                    dst = fin if k_ == 0 else os.path.join(r["out"], f"{r['name']}_hair{k_ + 1}.txt")
                    open(dst, "w", encoding="utf-8").write(H.HAIR_RE.sub(f"custom_hair={g['styles'][j]}", text))
                hair_lines.append(f"{r['name']}: " + ", ".join(f"{k_ + 1}. {g['styles'][j]}" for k_, j in enumerate(top)))
            if hair_lines:
                open(os.path.join(HERE, "results", a.out, "hair_picks.txt"), "w").write("\n".join(hair_lines) + "\n")
                print("\nHairstyles (1st is in _final.txt; the others are _hair2.txt, _hair3.txt):\n  "
                      + "\n  ".join(hair_lines))
    # reference correction on the final face (ref_offset.py): <person>_ref.txt next to _final.txt
    try:
        import ref_offset as RO
        op = RO.offset_path(a.sex)
    except ImportError:
        op = None
    if op and os.path.exists(op) and done:
        off = json.load(open(op))["offset"]
        n_ = 0
        for r in done:
            fin = os.path.join(r["out"], f"{r['name']}_final.txt")
            if os.path.exists(fin):
                open(os.path.join(r["out"], f"{r['name']}_ref.txt"), "w", encoding="utf-8").write(
                    RO.apply(open(fin, encoding="utf-8").read(), tg, off))
                n_ += 1
        print(f"\nReference correction ({len(off)} sliders): <person>_ref.txt for {n_} people.")
    # men: the reference correction and the narrow-only proportion fit (geo_fit.py) are part of _final.txt;
    # the face without them is kept as _final_raw.txt
    if a.sex == "male" and op and os.path.exists(op) and done and not a.raw_final:
        import geo_fit as GF
        Z = dict(np.load(GF.fit_path(a.sex))) if os.path.exists(GF.fit_path(a.sex)) else None
        if Z is None:
            print("(No results\\geo_fit_male.npz - run geo_fit.py --narrow-only once for the proportion fit.)")
        n_ = 0
        for r in done:
            fin = os.path.join(r["out"], f"{r['name']}_final.txt")
            if not os.path.exists(fin):
                continue
            photo = GF.photo_measures(r["folder"], mesh, sys.modules[__name__]) if Z is not None else None
            for tag in ("_final", "_hair2", "_hair3"):
                f_ = os.path.join(r["out"], f"{r['name']}{tag}.txt")
                if not os.path.exists(f_):
                    continue
                t_ = open(f_, encoding="utf-8").read()
                if tag == "_final":
                    open(os.path.join(r["out"], f"{r['name']}_final_raw.txt"), "w", encoding="utf-8").write(t_)
                t_ = RO.apply(t_, tg, off)
                if photo is not None:
                    t_ = GF.narrow_apply(t_, photo, tg, Z)
                open(f_, "w", encoding="utf-8").write(t_)
            n_ += 1
        print(f"Men: reference correction" + (" + proportion fit" if Z is not None else "") +
              f" now in _final.txt (and _hair2/_hair3) for {n_} people; uncorrected: _final_raw.txt")
    # men: beard from his photos (beard.py's gallery): best one into _final (and _final_raw, _hair2, _hair3),
    # the next ones as _beard2, _beard3 (his _final face with that beard)
    beard_lines = []
    if a.sex == "male" and a.beard and done:
        try:
            import beard as BD
            gb = BD.load_gallery()
        except ImportError:
            gb = None
        if gb is None:
            print("\n(No beard picking: run  python beard.py shots / gallery  once to set it up - see beard.py.)")
        else:
            import collect_noise as CN
            bclip = BD.Clip(dev)
            reads = []
            for r in done:
                crops = BD.person_crops(r["folder"], mesh)
                if crops and os.path.exists(os.path.join(r["out"], f"{r['name']}_final.txt")):
                    reads.append((r, *BD.read_person(bclip, crops)))
            gb = BD.set_photo_mean(gb, [(e, at, kept) for _, e, at, kept in reads])
            put = lambda t, b: CN.shave(t) if b == BD.NONE else CN.set_beard(t, b)     # noqa: E731
            for r, e, at, kept in reads:
                picks = BD.choose(gb, e, at, kept, max(1, a.beard))
                base = os.path.join(r["out"], r["name"])
                fin_text = open(base + "_final.txt", encoding="utf-8").read()
                for k_, b in enumerate(picks[1:], 2):
                    open(f"{base}_beard{k_}.txt", "w", encoding="utf-8").write(put(fin_text, b))
                for tag in ("_final", "_final_raw", "_hair2", "_hair3"):
                    if os.path.exists(base + tag + ".txt"):
                        t_ = open(base + tag + ".txt", encoding="utf-8").read()
                        open(base + tag + ".txt", "w", encoding="utf-8").write(put(t_, picks[0]))
                beard_lines.append(f"{r['name']} (photos {kept}): " +
                                   ", ".join(f"{k_ + 1}. {b}" for k_, b in enumerate(picks)))
            if beard_lines:
                open(os.path.join(HERE, "results", a.out, "beard_picks.txt"), "w").write("\n".join(beard_lines) + "\n")
                print("\nBeards (1st is in _final.txt; the others are _beard2.txt, _beard3.txt):\n  "
                      + "\n  ".join(beard_lines))
    print("\nYour DNA:  results\\" + a.out + "\\<person>\\<person>_final.txt" +
          ("   (other hairstyles: _hair2.txt, _hair3.txt)" if hair_lines else ""))
    print(f"Screenshot:  python autoshot.py folder results\\{a.out}\\<person>   (and ...\\<person>\\single)")


if __name__ == "__main__":
    main()
