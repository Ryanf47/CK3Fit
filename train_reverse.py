"""Reverse network: read the DNA back off a game screenshot. Version 2.

Changes from version 1 (which only learned colours and an average face):
  - targets now follow the game's own gene files:
      * plain sliders (head/jaw/chin/eyes/mouth/forehead/neck, 30 of them): the value IS the slider, 0-255;
      * paired blend-shape sliders (gene_bs_... with neg/pos or min/max buttons, plus vanilla): one signed slider,
        neg side below the middle, pos side above, vanilla = middle;
      * real button choices (head shape, nose profile, eye shape, complexion, ...): which button + its strength;
  - the image network (DINOv2-small) is now trained on these faces too (fine-tuned), instead of frozen, with mild
    shifts/brightness changes so it can't just memorise. This is where shape reading has to come from.

Runs on Windows next to the data, on the RTX card (torch + transformers from probe_tools.py).

    python train_reverse.py check   --data random_faces\\noise_female   (crop check picture)
    python train_reverse.py prepare --data random_faces\\noise_female   (once: cuts and caches the crops, ~1 GB RAM file)
    python train_reverse.py train   --data random_faces\\noise_female   (~20-40 min; prints progress each epoch)

Then screenshot the 24 held-out pairs it writes:
    python autoshot.py folder results\\pred_N      (N = the name it prints)
Outputs go to CK3Fit\\results, named by face count and settings: report_N.txt, model_N.pt (kept for the photo
test later), pred_N\\NNNNN_true.txt / NNNNN_pred.txt.
"""
import argparse
import csv
import glob
import os
import re
import sys
import time

import numpy as np

GENE_RE = re.compile(r'\b(\w+)=\{\s*"([^"]+)"\s+(\d+)\s+"([^"]+)"\s+(\d+)\s*\}')
COLOUR_RE = re.compile(r"\b(hair_color|skin_color|eye_color)=\{\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*\}")
SKIP_PREFIX = ("gene_dragon", "gene_height", "gene_bs_body_shape", "gene_bs_bust", "gene_age", "gene_body_hair",
               "gene_hair_type", "gene_baldness", "hairstyles", "clothes", "beards", "hair_aging", "teeth_accessory",
               "gene_no_portrait", "gene_camera_zoom", "legwear", "cloaks", "headgear", "gene_bs_ears_fantasy")
SKIP_ANCHORS = {"eth_0098"}
FRONT = (0.15, 0.12, 0.55, 0.62)
TURNED = (0.57, 0.15, 0.89, 0.62)
SIZE_DEFAULT = SIZE = 224        # default crop size; --size 336 for more detail (cache = size + 16)
MODEL = "facebook/dinov2-small"


FACE_BACKBONES = ("arcface", "glint")   # --model arcface / glint: a face-recognition network instead of DINOv2


def face_backbone(name):
    """A face-recognition network (InsightFace ArcFace r50 or glintr100, the bridge's own models) cut just before it
    boils the face down to 512 identity numbers, so it keeps its detailed face map and can be fully fine-tuned.
    Takes ImageNet-normalised N,3,H,W crops; returns N, out_dim (overall average + 2x2 layout of the face map)."""
    import torch
    import torch.nn as nn
    import onnx
    import onnx2torch
    import bridge2 as B2
    if name == "arcface":
        if not os.path.exists(B2.B1.MODEL):
            B2.B1.ArcFace()                               # downloads the ArcFace pack once (same as the bridge)
        path = B2.B1.MODEL
    else:
        if not os.path.exists(B2.GLINT):
            B2.Feats(("arc", "glint"))                    # downloads glintr100 once (same as the bridge)
        path = B2.GLINT
    gm = onnx2torch.convert(onnx.load(path))
    flat = next(n for n in gm.graph.nodes if "Flatten" in str(n.target))
    out = next(n for n in gm.graph.nodes if n.op == "output")
    out.args = (flat.args[0],)                            # stop at the face map (N, 512, h, w)
    gm.graph.eliminate_dead_code()
    gm.recompile()

    class FaceBB(nn.Module):
        def __init__(s):
            super().__init__()
            s.m = gm
            s.out_dim = 512 * 5
            s.norm = nn.LayerNorm(s.out_dim)
            s.register_buffer("mu", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            s.register_buffer("sg", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def train(s, mode=True):                          # batch-norm statistics stay as learned on real faces
            super().train(mode)
            for mod in s.m.modules():
                if "BatchNorm" in type(mod).__name__:
                    mod.eval()
            return s

        def forward(s, x):
            x = (x * s.sg + s.mu) * 2 - 1                 # ImageNet scaling -> InsightFace's (-1..1)
            h = s.m(x)
            f = torch.cat([h.mean((2, 3)), nn.functional.adaptive_avg_pool2d(h, 2).flatten(1)], 1)
            return s.norm(f.float())
    return FaceBB()


# ---------------------------------------------------------------- data
def parse(text):
    genes = {m.group(1): (m.group(2), int(m.group(3))) for m in GENE_RE.finditer(text)
             if not m.group(1).startswith(SKIP_PREFIX)}
    cols = {m.group(1): (int(m.group(2)), int(m.group(3))) for m in COLOUR_RE.finditer(text)}
    return genes, cols


HOLD_REFS = False      # --hold-out-refs: also skip faces built on any reference person (refs.csv), for fair tests


def ref_keys():
    p = os.path.join((os.environ.get("CK3FIT_HOME") or os.path.dirname(os.path.abspath(__file__))), "refs.csv")
    if not os.path.exists(p):
        return set()
    return {r["key"].strip() for r in csv.DictReader(open(p, newline="", encoding="utf-8-sig"))}


def shots(data):
    skip = set()
    bad = set(SKIP_ANCHORS) | (ref_keys() if HOLD_REFS else set())
    mp = os.path.join(data, "meta.csv")
    if os.path.exists(mp):
        for r in csv.DictReader(open(mp, newline="")):
            if r.get("anchor") in bad:
                skip.add(r["shot"])
    names = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(data, "*.jpg")))
    return [n for n in names if n not in skip and os.path.exists(os.path.join(data, n + ".txt"))], len(skip)


def crops(path, size):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    W, H = im.size
    return [im.crop((int(W * b[0]), int(H * b[1]), int(W * b[2]), int(H * b[3]))).resize((size, size))
            for b in (FRONT, TURNED)]


SIDE = re.compile(r"^(.*?)_?(neg|pos|min|max)(_\d+)?$")


def kind_of(g, templates):
    """plain: value is the slider. signed: paired neg/pos (or min/max) shapes, vanilla = middle. choice: buttons."""
    real = [t for t in templates if not t.startswith("vanilla_")]
    sides = [SIDE.match(t) for t in real]
    paired = len(real) == 2 and all(sides) and {s.group(2) for s in sides} in ({"neg", "pos"}, {"min", "max"}) \
        and len({(s.group(1), s.group(3)) for s in sides}) == 1
    if not g.startswith("gene_bs_") and paired and {s.group(2) for s in sides} == {"neg", "pos"}:
        return "plain"                           # same attribute swept low->high; min/max pairs are blend shapes
    if g in ("gene_forehead_brow_height", "gene_mouth_open"):
        return "plain"
    if paired or (len(real) == 1 and SIDE.match(real[0]) and len(templates) == 1):
        return "signed" if paired else "value"
    if len(templates) == 1:
        return "value"
    return "choice"


# sliders the game only has a _pos side for (01_genes_morph.txt): a "_neg" name isn't a real option there, so the
# game draws it like _pos at the same strength. Below the middle these are written as _pos 0 (none).
ONE_SIDED = {"gene_bs_mouth_lower_lip_def", "gene_bs_mouth_upper_lip_def", "gene_bs_mouth_philtrum_def",
             "gene_bs_ear_inner_shape"}


def positive(t):
    m = SIDE.match(t)
    return bool(m) and m.group(2) in ("pos", "max")


class Targets:
    def __init__(self, data, names, fixed=None):
        parsed = [parse(open(os.path.join(data, n + ".txt"), encoding="utf-8").read()) for n in names]
        if fixed:                                     # continuing a saved model: keep its settings and buttons
            self.genes, self.tmpl, self.kind = fixed
        else:
            self.genes = sorted(set().union(*[p[0].keys() for p in parsed]))
            self.tmpl = {g: sorted({p[0][g][0] for p in parsed if g in p[0]}) for g in self.genes}
            self.kind = {g: kind_of(g, self.tmpl[g]) for g in self.genes}
        n, G = len(names), len(self.genes)
        self.V = np.zeros((n, G), np.float32)          # 0-1 target per gene
        self.M = np.zeros((n, G), np.float32)          # 1 = counts in the value loss
        self.T = np.full((n, G), -1, np.int64)         # button index (choice genes)
        for i, (gs, _) in enumerate(parsed):
            for j, g in enumerate(self.genes):
                if g not in gs:
                    continue
                t, v = gs[g]
                k = self.kind[g]
                if k == "plain" or k == "value":
                    self.V[i, j], self.M[i, j] = v / 255, 1
                elif k == "signed":
                    up = positive(t) or g in ONE_SIDED          # a _neg name on these is drawn as _pos
                    self.V[i, j] = 0.5 if t.startswith("vanilla_") else (0.5 + v / 510 if up else 0.5 - v / 510)
                    self.M[i, j] = 1
                elif t in self.tmpl[g]:
                    self.T[i, j] = self.tmpl[g].index(t)
                    self.V[i, j], self.M[i, j] = v / 255, 0.0 if t.startswith(("vanilla_", "no_")) else 1.0
        self.C = np.array([[c for k in ("hair_color", "skin_color", "eye_color") for c in p[1].get(k, (0, 0))]
                           for p in parsed], np.float32) / 255
        self.choice = [j for j, g in enumerate(self.genes) if self.kind[g] == "choice"]

    def write(self, text, v, cls, col):
        """Put predictions (one face) into a DNA text."""
        gi = {g: j for j, g in enumerate(self.genes)}

        def rep(m):
            g = m.group(1)
            if g not in gi:
                return m.group(0)
            j, k, tm = gi[g], self.kind[g], self.tmpl[g]
            if k == "signed":
                s = float(v[j])
                side = [t for t in tm if not t.startswith("vanilla_") and positive(t) == (s >= 0.5)
                        and not (g in ONE_SIDED and not positive(t))]           # no _neg for these in the game
                if side:
                    name, val = side[0], int(round(min(255, abs(s - 0.5) * 510)))
                else:                                 # a one-sided slider (only _pos in the game, e.g. lip
                    real = [t for t in tm if not t.startswith("vanilla_")]   # definition): below the middle
                    name, val = (real[0] if real else m.group(2)), 0         # means none - not mirrored to high
            elif k == "choice":
                name, val = tm[cls[j]], int(round(float(v[j]) * 255))
            else:
                name, val = m.group(2), int(round(float(v[j]) * 255))
            return f'{g}={{ "{name}" {val} "{name}" {val} }}'

        def rep_col(m):
            o = {"hair_color": 0, "skin_color": 2, "eye_color": 4}[m.group(1)]
            x, y = (int(round(float(col[o + q]) * 255)) for q in (0, 1))
            return f"{m.group(1)}={{ {x} {y} {x} {y} }}"

        return COLOUR_RE.sub(rep_col, GENE_RE.sub(rep, text))


def ref_targets(text, tg):
    """A hand-made DNA -> targets in the same form as the training faces (0-1 values, mask, button, colours)."""
    genes, cols = parse(text)
    G = len(tg.genes)
    v, m, t = np.full(G, np.nan, np.float32), np.zeros(G, np.float32), np.full(G, -1, np.int64)
    for j, g in enumerate(tg.genes):
        if g not in genes:
            continue
        name, x = genes[g]
        k = tg.kind[g]
        if k in ("plain", "value"):
            v[j], m[j] = x / 255, 1
        elif k == "signed":
            up = positive(name) or g in ONE_SIDED
            v[j] = 0.5 if name.startswith("vanilla_") else (0.5 + x / 510 if up else 0.5 - x / 510)
            m[j] = 1
        elif name in tg.tmpl[g]:
            t[j] = tg.tmpl[g].index(name)
            v[j], m[j] = x / 255, 0.0 if name.startswith(("vanilla_", "no_")) else 1.0
    v = np.where(np.isnan(v), 0.5, v).astype(np.float32)   # masked anyway
    c = np.array([q for key in ("hair_color", "skin_color", "eye_color") for q in cols.get(key, (np.nan, np.nan))],
                 np.float32) / 255
    return v, m, t, c


# ---------------------------------------------------------------- commands
def per_folder(fn):
    def run(a):
        import copy
        for d in a.data:
            b = copy.copy(a)
            b.data = d
            fn(b)
    return run


def check(a):
    from PIL import Image
    names, _ = shots(a.data)
    tiles = [c for n in names[:4] for c in crops(os.path.join(a.data, n + ".jpg"), SIZE)]
    sheet = Image.new("RGB", (SIZE * 2, SIZE * 4))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % 2) * SIZE, (i // 2) * SIZE))
    out = os.path.join(a.data, "_check_crops.jpg")
    sheet.save(out)
    print("Saved", out)


VIEW = ""              # "_strip": the stripped pictures made by strip.py instead of the colour crops (--view strip)


def cache_name(size, ext):
    return f"_crops{'' if size == 224 else '_' + str(size)}{VIEW}{ext}"


def prepare(a):
    names, skipped = shots(a.data)
    if a.limit:
        names = names[:a.limit]
    CACHE_SIZE = a.size + 16
    print(f"{len(names)} faces ({skipped} skipped). Cutting crops ...")
    arr = np.lib.format.open_memmap(os.path.join(a.data, cache_name(a.size, ".npy")), mode="w+", dtype=np.uint8,
                                    shape=(len(names), 2, CACHE_SIZE, CACHE_SIZE, 3))
    for i, n in enumerate(names):
        for k, c in enumerate(crops(os.path.join(a.data, n + ".jpg"), CACHE_SIZE)):
            arr[i, k] = np.asarray(c)
        if i % 500 == 0:
            print(f"  {i} / {len(names)}")
    arr.flush()
    np.save(os.path.join(a.data, cache_name(a.size, "_names.npy")), np.array(names))
    print("Done. Next: python train_reverse.py train --data " + a.data)


def prepare_tilt(a):
    """crop cache of the tipped-up / tipped-down front views (autoshot.py tilt) -> <tilt>\\_tilt_crops.npy"""
    names = sorted(f[:-7] for f in os.listdir(a.tilt) if f.endswith("_up.jpg")
                   and os.path.exists(os.path.join(a.tilt, f[:-7] + "_down.jpg")))
    if not names:
        sys.exit(f"No <name>_up.jpg / _down.jpg pairs in {a.tilt}.")
    C = a.size + 16
    arr = np.lib.format.open_memmap(os.path.join(a.tilt, cache_name(a.size, "_tilt.npy")), mode="w+",
                                    dtype=np.uint8, shape=(len(names), 2, C, C, 3))
    for i, n in enumerate(names):
        for k, v in enumerate(("up", "down")):
            arr[i, k] = np.asarray(crops(os.path.join(a.tilt, f"{n}_{v}.jpg"), C)[0])
        if i % 500 == 0:
            print(f"  {i} / {len(names)}")
    arr.flush()
    np.save(os.path.join(a.tilt, cache_name(a.size, "_tilt_names.npy")), np.array(names))
    print(f"{len(names)} tilted faces cached. Train with  --tilt-data {a.tilt}")


def train(a):
    import torch
    import torch.nn as nn
    from transformers import AutoModel
    global VIEW
    VIEW = "_" + a.view if a.view else ""
    if VIEW:
        print(f"Training on the '{a.view}' pictures (strip.py) instead of colour crops.")
    SIZE, CACHE_SIZE = a.size, a.size + 16
    parts, names, where = [], [], []                  # one or more face folders, each with its own crop cache
    for d in a.data:
        if not os.path.exists(os.path.join(d, cache_name(a.size, ".npy"))):
            sys.exit(f"Run the prepare step first for {d}." + (" (python strip.py prepare --data ...)" if VIEW else ""))
        k = len(parts)
        parts.append(np.load(os.path.join(d, cache_name(a.size, ".npy")), mmap_mode="r"))
        nm = [str(n) for n in np.load(os.path.join(d, cache_name(a.size, "_names.npy")))]
        drop = set()
        if a.hold_out_refs:                           # faces built on a reference person: left out (fair tests)
            mp = os.path.join(d, "meta.csv")
            refs = ref_keys()
            if os.path.exists(mp):
                drop = {r["shot"] for r in csv.DictReader(open(mp, newline="")) if r.get("anchor") in refs}
            print(f"{d}: {len(drop & set(nm))} faces built on reference people left out (--hold-out-refs)")
        for i, n in enumerate(nm):
            if n not in drop:
                names.append(os.path.join(d, n))
                where.append((k, i))

    class Imgs:
        def __getitem__(self, i):
            k, j = where[i]
            return parts[k][j]
    imgs = Imgs()
    tilt_arr, tilt_row = None, None
    if a.tilt_data:                                   # --tilt-data: the same faces tipped up / down (front view)
        tilt_arr = np.load(os.path.join(a.tilt_data, cache_name(a.size, "_tilt.npy")), mmap_mode="r")
        tn = {str(n): k for k, n in enumerate(np.load(os.path.join(a.tilt_data, cache_name(a.size, "_tilt_names.npy"))))}
        tilt_row = np.array([tn.get(os.path.basename(n), -1) if os.path.dirname(n) == a.data[0] else -1
                             for n in names])           # only the first --data folder's faces have tilted views
        print(f"Tilted front views for {int((tilt_row >= 0).sum())} faces; used on {a.tilt_p:.0%} of their draws")
    if a.limit:                                       # learning-curve runs: only the first N faces
        names = names[:a.limit]
        print(f"Using only the first {len(names)} faces (--limit).")
    tag = f"_{len(names)}_v3"                         # outputs named by face count + any non-default settings
    if a.model != MODEL:
        tag += "_" + a.model.split("-")[-1]
    if a.size != SIZE_DEFAULT:
        tag += f"_px{a.size}"
    if a.lr_backbone != 2e-5:
        tag += f"_lr{a.lr_backbone:g}"
    if a.photo_aug:
        tag += "_photo"
    if a.refs:
        tag += "_refs"
    if a.init:
        tag += "_ft"
    if a.name:
        tag = "_" + a.name
    # results go to CK3Fit\\results (next to the script), not into the big data folder
    res = os.path.join((os.environ.get("CK3FIT_HOME") or os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(res, exist_ok=True)
    print("Outputs will be named with", tag, "in", res)
    ck0 = torch.load(a.init, map_location="cpu", weights_only=False) if a.init else None
    ckg = ck0 or (torch.load(a.genes_from, map_location="cpu", weights_only=False) if a.genes_from else None)
    if ckg is not None and not a.init:
        print(f"Settings and button lists taken from {a.genes_from} (so the bridge's picks line up)")
    tg = Targets("", names, fixed=(ckg["genes"], ckg["tmpl"], ckg["kind"]) if ckg else None)
    print(f"{len(names)} faces; settings: " + ", ".join(
        f"{sum(1 for k in tg.kind.values() if k == x)} {x}" for x in ("plain", "signed", "value", "choice")))
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(names))
    n_test = max(24, len(names) // 10)
    te, tr = idx[:n_test], idx[n_test:]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using", torch.cuda.get_device_name(0) if dev == "cuda" else "the processor (very slow - stop and tell me)")

    G = len(tg.genes)
    nt = {j: len(tg.tmpl[tg.genes[j]]) for j in tg.choice}

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.face = a.model in FACE_BACKBONES
            if s.face:
                s.bb = face_backbone(a.model)
                d = s.bb.out_dim * 2
            else:
                s.bb = AutoModel.from_pretrained(a.model)
                d = s.bb.config.hidden_size * 4
            s.trunk = nn.Sequential(nn.Linear(d, 1024), nn.GELU(), nn.Dropout(0.1), nn.Linear(1024, 1024), nn.GELU())
            s.val, s.col = nn.Linear(1024, G), nn.Linear(1024, 6)
            s.cls = nn.ModuleDict({str(j): nn.Linear(1024, nt[j]) for j in tg.choice})

        def forward(s, x):                       # x: B,2,3,H,W
            B = x.shape[0]
            if s.face:
                f = s.bb(x.flatten(0, 1)).view(B, -1)
            else:
                h = s.bb(pixel_values=x.flatten(0, 1)).last_hidden_state
                f = torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).view(B, -1)
            z = s.trunk(f)
            return s.val(z), s.col(z), {j: c(z) for j, c in s.cls.items()}      # in units of each setting's spread

    net = Net().to(dev)
    opt = torch.optim.AdamW([{"params": net.bb.parameters(), "lr": a.lr_backbone},
                             {"params": [p for n_, p in net.named_parameters() if not n_.startswith("bb.")], "lr": a.lr_head}],
                            weight_decay=0.02)
    n_ref = int(round(a.batch * a.ref_frac)) if a.refs else 0
    gb = a.batch - n_ref                                  # game faces per batch
    steps = a.epochs * (len(tr) // gb)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr_backbone, a.lr_head], total_steps=steps,
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 1, 3, 1, 1)
    ce = nn.CrossEntropyLoss(ignore_index=-1)
    # every slider and colour is scaled by its own typical spread, so each one counts equally in the loss
    # (before, sliders were ~3% of the loss and the button choices swamped them)
    Mtr = tg.M[tr] > 0
    mu = np.array([tg.V[tr][Mtr[:, j], j].mean() if Mtr[:, j].any() else 0.5 for j in range(G)], np.float32)
    sd = np.array([max(tg.V[tr][Mtr[:, j], j].std(), 0.02) if Mtr[:, j].any() else 1.0 for j in range(G)], np.float32)
    cmu, csd = tg.C[tr].mean(0), np.maximum(tg.C[tr].std(0), 0.02)
    if a.init:
        if ck0["genes"] != tg.genes or ck0["model"] != a.model:
            sys.exit("--init model was trained on different settings or a different --model")
        net.load_state_dict(ck0["state"])
        mu, sd, cmu, csd = ck0["mu"], ck0["sd"], ck0["cmu"], ck0["csd"]
        print("Starting from", a.init)
    Vn = (tg.V - mu) / sd
    Cn = ((tg.C - cmu) / csd).astype(np.float32)
    V, M, C, T = (torch.from_numpy(x).to(dev) for x in (Vn.astype(np.float32), tg.M, Cn, tg.T))

    # real photos paired with your hand-made DNA (prep_refs.py): mixed into every training batch
    ref_tr, ref_te = [], []
    if a.refs:
        import pickle
        rp = pickle.load(open(a.refs, "rb"))
        if rp["size"] != SIZE:
            sys.exit(f"{a.refs} was made for size {rp['size']}; rerun prep_refs.py --size {SIZE}")
        for x in rp["people"]:
            v, m, t, c = ref_targets(x["dna"], tg)
            x.update(v=(v - mu) / sd, m=m, t=t, c=(np.where(np.isnan(c), cmu, c) - cmu) / csd, v01=v)
            (ref_te if x["held_out"] else ref_tr).append(x)
        RV, RM, RC, RT = (torch.from_numpy(np.stack([x[k] for x in ref_tr]).astype(dt)).to(dev)
                          for k, dt in (("v", np.float32), ("m", np.float32), ("c", np.float32), ("t", np.int64)))
        print(f"Real photos: {len(ref_tr)} people mixed in ({n_ref} of every {a.batch} per batch), "
              f"{len(ref_te)} held out: " + ", ".join(x["id"] for x in ref_te))

    refiner = None
    if a.refiner:
        import refine as RF
        refiner = RF.load(a.refiner, dev)
        print(f"Refiner {a.refiner}: {a.refine_p:.0%} of game crops made photo-like by it")

    # --rank-refs: the hand-made DNA used for ORDER only (who has the wider nose of two people), not exact values
    rk_tr, rk_te = [], []
    if a.rank_refs:
        import pickle
        rp = pickle.load(open(a.rank_refs, "rb"))
        if rp["size"] != SIZE:
            sys.exit(f"{a.rank_refs} was made for size {rp['size']}")
        slide_ok = np.array([tg.kind[g] != "choice" for g in tg.genes]) & (sd * 255 >= 2)
        for x in rp["people"]:
            if len(x["front"]) == 0 or len(x["turned"]) == 0:
                continue
            if a.rank_sex and not re.match(r"(extra_)?" + a.rank_sex[0], str(x["id"])):   # f_12 / extra_f02 / m_3
                continue
            v, m, t, c = ref_targets(x["dna"], tg)
            x.update(rv=((v - mu) / sd).astype(np.float32), rm=(m * slide_ok).astype(np.float32))
            (rk_te if x["held_out"] else rk_tr).append(x)
        RKV = torch.from_numpy(np.stack([x["rv"] for x in rk_tr])).to(dev)
        RKM = torch.from_numpy(np.stack([x["rm"] for x in rk_tr])).to(dev)
        print(f"Ranking with the guides: {len(rk_tr)} people for training, {len(rk_te)} held out for the check")

    # --consistency: unlabelled photo sets (several photos of one person): different photos of the same person
    # must read alike, compared with how differently different people read (so "everyone the same" doesn't pass)
    cs_tr, cs_te = [], []
    if a.consistency:
        import pickle
        cp_ = pickle.load(open(a.consistency, "rb"))
        if cp_["size"] != SIZE:
            sys.exit(f"{a.consistency} was made for size {cp_['size']}")
        ok_ = [x for x in cp_["people"] if len(x["front"]) * max(1, len(x["turned"])) >= 2]
        for i, x in enumerate(ok_):
            (cs_te if i % 5 == 4 else cs_tr).append(x)
        CS_SL = torch.from_numpy(np.array([tg.kind[g] != "choice" for g in tg.genes], np.float32)).to(dev)
        print(f"Consistency on photo sets: {len(cs_tr)} people for training, {len(cs_te)} held out for the check")

    def cons_batch(k, people, train_mode=True):
        who = np.random.choice(len(people), min(k, len(people)), replace=False)
        xs = []
        for w in who:
            fr, tu = people[w]["front"], people[w]["turned"]
            for _ in range(2):                         # two different photo pairs of the same person
                xs.append(np.stack([fr[np.random.randint(len(fr))], tu[np.random.randint(len(tu))]]))
        return prep(np.stack(xs), train_mode, game=False)

    def cons_loss(v):
        """v: 2K,G (pairs of the same person next to each other) -> within-person spread / between-person spread"""
        v = v.view(-1, 2, v.shape[1])
        within = ((v[:, 0] - v[:, 1]) ** 2).mean(0) / 2
        between = v.mean(1).var(0)
        return ((within / (between + 0.05)) * CS_SL).sum() / CS_SL.sum()

    def cons_check(people):
        if not people:
            return ""
        net.eval()
        with torch.no_grad(), torch.autocast(dev, enabled=dev == "cuda"):
            r_ = []
            for _ in range(4):
                v, _, _ = net(cons_batch(16, people, False))
                r_.append(cons_loss(v.float()).item())
        return f", photo sets: within/between spread {np.mean(r_):.2f} on {len(people)} held-out people"

    def rank_batch(k):
        who = np.random.choice(len(rk_tr), min(k, len(rk_tr)), replace=False)
        x = np.stack([np.stack([rk_tr[w]["front"][np.random.randint(len(rk_tr[w]["front"]))],
                                rk_tr[w]["turned"][np.random.randint(len(rk_tr[w]["turned"]))]]) for w in who])
        return prep(x, True, game=False), torch.from_numpy(who).to(dev)

    def rank_loss(v, who):
        """for every pair of people and every slider the guides clearly separate (>0.3 of a spread): the network's
        values must come out in the same order, by a margin"""
        tv, tm = RKV[who], RKM[who]
        dv = v.unsqueeze(1) - v.unsqueeze(0)
        dt = tv.unsqueeze(1) - tv.unsqueeze(0)
        ok = (tm.unsqueeze(1) * tm.unsqueeze(0)) * (dt.abs() > 0.3).float()
        return (torch.relu(0.3 - torch.sign(dt) * dv) * ok).sum() / ok.sum().clamp(min=1)

    def rank_check(people):
        """share of clearly-ordered slider pairs (guides differ by >0.3 spread) that the network orders the same way"""
        if not people:
            return ""
        net.eval()
        vs = []
        with torch.no_grad(), torch.autocast(dev, enabled=dev == "cuda"):
            for x in people:
                pr = [np.stack([f, t]) for f in x["front"] for t in x["turned"]][:16]
                v, _, _ = net(prep(np.stack(pr), False, game=False))
                vs.append(v.float().cpu().numpy().mean(0))
        vs = np.array(vs)
        tv, tm = np.stack([x["rv"] for x in people]), np.stack([x["rm"] for x in people])
        good = tot = 0
        for i in range(len(people)):
            for j in range(i + 1, len(people)):
                ok = (tm[i] > 0) & (tm[j] > 0) & (np.abs(tv[i] - tv[j]) > 0.3)
                good += (np.sign(vs[i] - vs[j]) == np.sign(tv[i] - tv[j]))[ok].sum()
                tot += ok.sum()
        return f", guides' order kept {good / max(tot, 1):.1%} on {len(people)} held-out people"

    import torch.nn.functional as F

    def photo_like(x):
        """Make game crops look more like aligned photos. Each crop (front, turned) gets its own random changes,
        as if they came from different photos: small misalignment, side lighting, brightness/contrast, a mild colour
        cast, blur or sharpness loss, and noise. x: B,2,3,H,W in 0-1 (cache size) -> B,2,3,SIZE,SIZE."""
        with torch.autocast(dev, enabled=False):                   # full precision (cheap; avoids half/float mixes)
            return _photo_like(x.float())

    def _photo_like(x):
        B = x.shape[0]
        x = x.flatten(0, 1)                                        # 2B,3,H,W
        n = x.shape[0]
        r = lambda *s: torch.rand(*s, device=dev)
        # misalignment: scale +-8%, shift +-5%, rotation +-6 degrees, then the training-size centre crop
        ang = (r(n) - 0.5) * 2 * np.radians(6)
        sc = 1 + (r(n) - 0.5) * 0.16
        k = SIZE / CACHE_SIZE / sc
        th = torch.stack([torch.stack([k * torch.cos(ang), -k * torch.sin(ang), (r(n) - 0.5) * 0.1], 1),
                          torch.stack([k * torch.sin(ang), k * torch.cos(ang), (r(n) - 0.5) * 0.1], 1)], 1)
        grid = F.affine_grid(th, (n, 3, SIZE, SIZE), align_corners=False)
        if a.rand_aug:                       # --rand-aug: other lenses / camera distances and squeeze (sim-to-real
            c0 = torch.tensor([0.0, 0.05], device=dev).view(1, 1, 1, 2)          # randomisation of the camera)
            g0 = grid - c0
            kr = ((r(n) - 0.5) * 0.30).view(n, 1, 1, 1)                          # radial: barrel <-> pincushion
            asp = torch.stack([1 + (r(n) - 0.5) * 0.12, torch.ones(n, device=dev)], 1).view(n, 1, 1, 2)
            grid = c0 + g0 * (1 + kr * (g0 ** 2).sum(-1, keepdim=True)) * asp
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
        if VIEW:                             # stripped pictures: no lighting or colour to imitate. Photos give fainter
            # or heavier lines, softer pictures, skin speckle and a stronger or weaker relief than game crops do.
            x = torch.cat([x[:, :1] * (0.5 + r(n, 1, 1, 1) * 0.9),
                           0.5 + (x[:, 1:] - 0.5) * (0.7 + r(n, 1, 1, 1) * 0.6)], 1)
            sig = r(n) * 1.5
            blur = r(n) < 0.5
            if blur.any():
                t = torch.arange(-3, 4, device=dev).float()
                kern = torch.exp(-(t.view(1, -1) ** 2) / (2 * sig[blur].view(-1, 1).clamp(min=0.3) ** 2))
                kern = kern / kern.sum(1, keepdim=True)
                xb = x[blur]
                nb = xb.shape[0]
                kx = kern.view(nb, 1, 1, 7).repeat_interleave(3, 0)
                ky = kern.view(nb, 1, 7, 1).repeat_interleave(3, 0)
                xb = xb.reshape(1, nb * 3, SIZE, SIZE)
                xb = F.conv2d(F.pad(xb, (3, 3, 0, 0), mode="replicate"), kx, groups=nb * 3)
                xb = F.conv2d(F.pad(xb, (0, 0, 3, 3), mode="replicate"), ky, groups=nb * 3)
                x[blur] = xb.view(nb, 3, SIZE, SIZE).to(x.dtype)
            x = torch.cat([x[:, :1] + torch.randn_like(x[:, :1]).clamp(min=0) * (r(n) * 0.10).view(n, 1, 1, 1), x[:, 1:]], 1)
            return x.clamp(0, 1).view(B, 2, 3, SIZE, SIZE)
        # side lighting: a soft brightness ramp across the face in a random direction (+-30%)
        d = r(n) * 2 * np.pi
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, SIZE, device=dev), torch.linspace(-1, 1, SIZE, device=dev),
                                indexing="ij")
        ramp = torch.cos(d).view(n, 1, 1) * xx + torch.sin(d).view(n, 1, 1) * yy
        x = x * (1 + ramp.unsqueeze(1) * (r(n) * 0.3).view(n, 1, 1, 1))
        # brightness +-20%, contrast +-25%, colour cast +-6% per channel (colours are targets: keep casts mild)
        m = x.mean((1, 2, 3), keepdim=True)
        x = (x - m) * (1 + (r(n, 1, 1, 1) - 0.5) * 0.5) + m
        x = x * (1 + (r(n, 1, 1, 1) - 0.5) * 0.4) * (1 + (r(n, 3, 1, 1) - 0.5) * 0.12)
        # blur on half the crops (sigma up to 1.5 px)
        sig = r(n) * 1.5
        blur = r(n) < 0.5
        if blur.any():
            t = torch.arange(-3, 4, device=dev).float()
            kern = torch.exp(-(t.view(1, -1) ** 2) / (2 * sig[blur].view(-1, 1).clamp(min=0.3) ** 2))
            kern = kern / kern.sum(1, keepdim=True)
            xb = x[blur]
            nb = xb.shape[0]
            kx = kern.view(nb, 1, 1, 7).repeat_interleave(3, 0)
            ky = kern.view(nb, 1, 7, 1).repeat_interleave(3, 0)
            xb = xb.reshape(1, nb * 3, SIZE, SIZE)
            xb = F.conv2d(F.pad(xb, (3, 3, 0, 0), mode="replicate"), kx, groups=nb * 3)
            xb = F.conv2d(F.pad(xb, (0, 0, 3, 3), mode="replicate"), ky, groups=nb * 3)
            x[blur] = xb.view(nb, 3, SIZE, SIZE).to(x.dtype)
        if a.rand_aug:
            x = _rand_extra(x.clamp(0, 1), n, r, yy, xx)
        if a.hair_aug > 0:
            x = _hair_cover(x.clamp(0, 1), n, r, yy, xx)
        # sensor noise up to 3%
        x = x + torch.randn_like(x) * (r(n) * 0.03).view(n, 1, 1, 1)
        return x.clamp(0, 1).view(B, 2, 3, SIZE, SIZE)

    def _hair_cover(x, n, r, yy, xx):
        """--hair-aug: loose hair over the sides of the face and the jaw, as in many photos. On a share of the crops,
        one or both sides get a hair-coloured band with a wavy inner edge and strand streaks; its inner edge lands
        anywhere from the face's outline to well over the cheek, and it reaches down to somewhere between the eyes
        and below the chin. The DNA target is unchanged, so the network has to read the face shape through it."""
        u = (xx + 1) / 2                                            # 0..1 across, 0..1 down
        v = (yy + 1) / 2
        cover = torch.zeros(n, 1, SIZE, SIZE, device=dev)
        front = (torch.arange(n, device=dev) % 2 == 0).float()      # crops alternate front / turned
        for side in (0, 1):
            on = (r(n) < a.hair_aug * 0.6).float()
            if side == 1:                                           # the turned view faces right: hair only
                on = on * front                                     # on its left (cheek / back) side
            on = on.view(n, 1, 1)
            edge = (0.12 + 0.16 * r(n)) if side == 0 else (0.74 + 0.14 * r(n))
            bottom = 0.55 + 0.5 * r(n)
            wave = 0.025 * torch.sin(v.unsqueeze(0) * (6 + 8 * r(n)).view(n, 1, 1) + (r(n) * 6.3).view(n, 1, 1))
            dist = (edge.view(n, 1, 1) + wave - u.unsqueeze(0)) * (1 if side == 0 else -1)   # >0 inside the hair
            m = torch.sigmoid(dist * 60) * torch.sigmoid((bottom.view(n, 1, 1) - v.unsqueeze(0)) * 25)
            cover = torch.maximum(cover, (m * on).unsqueeze(1))
        base = torch.stack([0.10 + 0.55 * r(n), torch.zeros(n, device=dev), torch.zeros(n, device=dev)], 1)
        base[:, 1] = base[:, 0] * (0.80 + 0.1 * r(n))
        base[:, 2] = base[:, 0] * (0.55 + 0.15 * r(n))              # brown/black to blonde
        streak = F.interpolate(torch.rand(n, 1, 6, SIZE // 3, device=dev), size=(SIZE, SIZE), mode="bilinear",
                               align_corners=False)
        hair = (base.view(n, 3, 1, 1) * (0.6 + 0.8 * streak)).clamp(0, 1)
        return x * (1 - cover) + hair * cover

    def _rand_extra(x, n, r, yy, xx):
        """--rand-aug: what photos have and the Ruler Designer never does - other backgrounds, hard side light and
        shadow, different exposure curves and white balance, low resolution. x: n,3,S,S in 0-1."""
        # background: outside a soft oval around the head, random smooth colour fields (half the crops)
        bg = F.interpolate(torch.rand(n, 3, 4, 4, device=dev), size=(SIZE, SIZE), mode="bicubic", align_corners=False)
        bg = (bg * (0.3 + 0.7 * r(n, 1, 1, 1))).clamp(0, 1)
        ell = ((xx / 0.92) ** 2 + ((yy + 0.05) / 1.05) ** 2).unsqueeze(0).unsqueeze(0)
        m = torch.sigmoid((ell - 1.0) * 12) * (r(n, 1, 1, 1) < 0.5).float()
        x = x * (1 - m) + bg * m
        # hard side light: one half of the face up to 45% darker, with a soft edge
        d = r(n) * 2 * np.pi
        side = torch.cos(d).view(n, 1, 1) * xx + torch.sin(d).view(n, 1, 1) * yy
        sh = torch.sigmoid(side * (4 + 8 * r(n).view(n, 1, 1))).unsqueeze(1)
        x = x * (1 - sh * (r(n) * 0.45 * (r(n) < 0.6).float()).view(n, 1, 1, 1))
        # exposure curve (gamma 0.7-1.4) and white balance (warm / cool, +-10%)
        x = x.clamp(1e-4, 1) ** (0.7 + 0.7 * r(n, 1, 1, 1))
        wb = (r(n) - 0.5) * 0.2
        x = x * torch.stack([1 + wb, torch.ones(n, device=dev), 1 - wb], 1).view(n, 3, 1, 1)
        # low resolution: down to 56-160 px and back (40% of the crops)
        lo = r(n) < 0.4
        if lo.any():
            s_ = int(56 + np.random.rand() * 104)
            x[lo] = F.interpolate(F.interpolate(x[lo], size=(s_, s_), mode="area"), size=(SIZE, SIZE),
                                  mode="bilinear", align_corners=False)
        return x

    def batch(ids, train_mode):
        xs = [imgs[i] for i in ids]
        if train_mode and tilt_arr is not None:      # same DNA, front view tipped up or down
            for k_, i in enumerate(ids):
                r_ = tilt_row[i] if i < len(tilt_row) else -1
                if r_ >= 0 and np.random.rand() < a.tilt_p:
                    x_ = np.array(xs[k_])
                    x_[0] = tilt_arr[r_, np.random.randint(2)]
                    xs[k_] = x_
        return prep(np.stack(xs), train_mode)

    def ref_batch(k):
        """k random (front, turned) photo pairs from the training reference people -> images + person index."""
        who = np.random.randint(0, len(ref_tr), k)
        x = np.stack([np.stack([ref_tr[w]["front"][np.random.randint(len(ref_tr[w]["front"]))],
                                ref_tr[w]["turned"][np.random.randint(len(ref_tr[w]["turned"]))]]) for w in who])
        return prep(x, True, game=False), torch.from_numpy(who).to(dev)

    def ref_check():
        """Held-out reference people: slider error vs simply guessing the training reference people's average DNA."""
        if not ref_te:
            return ""
        net.eval()
        guess = np.nanmean(np.stack([x["v01"] for x in ref_tr]), 0)
        e_net, e_guess = [], []
        with torch.no_grad(), torch.autocast(dev, enabled=dev == "cuda"):
            for x in ref_te:
                pr = [np.stack([f, t]) for f in x["front"] for t in x["turned"]][:16]
                v, _, _ = net(prep(np.stack(pr), False))
                v = np.clip(v.float().cpu().numpy().mean(0) * sd + mu, 0, 1)
                ok = (x["m"] > 0) & np.array([tg.kind[g] != "choice" for g in tg.genes]) & (sd * 255 >= 2)
                e_net.append(np.abs(v - x["v01"])[ok].mean() * 255)
                e_guess.append(np.abs(guess - x["v01"])[ok].mean() * 255)
        return f", held-out real photos {np.mean(e_net):.1f} (guessing {np.mean(e_guess):.1f})"

    def prep(x, train_mode, game=True):
        x = torch.from_numpy(x).to(dev)                                          # B,2,240,240,3 uint8
        x = x.permute(0, 1, 4, 2, 3).float() / 255
        off = CACHE_SIZE - SIZE
        if train_mode and game and refiner is not None:     # --refiner: make (some) game crops look like photos
            sel = torch.rand(x.shape[0], device=dev) < a.refine_p
            if sel.any():
                with torch.no_grad():
                    xs = x[sel]
                    x[sel] = refiner(xs.flatten(0, 1)).view_as(xs).float()
        if train_mode and a.photo_aug:
            x = photo_like(x)
        elif train_mode:
            dx, dy = np.random.randint(0, off + 1, 2)
            x = x[..., dy:dy + SIZE, dx:dx + SIZE]
            b = 1 + (torch.rand(len(ids), 1, 1, 1, 1, device=dev) - 0.5) * 0.2      # brightness +-10%
            c = 1 + (torch.rand(len(ids), 1, 1, 1, 1, device=dev) - 0.5) * 0.2      # contrast +-10%
            m = x.mean((2, 3, 4), keepdim=True)
            x = ((x - m) * c + m) * b
            x = x.clamp(0, 1)
        else:
            x = x[..., off // 2:off // 2 + SIZE, off // 2:off // 2 + SIZE]
        return (x - mean) / std

    def predict(ids):
        net.eval()
        vs, cs, cl = [], [], {j: [] for j in tg.choice}
        with torch.no_grad(), torch.autocast(dev, enabled=dev == "cuda"):
            for k in range(0, len(ids), 64):
                v, c, l = net(batch(ids[k:k + 64], False))
                vs.append(v.float().cpu())
                cs.append(c.float().cpu())
                for j in tg.choice:
                    cl[j].append(l[str(j)].argmax(1).cpu())
        v = np.clip(torch.cat(vs).numpy() * sd + mu, 0, 1)
        c = np.clip(torch.cat(cs).numpy() * csd + cmu, 0, 1)
        return v, c, {j: torch.cat(x).numpy() for j, x in cl.items()}

    def score(ids, v):
        reads = []
        for j in range(G):
            m = tg.M[ids, j] > 0
            if m.sum() < 5:
                continue
            err = np.abs(v[m, j] - tg.V[ids][m, j]).mean()
            base = np.abs(tg.V[tr][tg.M[tr, j] > 0, j].mean() - tg.V[ids][m, j]).mean()
            if base * 255 < 2:                    # setting (almost) never changes here: nothing to read
                continue
            reads.append(1 - err / max(base, 1e-6))
        return np.median(reads)

    if rk_te or cs_te:
        print("Before training:" + rank_check(rk_te) + cons_check(cs_te))
    t0 = time.time()
    for ep in range(a.epochs):
        net.train()
        frozen = ep < a.freeze_epochs                  # --freeze-epochs: backbone untouched for the first epochs
        for p_ in net.bb.parameters():
            p_.requires_grad = not frozen
        if frozen and ep == 0:
            print(f"Backbone frozen for {a.freeze_epochs} of {a.epochs} epochs (only the slider reader learns)")
        perm = rng.permutation(tr)
        tot = 0.0
        for k in range(0, len(perm) - gb + 1, gb):
            ids = perm[k:k + gb]
            it = torch.from_numpy(ids).to(dev)
            x, Vb, Mb, Cb, Tb = batch(ids, True), V[it], M[it], C[it], T[it]
            if n_ref:
                xr, w = ref_batch(n_ref)
                x = torch.cat([x, xr])
                Vb, Mb, Cb, Tb = torch.cat([Vb, RV[w]]), torch.cat([Mb, RM[w]]), torch.cat([Cb, RC[w]]), \
                    torch.cat([Tb, RT[w]])
            ncs = 0
            if cs_tr:
                xc = cons_batch(a.cons_k, cs_tr)
                ncs = xc.shape[0]
                x = torch.cat([x, xc])
            nk = 0
            if rk_tr:
                xk, wk = rank_batch(a.rank_k)
                nk = xk.shape[0]
                x = torch.cat([x, xk])
            with torch.autocast(dev, enabled=dev == "cuda"):
                v, c, l = net(x)
            v, c = v.float(), c.float()
            rk_loss = 0.0
            if nk:
                rk_loss = rank_loss(v[-nk:], wk)
                v, c, l = v[:-nk], c[:-nk], {j: t[:-nk] for j, t in l.items()}
            cs_loss = 0.0
            if ncs:
                cs_loss = cons_loss(v[-ncs:])
                v, c, l = v[:-ncs], c[:-ncs], {j: t[:-ncs] for j, t in l.items()}
            val_loss = (((v - Vb) ** 2) * Mb).sum() / Mb.sum().clamp(min=1)
            col_loss = ((c - Cb) ** 2).mean()
            ce_loss = sum(ce(l[str(j)].float(), Tb[:, j]) for j in tg.choice) / max(1, len(tg.choice))
            loss = val_loss + 0.5 * col_loss + 0.5 * ce_loss + a.rank_w * rk_loss + a.cons_w * cs_loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        vt, _, _ = predict(te)
        print(f"  epoch {ep + 1}/{a.epochs}: loss {tot / max(1, len(perm) // gb):.4f}, held-out settings read "
              f"{score(te, vt) * 100:.0f}% (median){ref_check()}{rank_check(rk_te)}{cons_check(cs_te)}, {(time.time() - t0) / 60:.1f} min")

    torch.save({"state": net.state_dict(), "genes": tg.genes, "tmpl": tg.tmpl, "kind": tg.kind, "model": a.model,
                "mu": mu, "sd": sd, "cmu": cmu, "csd": csd, "size": a.size, "view": a.view,
                "refs_held_out": [x["id"] for x in ref_te], "refs_trained": [x["id"] for x in ref_tr]},
               os.path.join(res, f"model{tag}.pt"))
    v, c, cl = predict(te)
    vtr, _, _ = predict(tr[:len(te)])
    rows = []
    for j, g in enumerate(tg.genes):
        m = tg.M[te, j] > 0
        if m.sum() < 5:
            continue
        err = np.abs(v[m, j] - tg.V[te][m, j]).mean() * 255
        base = np.abs(tg.V[tr][tg.M[tr, j] > 0, j].mean() - tg.V[te][m, j]).mean() * 255
        if base < 2 and j not in cl:
            continue
        acc = ba = None
        if j in cl:
            acc = (cl[j] == tg.T[te, j]).mean()
            ba = (tg.T[te, j] == np.bincount(tg.T[tr, j][tg.T[tr, j] >= 0]).argmax()).mean()
        rows.append((g, tg.kind[g], err, base, 1 - err / max(base, 1e-6), acc, ba))
    rows.sort(key=lambda r: -r[4])
    ce_err = np.abs(c - tg.C[te]).mean(0) * 255
    ce_base = np.abs(tg.C[tr].mean(0) - tg.C[te]).mean(0) * 255
    by = {}
    for r in rows:
        by.setdefault(r[1], []).append(r[4])
    lines = [f"Held-out faces: {len(te)} (trained on {len(tr)}), {a.epochs} epochs, {a.model}.",
             f"Settings read, median: held-out {score(te, v) * 100:.0f}%, training faces {score(tr[:len(te)], vtr) * 100:.0f}%",
             "  by kind: " + ", ".join(f"{k} {np.median(x) * 100:.0f}% ({len(x)})" for k, x in by.items()),
             "Colours (error / guess-average error, 0-255):"]
    for k, nm in enumerate(["hair x", "hair y", "skin x", "skin y", "eye x", "eye y"]):
        lines.append(f"  {nm:7s} {ce_err[k]:5.1f} / {ce_base[k]:5.1f}")
    lines.append("\nSetting                              kind     value err / guess   read    button acc / guess")
    for g, k, err, base, rd, acc, ba in rows:
        b = f"   {acc * 100:4.0f}% / {ba * 100:4.0f}%" if acc is not None else ""
        lines.append(f"  {g:36s} {k:7s} {err:5.1f} / {base:5.1f}   {rd * 100:4.0f}%{b}")
    rep = "\n".join(lines)
    print(rep)
    open(os.path.join(res, f"report{tag}.txt"), "w").write(rep)

    out = os.path.join(res, f"pred{tag}")
    os.makedirs(out, exist_ok=True)
    for f in glob.glob(os.path.join(out, "*")):
        os.remove(f)
    for k, i in enumerate(te[:24]):
        true = open(names[i] + ".txt", encoding="utf-8").read()
        pred = tg.write(true, v[k], {j: cl[j][k] for j in cl}, c[k])
        stem = os.path.basename(os.path.dirname(names[i])) + "_" + os.path.basename(names[i])
        open(os.path.join(out, stem + "_true.txt"), "w", encoding="utf-8").write(true)
        open(os.path.join(out, stem + "_pred.txt"), "w", encoding="utf-8").write(pred)
    print(f"\nSaved in {res}: report{tag}.txt, model{tag}.pt, and 24 true/predicted pairs in pred{tag}.")
    print(f"Screenshot them:  python autoshot.py folder {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pt = sub.add_parser("prepare-tilt")
    pt.add_argument("--tilt", required=True, help="folder from autoshot.py tilt")
    pt.add_argument("--size", type=int, default=SIZE_DEFAULT)
    for name in ("check", "prepare", "train"):
        p = sub.add_parser(name)
        p.add_argument("--data", required=True, nargs="+", help="one or more folders of faces")
        p.add_argument("--epochs", type=int, default=20)
        p.add_argument("--batch", type=int, default=32)
        p.add_argument("--lr-backbone", type=float, default=2e-5)
        p.add_argument("--model", default=MODEL)
        p.add_argument("--limit", type=int, default=0, help="use only the first N faces")
        p.add_argument("--size", type=int, default=SIZE_DEFAULT, help="crop size in pixels (224 or 336)")
        p.add_argument("--view", default="", choices=["", "strip"],
                       help="train: 'strip' = the stripped pictures from strip.py prepare (lines + surface tilt) instead of colour")
        p.add_argument("--photo-aug", action="store_true", help="train: make game crops look more like photos")
        p.add_argument("--refs", default="", help="train: real photo + hand-made DNA pairs from prep_refs.py")
        p.add_argument("--ref-frac", type=float, default=0.15, help="train: share of each batch that is real photos")
        p.add_argument("--init", default="", help="train: start from this saved model instead of from scratch")
        p.add_argument("--lr-head", type=float, default=1e-3)
        p.add_argument("--name", default="", help="train: name the outputs model_<name>.pt etc. instead")
        p.add_argument("--genes-from", default="",
                       help="train: use this saved model's settings and button lists (not its weights), so the "
                            "new model matches the bridge - e.g. results\\model_male.pt")
        p.add_argument("--rand-aug", action="store_true",
                       help="train (with --photo-aug): also random lenses/camera distance, backgrounds, hard side "
                            "light, exposure and white balance, low resolution")
        p.add_argument("--tilt-data", default="", help="train: folder with the tilted views (autoshot.py tilt + prepare-tilt)")
        p.add_argument("--tilt-p", type=float, default=0.67, help="train: share of draws that use a tilted front view")
        p.add_argument("--hair-aug", type=float, default=0.0,
                       help="train (with --photo-aug): share of crops with loose hair over the face sides/jaw (e.g. 0.5)")
        p.add_argument("--refiner", default="", help="train: refine.py model that makes game crops look like photos")
        p.add_argument("--refine-p", type=float, default=0.6, help="train: share of game crops sent through it")
        p.add_argument("--rank-refs", default="",
                       help="train: reference people (prep_refs.py file): their hand-made DNA used only for the ORDER "
                            "of people on each slider")
        p.add_argument("--freeze-epochs", type=int, default=0,
                       help="train: keep the backbone frozen for this many epochs first (LP-FT; = --epochs: frozen)")
        p.add_argument("--consistency", default="",
                       help="train: photo sets (prep_photosets.py): photos of the same person must read alike")
        p.add_argument("--cons-k", type=int, default=8, help="train: people per batch for the consistency")
        p.add_argument("--cons-w", type=float, default=0.5, help="train: weight of the consistency loss")
        p.add_argument("--rank-sex", default="", choices=["", "female", "male"],
                       help="train: only the reference people of this sex (their ids start f / extra_f, m / extra_m)")
        p.add_argument("--rank-k", type=int, default=12, help="train: reference people per batch for the ranking")
        p.add_argument("--rank-w", type=float, default=0.5, help="train: weight of the ranking loss")
        p.add_argument("--hold-out-refs", action="store_true",
                       help="train: leave out faces built on any reference person in refs.csv (for fair tests)")
    a = ap.parse_args()
    {"check": per_folder(check), "prepare": per_folder(prepare), "train": train, "prepare-tilt": prepare_tilt}[a.cmd](a)


if __name__ == "__main__":
    main()
