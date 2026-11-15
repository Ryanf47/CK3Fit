"""Collect random game faces with their DNA, for training the reverse network (screenshot -> DNA).

Two kinds of face, mixed:
  WIDE (20%): every ordinary face slider set anywhere in 0-255 (the 'extra' sliders on 30% of them), every
      discrete face option (eyebrows, complexion, face details, wrinkles) random, colours random.
      These make sure every slider value and every option is seen. Many look odd; that's the point.
  ANCHORED (80%): start from the sex's template or a reference DNA (from refs.csv, minus a held-out set,
      by default the characters in exam\\); move 15-35 ordinary sliders by amounts matched to how much the
      references vary (strength 0.5x-1.2x); 'extra' sliders (ones at least 60% of references leave at the
      template value: swollen nose, flared nostrils, lips forward ...) move only 8% of the time; discrete options
      re-rolled 30% of the time, mostly at low strength; colours fully random 70% of the time, else nudged.
  Both kinds: face decals, head fat, skin and eye colour saturation, eyelashes and eye type re-rolled (always on
      wide faces, 60% on anchored), with strengths kept inside the range the game's own Randomize faces use.
      Age-related settings are left alone (every Ruler Designer character is 25).
Extra starting faces: put DNAs from Randomize as .txt files in anchors_female\\ or
anchors_male\\. Starting faces: template 5%, same-sex human-made references 20%, these Randomize faces 75%.
Anchored faces also get 0-6 ordinary sliders set anywhere in 0-255, so every slider reaches its ends on
otherwise natural faces, not only on the wide ones. They add
options to the pools, but slider spreads stay measured on the human-made references (Randomize stays narrow).
Hairstyle: drawn evenly from every female (or male) hairstyle seen in the starting faces, minus any listed in
hair_block_<sex>.txt (headwear that hides the forehead). 'python collect_noise.py hairs --sex female' writes one
DNA per hairstyle to screenshot and choose from. Hair colour and clothes come from the starting DNA.
Beards (men): removed from every face by default, so the jaw and chin are always visible; --beards 0.2 keeps
the starting beard on 20% of faces.
Every DNA is written by this script, so it is known exactly. No Randomize button, no ethnicity list.

Discrete options are pooled from all 71 reference DNAs plus the templates, so only options someone has used
are drawn. (A full list would need the game's gene definitions.)

    python collect_noise.py plan --sex female            (coverage report + 5 sample DNAs; no game needed)
    python collect_noise.py run --sex female --n 20      (try 20 first)
    python collect_noise.py run --sex female --n 10000

Set the Ruler Designer to the right sex first, default view (Head and Neck panel, both faces showing).
Output: random_faces\\noise_<sex>\\NNNNN.txt + NNNNN.jpg, plus log.csv and meta.csv (which kind, which anchor).
Running again carries on where it stopped. Mouse to the top-left corner or Ctrl+C stops.
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np

HERE = (os.environ.get("CK3FIT_HOME") or os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import dnatext      # noqa: E402
import headmodel    # noqa: E402

TEMPLATES = {"female": "template_hair17.txt", "male": "template_male.txt"}
WIDE_SHARE = 0.20
TEMPLATE_SHARE = 0.05       # starting face: template 5%, human-made references 20%, Randomize faces 75%
REF_SHARE = 0.20
EXTREME_RANGE = (0, 7)      # anchored faces: 0-6 ordinary sliders also set anywhere in 0-255 (one strong feature)
WIDEN = 1.25                # anchored slider spread = this x the reference spread ...
MIN_SD, MAX_SD = 20.0, 60.0  # ... kept within these
MOVE_RANGE = (15, 36)       # anchored faces: how many ordinary sliders move (15-35)
STRENGTH = (0.5, 1.2)
RARE_SHARE = 0.60           # a slider is "extra" if this share of references leave it at the template value
RARE_MOVE_ANCHORED = 0.08   # chance an extra slider moves on an anchored face
RARE_MOVE_WIDE = 0.30       # chance an extra slider goes anywhere on a wide face
DISCRETE_REROLL = 0.3       # anchored faces: chance each discrete option is re-rolled (mostly low strength)
COLOUR_SD = {"hair_color": 35.0, "eye_color": 35.0, "skin_color": 20.0}
COLOUR_UNIFORM = 0.70       # anchored faces: chance colours are fully random (so colour says nothing about shape)

DISCRETE = [
    "gene_eyebrows_shape", "gene_eyebrows_fullness",
    "face_detail_cheek_def", "face_detail_cheek_fat", "face_detail_chin_cleft", "face_detail_chin_def",
    "face_detail_eye_lower_lid_def", "face_detail_eye_upper_lid_def", "face_detail_eye_socket",
    "face_detail_nasolabial", "face_detail_nose_ridge_def", "face_detail_nose_tip_def", "face_detail_temple_def",
    "expression_brow_wrinkles", "expression_eye_wrinkles", "expression_forehead_wrinkles", "expression_other",
]
# other settings that change how the face looks: option and strength both drawn evenly (not mostly subtle)
# (not gene_age / hair_aging: every Ruler Designer character is 25, where aging barely shows)
APPEAR = [
    "complexion",                                                    # skin type button + Lip Color (its strength)
    "gene_face_dacals", "gene_bs_body_type",                         # face decals, head fat
    "skin_color_saturation", "eye_color_saturation",
    "eyelashes_accessory", "eye_accessory",
]
APPEAR_REROLL = 0.6         # anchored faces: chance each of these is re-rolled (wide faces: always)
HEAD_FAT = "gene_bs_body_type"
HEAD_FAT_REROLL = 0.35      # head fat: re-rolled on 35% of faces (wide or not), otherwise the starting face's
HEAD_FAT_SD = 6.0           # ... and then mostly within a few steps of neutral (127)
HEAD_FAT_EXTREME = 0.10     # ... except 10% of re-rolls, which go anywhere in the game's own range
# face sliders that also have option buttons in the Ruler Designer (Ear Bend: Lower/Upper/Both, Head Shapes:
# Round/Archetype 01-10, Nose Profile: Straight/Beak/Hawk..., etc.). The slider code sets their value; the button
# (which shape the value applies to) is re-rolled here, keeping the value.
CHOICE = [
    "gene_bs_ear_bend", "gene_bs_ear_lobe",
    "gene_bs_eyebrow_straight", "gene_eyebrow_inner_width",
    "gene_bs_head_asymmetry_1", "gene_bs_head_lower_height", "gene_bs_head_shape",
    "gene_bs_eye_fold_shape", "gene_bs_eye_height_inside", "gene_bs_eye_height_outisde", "gene_bs_eye_outer_width",
    "gene_bs_eye_shape", "gene_bs_eye_upper_lid_size", "gene_bs_eye_lower_lid_size",
    "gene_bs_nose_central_width", "gene_bs_nose_flared_nostril", "gene_bs_nose_profile",
    "gene_bs_nose_septum_height", "gene_bs_nose_septum_width", "gene_bs_nose_swollen",
    "gene_bs_mouth_center_curve", "gene_bs_mouth_glamour_lips", "gene_bs_mouth_lower_lip_forward",
    "gene_bs_mouth_lower_lip_profile", "gene_bs_mouth_upper_lip_forward",
]
CHOICE_REROLL = 0.5         # anchored faces: chance each button is re-rolled (wide faces: always)
GENE_RE = re.compile(r'\b(%s)=\{\s*"([^"]+)"\s+(\d+)\s+"([^"]+)"\s+(\d+)\s*\}' % "|".join(DISCRETE + APPEAR + CHOICE))
# never drawn: stubble on a woman, a blank face, no eyebrows/eyelashes, blind/bloodshot/missing eyes
BAD_OPTIONS = {"body_hair_dense_unisex", "complexion_no_face", "no_eyebrows", "no_eyelashes",
               "bloodshot_eyes", "blind_eyes", "no_eyes"}
GAME_OPTIONS = "game_options.json"   # every visible option per gene, read from the game files (local only)
COLOUR_RE = re.compile(r"\b(hair_color|skin_color|eye_color)=\{\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*\}")
HAIR_RE = re.compile(r"(custom_hair=)(\S+)")
BEARD_RE = re.compile(r"(custom_beards=)(\S+)")


def shave(text):
    """no beard at all (men only; women's DNA has no beard to remove): the beard gene set to 'no_beard' and
    custom_beards=male_empty, added after custom_hair if the face has no custom_beards line"""
    if not re.search(r"\btype=male\b", text):
        return text
    text = re.sub(r'\bbeards=\{[^}]*\}', 'beards={ "no_beard" 0 "no_beard" 0 }', text)
    if BEARD_RE.search(text):
        return BEARD_RE.sub(lambda m: m.group(1) + "male_empty", text)
    return re.sub(r"(custom_hair=\S+)", lambda m: m.group(1) + "\n\t\t\tcustom_beards=male_empty", text, count=1)


def hair_block(sex):
    """Hairstyles never to use (headwear that hides the forehead etc.): one name per line in hair_block_<sex>.txt."""
    p = os.path.join(HERE, f"hair_block_{sex}.txt")
    if not os.path.exists(p):
        return set()
    return {l.strip() for l in open(p, encoding="utf-8") if l.strip() and not l.startswith("#")}


def find_file(p):
    for cand in (p, os.path.join(HERE, p), os.path.join(HERE, "agot_dna2", p)):
        if os.path.isfile(cand):
            return cand
    return None


def load_refs():
    rows = list(csv.DictReader(open(os.path.join(HERE, "refs.csv"), newline="", encoding="utf-8-sig")))
    out, missing = {}, []
    for r in rows:
        path = find_file(r["dna"].strip())
        if path:
            out[r["key"].strip()] = (r.get("sex", "").strip().lower(), dnatext.read(path))
        else:
            missing.append(r["key"])
    if missing:
        print(f"  (couldn't find the DNA file for: {', '.join(missing)})")
    return out


def load_extra(sex):
    """Randomize-per-ethnicity DNAs saved by hand in anchors_<sex>\\ (any .txt names)."""
    d = os.path.join(HERE, f"anchors_{sex}")
    if not os.path.isdir(d):
        return {}
    return {"eth_" + f[:-4]: dnatext.read(os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.lower().endswith(".txt")}


def default_holdout():
    d = os.path.join(HERE, "exam")
    return sorted(n for n in os.listdir(d) if os.path.isdir(os.path.join(d, n))) if os.path.isdir(d) else []


def option_pools(texts):
    pools = defaultdict(set)
    for t in texts:
        for m in GENE_RE.finditer(t):
            pools[m.group(1)].update([m.group(2), m.group(4)])
            if m.group(1) in APPEAR:                     # strengths stay within what the game itself uses
                lo, hi = RANGES.get(m.group(1), (255, 0))
                v = int(m.group(3))
                RANGES[m.group(1)] = (min(lo, v), max(hi, v))
    p = os.path.join(HERE, GAME_OPTIONS)
    if os.path.exists(p):                                # add options no starting face happened to use
        for g, opts in json.load(open(p)).items():
            if g in pools:
                pools[g].update(opts)
    return {g: sorted(v - BAD_OPTIONS) for g, v in pools.items()}


RANGES = {}


def setup(a):
    survey = json.load(open(os.path.join(HERE, "survey.json")))
    template = dnatext.read(os.path.join(HERE, TEMPLATES[a.sex]))
    genes = headmodel.shape_genes(survey, template)
    refs = load_refs()
    holdout = [k.strip() for k in a.holdout.split(",") if k.strip()] if a.holdout is not None else default_holdout()
    same_sex = {k: t for k, (s, t) in refs.items() if s == a.sex}
    extra = load_extra(a.sex)
    print(f"  {len(extra)} ethnicity anchors from anchors_{a.sex}\\")
    # spreads and 'extra' sliders come from the human-made references only: Randomize keeps most sliders in a
    # narrow middle band, and counting 1,000 of those would shrink every spread
    pos = [dnatext.positions(t, set(genes)) for t in same_sex.values()]
    same_sex.update(extra)
    anchors = {k: t for k, t in same_sex.items() if k not in holdout}
    tpos = dnatext.positions(template, set(genes))
    sd, rare = {}, set()
    for g in genes:
        vals = [p[g] for p in pos if g in p]
        s = float(np.std(vals)) if len(vals) >= 3 else MAX_SD / 2
        sd[g] = float(np.clip(s * WIDEN, MIN_SD, MAX_SD))
        if vals and g in tpos and np.mean([abs(v - tpos[g]) <= 3 for v in vals]) >= RARE_SHARE:
            rare.add(g)
    a.rare = rare
    a.shape = frozenset(genes)
    texts = list(same_sex.values()) + [template]       # same sex only, so no male-only options on women
    pools = option_pools(texts)
    blocked = hair_block(a.sex)
    seen_hair = {m.group(2) for t in list(same_sex.values()) + [template] for m in HAIR_RE.finditer(t)}
    a.all_hairs = sorted(seen_hair)
    allow = os.path.join(HERE, f"hair_list_{a.sex}.txt")
    if os.path.exists(allow):                          # your own list of allowed hairstyles wins
        a.hairs = sorted({l.strip() for l in open(allow, encoding="utf-8") if l.strip() and not l.startswith("#")} - blocked)
        print(f"  {len(a.hairs)} hairstyles in use, from hair_list_{a.sex}.txt")
    else:
        a.hairs = sorted(seen_hair - blocked)
        print(f"  {len(a.hairs)} hairstyles in use ({len(seen_hair & blocked)} blocked by hair_block_{a.sex}.txt)")
    return template, genes, anchors, holdout, sd, pools


def reroll_discrete(text, rng, pools, chance, wide, shape=frozenset()):
    def rep(m):
        g = m.group(1)
        if g in CHOICE and g in pools:                  # option button: new shape, same slider value
            if rng.random() >= (1.0 if wide else CHOICE_REROLL):
                return m.group(0)
            name = pools[g][rng.integers(len(pools[g]))]
            return f'{g}={{ "{name}" {m.group(3)} "{name}" {m.group(5)} }}'
        if g in shape or g not in pools:                # face sliders are handled by the slider code
            return m.group(0)
        appear = g in APPEAR
        if g == HEAD_FAT:
            c = HEAD_FAT_REROLL
        else:
            c = 1.0 if wide else (APPEAR_REROLL if appear else chance)
        if rng.random() >= c:
            return m.group(0)
        name = pools[g][rng.integers(len(pools[g]))]
        if g == HEAD_FAT:                               # mostly near neutral, an occasional extreme
            lo, hi = RANGES.get(g, (102, 165))
            if rng.random() < HEAD_FAT_EXTREME:
                v = int(rng.integers(lo, hi + 1))
            else:
                v = int(np.clip(round(rng.normal(127, HEAD_FAT_SD)), lo, hi))
        elif appear:
            lo, hi = RANGES.get(g, (0, 255))
            v = int(rng.integers(lo, hi + 1))
        elif wide:
            v = int(rng.integers(0, 256))
        else:
            v = int(rng.beta(1.3, 3.0) * 255)           # anchored face details: mostly subtle
        return f'{g}={{ "{name}" {v} "{name}" {v} }}'
    return GENE_RE.sub(rep, text)


def set_colours(text, rng, uniform):
    def rep(m):
        x, y = int(m.group(2)), int(m.group(3))
        if uniform:
            x, y = (int(v) for v in rng.integers(0, 256, 2))
        else:
            sdc = COLOUR_SD[m.group(1)]
            x = int(np.clip(round(x + rng.normal(0, sdc)), 0, 255))
            y = int(np.clip(round(y + rng.normal(0, sdc)), 0, 255))
        return f"{m.group(1)}={{ {x} {y} {x} {y} }}"
    return COLOUR_RE.sub(rep, text)


def make_face(i, a, template, genes, anchors, sd, pools):
    rng = np.random.default_rng([a.seed, i])          # same face for the same index on every run
    refs = sorted(k for k in anchors if not k.startswith("eth_"))
    eth = sorted(k for k in anchors if k.startswith("eth_"))
    wide = rng.random() < WIDE_SHARE
    u = rng.random()                                   # template / references / Randomize faces, kept in fixed shares
    if u < TEMPLATE_SHARE or not (refs or eth):
        anchor, text = "template", template
    else:
        pool = refs if (u < TEMPLATE_SHARE + REF_SHARE and refs) or not eth else eth
        anchor = pool[rng.integers(len(pool))]
        text = anchors[anchor]
    pos = dnatext.positions(text, set(genes))
    rare = a.rare
    new = dict(pos)
    if wide:
        scale = 0.0
        for g in pos:
            if g not in rare or rng.random() < RARE_MOVE_WIDE:
                new[g] = float(rng.uniform(0, 255))
    else:
        scale = float(rng.uniform(*STRENGTH))
        ordinary = [g for g in sorted(pos) if g not in rare]
        k = min(len(ordinary), int(rng.integers(*MOVE_RANGE)))
        moving = set(rng.choice(ordinary, size=k, replace=False)) if k else set()
        moving |= {g for g in sorted(pos) if g in rare and rng.random() < RARE_MOVE_ANCHORED}
        for g in moving:
            new[g] = float(np.clip(pos[g] + rng.normal(0, sd[g] * scale), 0, 255))
        n_ext = min(len(ordinary), int(rng.integers(*EXTREME_RANGE)))
        for g in (rng.choice(ordinary, size=n_ext, replace=False) if n_ext else []):
            new[g] = float(rng.uniform(0, 255))
    text = dnatext.set_positions(text, new)
    text = reroll_discrete(text, rng, pools, 1.0 if wide else DISCRETE_REROLL, wide, a.shape)
    text = set_colours(text, rng, wide or rng.random() < COLOUR_UNIFORM)
    if a.hairs:                                        # every allowed hairstyle equally often, whatever the anchor
        h = a.hairs[rng.integers(len(a.hairs))]
        text = HAIR_RE.sub(lambda m: m.group(1) + h, text)
    if a.sex == "male" and getattr(a, "beard_list", None) and rng.random() < a.beards:
        text = set_beard(text, a.beard_list[rng.integers(len(a.beard_list))])   # a random beard (the shape under it
    else:                                              # is still known exactly from the DNA)
        text = shave(text)
    return text, {"kind": "wide" if wide else "anchored", "anchor": anchor, "scale": round(scale, 3)}


def plan(a):
    template, genes, anchors, holdout, sd, pools = setup(a)
    print(f"{a.sex}: {len(genes)} face sliders, {len(anchors)} reference anchors + the template")
    print(f"Held out (never used as anchors): {', '.join(holdout) if holdout else 'none'}")
    print(f"'Extra' sliders, mostly left alone ({len(a.rare)}): " + ", ".join(sorted(a.rare)))
    print(f"Discrete options pooled: " + ", ".join(f"{g.replace('gene_', '')} {len(v)}" for g, v in pools.items()))

    n = a.check
    print(f"\nSimulating {n} faces to check coverage (scaled to 10,000 below) ...")
    bins = np.zeros((len(genes), 8))
    opts = defaultdict(Counter)
    for i in range(n):
        text, _ = make_face(i, a, template, genes, anchors, sd, pools)
        p = dnatext.positions(text, set(genes))
        for j, g in enumerate(genes):
            if g in p:
                bins[j, min(7, int(p[g] // 32))] += 1
        for m in GENE_RE.finditer(text):
            opts[m.group(1)][m.group(2)] += 1
    per10k = bins * 10000 / n
    print("Slider range split into 8 parts (0-31, 32-63, ... 224-255). Faces per part, per 10,000:")
    worst = np.argsort(per10k.min(1))
    for j in list(worst[:10]):
        print(f"  {genes[j]:42s} " + " ".join(f"{v:5.0f}" for v in per10k[j]))
    print(f"  Every slider, every part: at least {per10k.min():.0f} faces per 10,000 (the 10 worst sliders are above).")
    rare = [(g, o, c * 10000 / n) for g, cnt in opts.items() for o, c in cnt.items()]
    rare.sort(key=lambda r: r[2])
    print("Rarest discrete options, faces per 10,000: " +
          ", ".join(f"{o} {c:.0f}" for _, o, c in rare[:6]))

    out = os.path.join(HERE, "random_faces", f"plan_{a.sex}")
    os.makedirs(out, exist_ok=True)
    print(f"\nSample DNAs in {out}:")
    for i in range(12):
        text, m = make_face(i, a, template, genes, anchors, sd, pools)
        open(os.path.join(out, f"{i:05d}.txt"), "w").write(text)
        print(f"  {i:05d}: {m['kind']}, anchor {m['anchor']}" + (f", strength {m['scale']}" if m['kind'] == 'anchored' else ""))


def hairs(a):
    """One DNA per hairstyle (the template's face) into random_faces\\hairs_<sex>, to screenshot and pick blocks."""
    template, *_ = setup(a)
    out = os.path.join(HERE, "random_faces", "screen_male", "hairs") if a.sex == "male" else \
        os.path.join(HERE, "random_faces", f"hairs_{a.sex}")
    os.makedirs(out, exist_ok=True)
    names = read_names(a.names) if a.names else a.all_hairs
    base = screen_base(a, template)
    for h in names:
        t = HAIR_RE.sub(lambda m: m.group(1) + h, base)
        open(os.path.join(out, h + ".txt"), "w").write(set_beard(t, "male_empty") if a.sex == "male" else t)
    print(f"{len(names)} hairstyle DNAs in {out}. Screenshot them with:  python autoshot.py folder {out}")
    print(f"Then list the ones to never use (one name per line, the file name without .txt) in hair_block_{a.sex}.txt")


def read_names(path):
    path = path if os.path.isabs(path) else os.path.join(HERE, path)
    return [l.strip() for l in open(path, encoding="utf-8") if l.strip() and not l.startswith("#")]


def screen_base(a, template):
    """the face to show hairstyles / beards on: --base (a DNA file), else the template"""
    if not a.base:
        return template
    p = a.base if os.path.isabs(a.base) else os.path.join(HERE, a.base)
    return open(p, encoding="utf-8").read()


def set_beard(text, b):
    """this beard, whatever the face had: the beard gene opened up to every beard, and custom_beards set (added
    after custom_hair if the face has none)"""
    text = re.sub(r'\bbeards=\{[^}]*\}', 'beards={ "all_beards" 128 "all_beards" 128 }', text)
    if re.search(r"custom_beards=\S+", text):
        return re.sub(r"(custom_beards=)(\S+)", lambda m: m.group(1) + b, text)
    return re.sub(r"(custom_hair=\S+)", lambda m: m.group(1) + "\n\t\t\tcustom_beards=" + b, text, count=1)


def beards(a):
    """One DNA per beard (on --base or the template) into random_faces\\screen_male\\beards, to screenshot."""
    template, *_ = setup(a)
    if a.sex != "male":
        sys.exit("Beards are for --sex male.")
    base = screen_base(a, template)
    names = read_names(a.names or "beard_all_male.txt")
    out = os.path.join(HERE, "random_faces", "screen_male", "beards")
    os.makedirs(out, exist_ok=True)
    for b in names:
        open(os.path.join(out, b + ".txt"), "w").write(set_beard(base, b))
    print(f"{len(names)} beard DNAs in {out}. Screenshot them with:  python autoshot.py folder {out}")


def run(a):
    if os.name != "nt":
        sys.exit("This runs on Windows, next to the game.")
    import autoshot
    s = autoshot.settings()
    if "paste_button" not in s or "yes_button" not in s:
        sys.exit("Run 'python autoshot.py calibrate' first (both steps).")
    template, genes, anchors, holdout, sd, pools = setup(a)
    out = os.path.join(HERE, "random_faces", a.out or f"noise_{a.sex}")
    os.makedirs(out, exist_ok=True)
    todo = [i for i in range(a.n) if not os.path.exists(os.path.join(out, f"{i:05d}.jpg"))]
    print(f"{a.sex}: {len(anchors)} anchors + template, holdout {len(holdout)}; {len(todo)} of {a.n} faces to shoot.")
    print("Ruler Designer set to " + a.sex.upper() + "? Switch to the game now (Alt+Tab); starting in 10 s ...")
    time.sleep(10)
    new_log = not os.path.exists(os.path.join(out, "log.csv"))
    logf = open(os.path.join(out, "log.csv"), "a", newline="")
    metaf = open(os.path.join(out, "meta.csv"), "a", newline="")
    log = csv.writer(logf); meta = csv.writer(metaf)
    if new_log:
        log.writerow(["shot", "result", "seconds", "change", "box"])
        meta.writerow(["shot", "kind", "anchor", "scale"])
    times, bad, t_start = [], 0, time.time()
    try:
        for n, i in enumerate(todo):
            name = f"{i:05d}"
            text, m = make_face(i, a, template, genes, anchors, sd, pools)
            open(os.path.join(out, name + ".txt"), "w").write(text)
            r = autoshot.shoot(text, os.path.join(out, name + ".jpg"), s)
            if r["result"] != "ok":
                bad += 1
                jpg = os.path.join(out, name + ".jpg")
                if os.path.exists(jpg):
                    os.remove(jpg)          # not settled: drop it, a later run retries this index
            log.writerow([name, r["result"], r.get("seconds", ""), r.get("change", ""), r.get("box", "")])
            meta.writerow([name, m["kind"], m["anchor"], m["scale"]])
            logf.flush(); metaf.flush()
            times.append(r.get("seconds", 0))
            if n < 20 or n % 100 == 0:
                left = (len(todo) - n - 1) * np.mean(times) / 3600
                print(f"  {name}: {r['result']}, {r['seconds']} s  ({m['kind']}, {m['anchor']}; ~{left:.1f} h left)")
    except (autoshot.Stop, KeyboardInterrupt):
        print("Stopped. Run the same command again to carry on.")
    logf.close(); metaf.close()
    if times:
        print(f"\n{len(times)} shot, {bad} not settled (dropped, retried next run); "
              f"{np.mean(times):.2f} s per face, {(time.time() - t_start) / 3600:.2f} h this run. Folder: {out}")
    import winsound
    winsound.Beep(660, 500)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run", "hairs", "beards"):
        p = sub.add_parser(name)
        p.add_argument("--sex", choices=["female", "male"], required=True)
        p.add_argument("--n", type=int, default=20)
        p.add_argument("--seed", type=int, default=1)
        p.add_argument("--beards", type=float, default=0.0,
                       help="men: share of faces given a random beard from beard_list_male.txt (default 0 = all "
                            "clean-shaven)")
        p.add_argument("--out", default="", help="run: folder inside random_faces (default noise_<sex>)")
        p.add_argument("--check", type=int, default=2000, help="plan: faces to simulate for the coverage report")
        p.add_argument("--holdout", default=None, help="comma-separated keys; default: the folders in exam\\")
        p.add_argument("--base", default=None, help="hairs / beards: show them on this DNA's face instead of the template")
        p.add_argument("--names", default=None, help="hairs / beards: a file listing the names to write (one per line)")
    a = ap.parse_args()
    a.beard_list = []
    if a.sex == "male" and a.beards > 0:
        bl = os.path.join(HERE, "beard_list_male.txt")
        if not os.path.exists(bl):
            sys.exit("beard_list_male.txt not found.")
        a.beard_list = [b for b in read_names(bl) if b != "male_empty"]
    {"plan": plan, "run": run, "hairs": hairs, "beards": beards}[a.cmd](a)


if __name__ == "__main__":
    main()
