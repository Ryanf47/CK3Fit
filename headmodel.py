"""The CK3 head as a function of DNA: slider positions -> head vertices -> the 478 face points.

Built on the replica code (replica.Head reads the game's head mesh, blendshapes and slider
animations from your local game files; dna.attributes turns DNA into blend/bone amounts).

FaceModel.points(pos)      exact: builds the head, then reads the bound face points off it.
FaceModel.fast(pvec)       quick additive approximation from per-slider response tables, used
                           inside the solver; the solver corrects it with exact builds.
"""
import hashlib
import json
import os
import numpy as np

import dnatext

HERE = (os.environ.get("CK3FIT_HOME") or os.path.dirname(os.path.abspath(__file__)))

# Slider groups the fitter works on. Body, neck, age, expression and special genes stay as in
# the template.
FIT_GROUPS = {"face", "eyes", "mouth", "nose", "head_neck", "ears"}
NEVER_FIT = {"gene_neck_length", "gene_neck_width", "gene_mouth_open", "gene_bs_head_asymmetry_1",
             "gene_bs_ears_fantasy", "gene_no_portrait", "gene_height", "expression_other",
             "gene_bs_mouth_glamour_lips"}
GRID = (0.0, 32.0, 64.0, 96.0, 127.5, 160.0, 192.0, 224.0, 255.0)


def shape_genes(survey, template_text):
    """Genes in the template that change the head's shape (have a mesh or animation file)."""
    mg, at = survey["morph_genes"], survey["attributes"]
    out = []
    for g in dnatext.genes(template_text):
        G = mg.get(g)
        if not G or g in NEVER_FIT or G.get("group") not in FIT_GROUPS:
            continue
        files = False
        for tb in G["templates"].values():
            if isinstance(tb["female"], list):
                for s in tb["female"]:
                    a = s["attribute"]
                    if a in at and at[a].get("file"):
                        files = True
        if files:
            out.append(g)
    return out


def load_real(settings_survey=None):
    """(survey, head, attrs_fn) from the replica code and your game files."""
    import dna as dnam
    from replica import Head
    path = settings_survey or os.path.join(HERE, "survey.json")
    survey = json.load(open(path))
    head = Head(survey)
    presets = dnam.load_age_presets(survey["gene_files"])

    def attrs_fn(d):
        return dnam.attributes(d, survey, presets=presets)[0]
    return survey, head, attrs_fn


class FaceModel:
    def __init__(self, head, attrs_fn, template_text, genes, bind, cache_dir=None, log=print):
        """bind: {"tri": (n,), "bary": (n,3), "weight": (n,)} as written by bind.py."""
        self.head, self.attrs_fn, self.log = head, attrs_fn, log
        self.template = template_text
        self.dna0 = dnatext.genes(template_text)
        self.genes = list(genes)
        self.p0 = np.array([dnatext.position(g, *self.dna0[g]) for g in self.genes])
        self.tri_ix = np.asarray(bind["tri"], int)
        self.bary = np.asarray(bind["bary"], float)
        self.weight = np.asarray(bind["weight"], float)
        self.corners = np.asarray(head.tri)[self.tri_ix]          # (n, 3) vertex ids
        self.builds = 0
        self._tables(cache_dir)

    # ---------- exact ----------
    def pos_dict(self, pvec):
        return {g: float(v) for g, v in zip(self.genes, pvec)}

    def dna(self, pvec):
        return dnatext.with_positions(self.dna0, self.pos_dict(pvec))

    def verts(self, pvec):
        self.builds += 1
        return self.head.build(self.attrs_fn(self.dna(pvec)))

    def points_from_verts(self, V):
        return np.einsum("nk,nkd->nd", self.bary, V[self.corners])

    def points(self, pvec):
        return self.points_from_verts(self.verts(pvec))

    # ---------- per-slider response tables ----------
    def _tables(self, cache_dir):
        key = hashlib.sha1((json.dumps(self.genes) + self.template +
                            self.tri_ix.tobytes().hex()[:2000] + self.bary.tobytes().hex()[:2000] +
                            str(len(self.head.P))).encode()).hexdigest()[:16]
        path = os.path.join(cache_dir, f"tables_{key}.npz") if cache_dir else None
        if path and os.path.exists(path):
            z = np.load(path, allow_pickle=True)
            self.base, self.grids, self.tabs = z["base"], list(z["grids"]), list(z["tabs"])
            self.vbase, self.vtabs = z["vbase"], list(z["vtabs"])
            return
        self.log(f"Measuring how each of {len(self.genes)} sliders moves the face (once, cached) ...")
        V0 = self.verts(self.p0)
        self.vbase = V0
        self.base = self.points_from_verts(V0)
        self.grids, self.tabs, self.vtabs = [], [], []
        for i, g in enumerate(self.genes):
            grid = sorted(set(GRID) | {float(self.p0[i])})
            tab, vtab = [], []
            for p in grid:
                pv = self.p0.copy(); pv[i] = p
                V = V0 if p == self.p0[i] else self.verts(pv)
                tab.append(self.points_from_verts(V) - self.base)
                vtab.append((V - V0).astype(np.float32))
            self.grids.append(np.array(grid)); self.tabs.append(np.array(tab))
            self.vtabs.append(np.array(vtab))
        if path:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez_compressed(path, base=self.base, vbase=self.vbase,
                                grids=np.array(self.grids, dtype=object),
                                tabs=np.array(self.tabs, dtype=object),
                                vtabs=np.array(self.vtabs, dtype=object))

    def _seg(self, i, p):
        grid = self.grids[i]
        k = int(np.clip(np.searchsorted(grid, p) - 1, 0, len(grid) - 2))
        u = (p - grid[k]) / (grid[k + 1] - grid[k])
        return k, u, grid[k + 1] - grid[k]

    def fast(self, pvec, offset=None):
        """Additive approximation of points(pvec); also returns d points / d p_i per slider."""
        X = self.base.copy()
        D = np.empty((len(self.genes),) + X.shape)
        for i, p in enumerate(pvec):
            k, u, w = self._seg(i, p)
            t = self.tabs[i]
            X += t[k] * (1 - u) + t[k + 1] * u
            D[i] = (t[k + 1] - t[k]) / w
        if offset is not None:
            X = X + offset
        return X, D

    def fast_verts(self, pvec):
        V = self.vbase.astype(np.float64).copy()
        for i, p in enumerate(pvec):
            k, u, _ = self._seg(i, p)
            t = self.vtabs[i]
            V += t[k] * (1 - u) + t[k + 1] * u
        return V

    def reach(self):
        """How far each slider can move the face points over its full range (face-size units)."""
        size = np.sqrt(((self.base - self.base.mean(0)) ** 2).sum(1).mean())
        return {g: float(np.abs(t - t[:1]).max() / size) for g, t in zip(self.genes, self.tabs)}
