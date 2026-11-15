"""Read and write Ruler Designer DNA text, and convert genes to one 0-255 "slider position".

Slider positions follow CK3Face's convention:
- Two-sided genes (gene_bs_... / face_detail_... with a _neg/_pos style template pair):
  0 = full neg side, 127.5 = neutral, 255 = full pos side.
- Every other gene: the DNA value itself (0-255); its template name is left unchanged.
"""
import re

GENE_RE = re.compile(
    r'(\b(?P<gene>\w+)=\{\s*"(?P<t1>[^"]+)"\s+)(?P<v1>\d+)(\s+"(?P<t2>[^"]+)"\s+)(?P<v2>\d+)(\s*\})')
SIDE_PAIRS = (("_neg_02", "_pos_02"), ("_neg", "_pos"), ("_min", "_max"))


def genes(text):
    """{gene: (template, value)} using the dominant (first) copy, as the Ruler Designer shows it."""
    return {m["gene"]: (m["t1"], int(m["v1"])) for m in GENE_RE.finditer(text)}


def sides(gene, tmpl):
    """(stem, low_suffix, high_suffix) for two-sided genes, else None."""
    if gene.startswith(("gene_bs_", "face_detail_")):
        for lo, hi in SIDE_PAIRS:
            if tmpl.endswith(lo):
                return tmpl[:-len(lo)], lo, hi
            if tmpl.endswith(hi):
                return tmpl[:-len(hi)], lo, hi
    return None


def position(gene, tmpl, v):
    s = sides(gene, tmpl)
    if s is None:
        return float(v)
    return 127.5 - v / 2 if tmpl.endswith(s[1]) else 127.5 + v / 2


def gene_value(gene, tmpl, pos):
    """slider position -> (template, value) for this gene, given its current template."""
    s = sides(gene, tmpl)
    pos = max(0.0, min(255.0, float(pos)))
    if s is None:
        return tmpl, int(round(pos))
    stem, lo, hi = s
    x = (pos - 127.5) * 2
    if x < 0:
        return stem + lo, int(round(min(255, -x)))
    return stem + hi, int(round(min(255, x)))


def positions(text, only=None):
    g = genes(text)
    return {k: position(k, t, v) for k, (t, v) in g.items() if only is None or k in only}


def set_positions(text, pos):
    """Return text with each gene in pos set (both copies) to that slider position.
    Everything else in the DNA stays exactly as it was."""
    cur = genes(text)

    def sub(m):
        gname = m.group("gene")
        if gname not in pos:
            return m.group(0)
        old_t, old_v = cur[gname]
        tmpl, val = gene_value(gname, old_t, pos[gname])
        if val == 0 and old_v == 0:
            tmpl = old_t                      # neutral either way: keep the side it was on
        if (tmpl, val) == (old_t, old_v):
            return m.group(0)                 # unchanged: leave the line exactly as it was
        return f'{gname}={{ "{tmpl}" {val} "{tmpl}" {val} }}'

    return GENE_RE.sub(sub, text)


def as_dna(text_or_dict):
    """{gene: (template, value)} from DNA text, or passed through."""
    return genes(text_or_dict) if isinstance(text_or_dict, str) else text_or_dict


def with_positions(dna, pos):
    """{gene: (template, value)} with positions applied (no text involved)."""
    out = dict(dna)
    for g, p in pos.items():
        if g in out:
            out[g] = gene_value(g, out[g][0], p)
    return out


def read(path):
    with open(path, encoding="utf-8-sig") as f:
        return f.read()
