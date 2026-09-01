#!/usr/bin/env python3
"""
svg2panel.py - convert an Inkscape panel SVG into a fab-ready DXF (+ 1:1 PDF print artwork).

Fixes the things Inkscape's built-in "Desktop Cutting Plotter (DXF R12)" export gets wrong
for front-panel work:

  * writes a real LAYER table (R12 export references layer names that don't exist)
  * declares millimetre units ($INSUNITS=4, $MEASUREMENT=1) and correct $EXTMIN/$EXTMAX
  * flattens Beziers to a tolerance you choose (default 5 um) instead of ~70 um
  * emits true CIRCLE entities for round holes instead of 17-sided polygons
  * emits ARC bulges for rounded-rectangle corners
  * emits genuinely CLOSED LWPOLYLINEs (no duplicated end vertex, closed flag set)
  * fixes contour winding: outer contours CCW, holes CW, so both even-odd and
    non-zero island detection give the same result
  * optionally adds solid HATCH so the silkscreen renders as filled ink, not outlines

Usage:
  python3 svg2panel.py in.svg out_prefix
      [--tol 0.005] [--layer-map silksceren=SILKSCREEN,edge-cuts=EDGE_CUTS]
      [--silk-layer SILKSCREEN] [--no-hatch] [--dxf-version AC1015]
"""
import argparse, math, re, sys
import numpy as np
from lxml import etree
from svgpathtools import parse_path, Path, Line, CubicBezier, QuadraticBezier, Arc

SVG = '{http://www.w3.org/2000/svg}'
INK = '{http://www.inkscape.org/namespaces/inkscape}'


# ---------------------------------------------------------------- transforms
def parse_transform(ts):
    M = np.eye(3)
    if not ts:
        return M
    for name, args in re.findall(r'(\w+)\s*\(([^)]*)\)', ts):
        a = [float(x) for x in re.split(r'[,\s]+', args.strip()) if x]
        if name == 'translate':
            m = np.array([[1, 0, a[0]], [0, 1, a[1] if len(a) > 1 else 0], [0, 0, 1]])
        elif name == 'scale':
            sx = a[0]; sy = a[1] if len(a) > 1 else a[0]
            m = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
        elif name == 'rotate':
            th = math.radians(a[0]); c, s = math.cos(th), math.sin(th)
            m = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            if len(a) == 3:
                t1 = np.array([[1, 0, a[1]], [0, 1, a[2]], [0, 0, 1]])
                t2 = np.array([[1, 0, -a[1]], [0, 1, -a[2]], [0, 0, 1]])
                m = t1 @ m @ t2
        elif name == 'matrix':
            m = np.array([[a[0], a[2], a[4]], [a[1], a[3], a[5]], [0, 0, 1]])
        elif name == 'skewX':
            m = np.array([[1, math.tan(math.radians(a[0])), 0], [0, 1, 0], [0, 0, 1]])
        elif name == 'skewY':
            m = np.array([[1, 0, 0], [math.tan(math.radians(a[0])), 1, 0], [0, 0, 1]])
        else:
            m = np.eye(3)
        M = M @ m
    return M


def apply(M, pts):
    P = np.c_[pts, np.ones(len(pts))]
    return (M @ P.T).T[:, :2]


# ---------------------------------------------------------------- shapes -> Path
def num(v, default=0.0):
    if v is None:
        return default
    m = re.match(r'\s*(-?[\d.eE+-]+)', str(v))
    return float(m.group(1)) if m else default


def element_to_path(el):
    tag = etree.QName(el).localname
    if tag == 'path':
        d = el.get('d')
        return parse_path(d) if d else None
    if tag == 'rect':
        x, y = num(el.get('x')), num(el.get('y'))
        w, h = num(el.get('width')), num(el.get('height'))
        rx = el.get('rx'); ry = el.get('ry')
        rx = num(rx, 0) if rx is not None else (num(ry, 0) if ry is not None else 0)
        ry = num(ry, 0) if ry is not None else rx
        if rx <= 0 and ry <= 0:
            return parse_path(f'M {x},{y} H {x+w} V {y+h} H {x} Z')
        rx = min(rx, w / 2); ry = min(ry, h / 2)
        return parse_path(
            f'M {x+rx},{y} H {x+w-rx} A {rx},{ry} 0 0 1 {x+w},{y+ry} '
            f'V {y+h-ry} A {rx},{ry} 0 0 1 {x+w-rx},{y+h} '
            f'H {x+rx} A {rx},{ry} 0 0 1 {x},{y+h-ry} '
            f'V {y+ry} A {rx},{ry} 0 0 1 {x+rx},{y} Z')
    if tag in ('circle', 'ellipse'):
        cx, cy = num(el.get('cx')), num(el.get('cy'))
        if tag == 'circle':
            rx = ry = num(el.get('r'))
        else:
            rx, ry = num(el.get('rx')), num(el.get('ry'))
        if rx <= 0 or ry <= 0:
            return None
        return parse_path(f'M {cx-rx},{cy} A {rx},{ry} 0 1 0 {cx+rx},{cy} '
                          f'A {rx},{ry} 0 1 0 {cx-rx},{cy} Z')
    if tag == 'line':
        return parse_path(f"M {num(el.get('x1'))},{num(el.get('y1'))} "
                          f"L {num(el.get('x2'))},{num(el.get('y2'))}")
    if tag in ('polyline', 'polygon'):
        pts = [p for p in re.split(r'[,\s]+', (el.get('points') or '').strip()) if p]
        if len(pts) < 4:
            return None
        d = 'M ' + ' L '.join(f'{pts[i]},{pts[i+1]}' for i in range(0, len(pts) - 1, 2))
        if tag == 'polygon':
            d += ' Z'
        return parse_path(d)
    return None


# ---------------------------------------------------------------- flattening
def flatten_segment(seg, tol, out):
    """Adaptive subdivision of one segment; appends points EXCLUDING the start point."""
    if isinstance(seg, Line):
        out.append(seg.end)
        return
    stack = [(0.0, 1.0, seg.point(0.0), seg.point(1.0))]
    pts = []
    while stack:
        t0, t1, p0, p1 = stack.pop()
        tm = 0.5 * (t0 + t1)
        pm = seg.point(tm)
        chord = p1 - p0
        L = abs(chord)
        if L < 1e-12:
            dev = abs(pm - p0)
        else:
            dev = abs(((pm - p0).real * (-chord.imag) + (pm - p0).imag * chord.real) / L)
        if dev <= tol or (t1 - t0) < 1e-4:
            pts.append((t1, p1))
        else:
            stack.append((tm, t1, pm, p1))
            stack.append((t0, tm, p0, pm))
    pts.sort(key=lambda x: x[0])
    for _, p in pts:
        out.append(p)


def subpath_points(sp, tol):
    pts = [sp[0].start]
    for seg in sp:
        flatten_segment(seg, tol, pts)
    return np.array([[z.real, z.imag] for z in pts])


# ---------------------------------------------------------------- circle / arc fitting
def fit_circle(P):
    """Algebraic circle fit. Returns (cx, cy, r, max_residual)."""
    x, y = P[:, 0], P[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(P))]
    b = x ** 2 + y ** 2
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy, c = sol
    r2 = c + cx ** 2 + cy ** 2
    if r2 <= 0:
        return None
    r = math.sqrt(r2)
    res = np.abs(np.hypot(x - cx, y - cy) - r).max()
    return cx, cy, r, res


def seg_as_arc(seg, tol):
    """If a curve segment is a circular arc within tol, return (bulge, ) else None."""
    if isinstance(seg, Line):
        return None
    ts = np.linspace(0, 1, 9)
    P = np.array([[seg.point(t).real, seg.point(t).imag] for t in ts])
    # near-straight curve -> plain line segment
    chord = P[-1] - P[0]
    L = np.hypot(*chord)
    if L > 1e-12:
        dev = np.abs((P - P[0]) @ np.array([-chord[1], chord[0]]) / L).max()
        if dev <= tol:
            return 0.0
    f = fit_circle(P)
    if f is None:
        return None
    cx, cy, r, res = f
    if res > tol or r <= 0:
        return None
    a0 = math.atan2(P[0, 1] - cy, P[0, 0] - cx)
    a1 = math.atan2(P[-1, 1] - cy, P[-1, 0] - cx)
    am = math.atan2(P[4, 1] - cy, P[4, 0] - cx)

    def norm(a):
        while a <= -math.pi: a += 2 * math.pi
        while a > math.pi: a -= 2 * math.pi
        return a
    d_ccw = norm(a1 - a0) % (2 * math.pi)
    # decide sweep direction from the midpoint
    if (norm(am - a0) % (2 * math.pi)) <= d_ccw:
        sweep = d_ccw
    else:
        sweep = -((2 * math.pi) - d_ccw)
    if abs(sweep) < 1e-9 or abs(sweep) > math.pi * 0.95:
        return None
    bulge = math.tan(sweep / 4.0)
    # validate: rebuild the arc and make sure the real curve stays within tol of it
    ang = np.linspace(a0, a0 + sweep, 48)
    A = np.c_[cx + r * np.cos(ang), cy + r * np.sin(ang)]
    chk = np.array([[seg.point(t).real, seg.point(t).imag] for t in np.linspace(0, 1, 17)])
    d2 = ((chk[:, None, :] - A[None, :, :]) ** 2).sum(-1).min(1)
    if math.sqrt(d2.max()) > tol:
        return None
    return bulge


# ---------------------------------------------------------------- contour container
class Contour:
    def __init__(self, pts, closed, layer, bulges=None, circle=None, el=-1, fill=None):
        self.el = el              # document order index of the owning SVG element
        self.fill = fill          # resolved fill colour of that element
        self.poly_id = None       # which resolved island this ring belongs to
        self.is_hole = False      # True for a knockout ring
        self.dense = None         # accurate flattened polyline, for boolean ops
        self.pts = pts            # Nx2, already in output (mm, y-up) space
        self.closed = closed
        self.layer = layer
        self.bulges = bulges      # list len == len(pts), bulge for segment starting at i
        self.circle = circle      # (cx, cy, r) if a true circle
        self.dense = None

    def geom(self):
        """Accurate outline. self.pts may hold only arc endpoints, so anything
        doing real geometry must use this, never self.pts."""
        return self.dense if self.dense is not None else self.pts

    def area(self):
        P = self.geom()
        if len(P) < 3:
            return 0.0
        x, y = P[:, 0], P[:, 1]
        return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# ---------------------------------------------------------------- main extraction
def extract(svg_file, tol_user, layer_map, default_layer):
    tree = etree.parse(svg_file)
    root = tree.getroot()

    vb = root.get('viewBox')
    W = root.get('width'); H = root.get('height')

    def to_mm(s):
        if s is None:
            return None
        m = re.match(r'\s*(-?[\d.eE+-]+)\s*([a-z%]*)', s)
        v = float(m.group(1)); u = m.group(2)
        return {'mm': 1, '': 25.4 / 96, 'px': 25.4 / 96, 'cm': 10, 'in': 25.4,
                'pt': 25.4 / 72, 'pc': 25.4 / 6}.get(u, 25.4 / 96) * v

    if vb:
        vx, vy, vw, vh = [float(v) for v in re.split(r'[,\s]+', vb.strip())]
    else:
        vx = vy = 0; vw = to_mm(W) or 100; vh = to_mm(H) or 100
    wmm = to_mm(W) or vw
    hmm = to_mm(H) or vh
    sx = wmm / vw
    sy = hmm / vh
    tol = tol_user / max(sx, sy)   # tolerance expressed in SVG user units

    contours = []
    stats = {'circles': 0, 'arcs': 0, 'polys': 0, 'skipped': 0}
    counter = [0]

    def resolved_fill(el):
        d = {}
        for kv in (el.get('style') or '').split(';'):
            if ':' in kv:
                k, v = kv.split(':', 1)
                d[k.strip()] = v.strip()
        f = d.get('fill', el.get('fill'))
        if f is None:
            f = '#000000'          # SVG default fill
        return f.strip().lower()

    def walk(el, M, layer):
        lab = el.get(INK + 'label')
        gm = el.get(INK + 'groupmode')
        if gm == 'layer' and lab:
            layer = layer_map.get(lab, re.sub(r'[^A-Za-z0-9_.-]', '_', lab).upper())
        M = M @ parse_transform(el.get('transform'))
        tag = etree.QName(el).localname
        if tag in ('defs', 'metadata', 'title', 'desc', 'namedview'):
            return
        style = (el.get('style') or '')
        if 'display:none' in style.replace(' ', ''):
            return
        p = element_to_path(el)
        if p is not None and len(p):
            emit(p, M, layer, counter[0], resolved_fill(el))
            counter[0] += 1
        for c in el:
            if isinstance(c.tag, str):
                walk(c, M, layer)

    # SVG user space -> DXF mm, y-up
    def out_xy(P):
        Q = P.copy()
        Q[:, 0] = (Q[:, 0] - vx) * sx
        Q[:, 1] = hmm - (Q[:, 1] - vy) * sy
        return Q

    def emit(path, M, layer, elidx, fill):
        uniform = abs(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]) > 0 and \
                  abs(abs(M[0, 0]) - abs(M[1, 1])) < 1e-6 and abs(M[0, 1]) < 1e-9 and abs(M[1, 0]) < 1e-9
        for sp in path.continuous_subpaths():
            P = subpath_points(sp, tol)
            closed = abs(sp[0].start - sp[-1].end) < 1e-7
            if closed and len(P) > 1 and np.hypot(*(P[0] - P[-1])) < 1e-9:
                P = P[:-1]
            if len(P) < 2:
                stats['skipped'] += 1
                continue
            Pt = out_xy(apply(M, P))
            dense = Pt                      # accurate; arc branches below may
                                            # replace Pt with sparse endpoints

            # --- whole-subpath circle?
            if closed and len(Pt) >= 6 and abs(sx - sy) < 1e-6 and uniform:
                probe = subpath_points(sp, tol / 4)      # untransformed samples
                D = out_xy(apply(M, probe))              # ...and in output space
                f = fit_circle(D)
                if f and f[3] <= tol_user * 2 and f[2] > 1e-4:
                    cc = Contour(Pt, True, layer, circle=(f[0], f[1], f[2]),
                                 el=elidx, fill=fill)
                    cc.dense = D
                    contours.append(cc)
                    stats['circles'] += 1
                    continue

            # --- per-segment arc bulges
            bulges = None
            if uniform and abs(sx - sy) < 1e-6:
                bl = []
                pts2 = [sp[0].start]
                ok = True
                for seg in sp:
                    b = seg_as_arc(seg, tol) if not isinstance(seg, Line) else 0.0
                    if b is None:
                        ok = False
                        break
                    bl.append(b)
                    pts2.append(seg.end)
                if ok and len(bl) <= 64:
                    Pv = np.array([[z.real, z.imag] for z in pts2])
                    if closed:
                        Pv = Pv[:-1]
                    Pv = out_xy(apply(M, Pv))
                    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
                    flip = -1.0 if det > 0 else 1.0   # y-flip reverses arc direction
                    bulges = [b * flip for b in bl]
                    if not closed:
                        bulges = bulges + [0.0]
                    Pt = Pv
                    stats['arcs'] += sum(1 for b in bl if abs(b) > 1e-12)
            if bulges is None:
                stats['polys'] += 1
            cc = Contour(Pt, closed, layer, bulges=bulges, el=elidx, fill=fill)
            cc.dense = dense
            contours.append(cc)

    walk(root, np.eye(3), default_layer)
    return contours, wmm, hmm, stats




# ---------------------------------------------------------------- ink resolution
INK_WORDS = {'#ffffff', '#fff', 'white'}


def resolve_ink(contours, layer, ink_colors=INK_WORDS):
    """Composite the layer into an unambiguous printed-ink region.

    Nesting depth alone cannot decide fill when two ink shapes overlap (e.g. a
    connecting bar crossing a pill that already contains knockout text).  So we
    replay the SVG in document order: each element becomes a region under its own
    even-odd rule, then ink-coloured elements are unioned in and every other
    colour is subtracted.  What comes out is a set of rings where every exterior
    is ink and every interior is bare substrate.

    The booleans run on each contour's accurate dense outline, never on its
    output form -- an arc-bulge contour stores only its arc endpoints in .pts,
    and feeding those to shapely would silently replace every curve with its
    chord.  Afterwards each resulting ring is matched back against the input; a
    ring the booleans left untouched (the overwhelming majority -- glyphs that
    overlap nothing) gets its exact CIRCLE / arc-bulge representation restored,
    so curves survive the round trip bit for bit.
    """
    from shapely.geometry import Polygon, MultiPolygon

    mine = [c for c in contours if c.layer == layer]
    others = [c for c in contours if c.layer != layer]
    if not mine:
        return contours

    by_el = {}
    for c in mine:
        by_el.setdefault(c.el, []).append(c)

    # index the originals so untouched rings can be restored exactly
    originals = []
    for c in mine:
        if not c.closed or len(c.geom()) < 3:
            continue
        g = Polygon(c.geom())
        if not g.is_valid:
            g = g.buffer(0)
        if not g.is_empty and g.geom_type == 'Polygon':
            originals.append((g, c))

    def restore(ring_poly):
        """Return the original contour identical to this ring, if there is one."""
        b = ring_poly.bounds
        best = None
        for g, c in originals:
            gb = g.bounds
            if max(abs(gb[i] - b[i]) for i in range(4)) > 1e-6:
                continue
            if abs(g.area - ring_poly.area) > 1e-9:
                continue
            try:
                if g.symmetric_difference(ring_poly).area < 1e-9:
                    best = c
                    break
            except Exception:
                continue
        return best

    ink = Polygon()
    for el in sorted(by_el):
        region = Polygon()
        for c in by_el[el]:
            if not c.closed or len(c.geom()) < 3:
                continue
            g = Polygon(c.geom())
            if not g.is_valid:
                g = g.buffer(0)
            if g.is_empty:
                continue
            region = region.symmetric_difference(g)      # even-odd within element
        if region.is_empty:
            continue
        fill = (by_el[el][0].fill or '').lower()
        ink = ink.union(region) if fill in ink_colors else ink.difference(region)
        if not ink.is_valid:
            ink = ink.buffer(0)

    if ink.is_empty:
        return contours
    polys = ink.geoms if isinstance(ink, MultiPolygon) else [ink]
    out = []
    stats = {'restored': 0, 'flattened': 0}
    pid = 0
    for g in polys:
        if g.is_empty or g.area <= 0:
            continue
        rings = [(g.exterior, True, False)] + [(r, False, True) for r in g.interiors]
        for ring, want_ccw, is_hole in rings:
            P = np.array(ring.coords)[:-1]
            if len(P) < 3:
                continue
            src = restore(Polygon(ring))
            if src is not None and (src.bulges or src.circle is not None):
                c = Contour(src.pts.copy(), True, layer,
                            bulges=list(src.bulges) if src.bulges else None,
                            circle=src.circle, el=-1, fill='#ffffff')
                c.dense = src.dense if src.dense is not None else P
                stats['restored'] += 1
            else:
                c = Contour(P, True, layer, el=-1, fill='#ffffff')
                c.dense = P
                stats['flattened'] += 1
            # orientation is decided on the accurate outline, then applied to
            # whichever representation we are actually emitting
            a = 0.5 * float(np.dot(c.dense[:, 0], np.roll(c.dense[:, 1], -1)) -
                            np.dot(c.dense[:, 1], np.roll(c.dense[:, 0], -1)))
            if (a > 0) != want_ccw:
                c.dense = c.dense[::-1]
                if c.circle is None:
                    c.pts = c.pts[::-1]
                    if c.bulges:
                        n = len(c.bulges)
                        c.bulges = [-c.bulges[(n - 2 - j) % n] for j in range(n)]
            c.poly_id = pid
            c.is_hole = is_hole
            out.append(c)
        pid += 1
    resolve_ink.stats = stats
    return others + out


# ---------------------------------------------------------------- winding
def fix_winding(contours, skip_layers=()):
    """Outer contours CCW, holes CW, based on nesting depth (even-odd)."""
    from shapely.geometry import Polygon, Point
    from shapely.prepared import prep
    by_layer = {}
    for c in contours:
        by_layer.setdefault(c.layer, []).append(c)
    for layer, cs in by_layer.items():
        if layer in skip_layers:
            continue
        closed = [c for c in cs if c.closed and len(c.geom()) >= 3]
        polys = []
        for c in closed:
            try:
                g = Polygon(c.geom())
                if not g.is_valid:
                    g = g.buffer(0)
            except Exception:
                g = None
            polys.append(g)
        prepped = [prep(g) if g is not None and not g.is_empty else None for g in polys]
        for i, c in enumerate(closed):
            depth = 0
            pt = Point(c.geom()[0])
            for j, pg in enumerate(prepped):
                if i == j or pg is None:
                    continue
                if polys[j].area <= polys[i].area if polys[i] is not None else False:
                    continue
                try:
                    if pg.contains(pt):
                        depth += 1
                except Exception:
                    pass
            want_ccw = (depth % 2 == 0)
            is_ccw = c.area() > 0
            if is_ccw != want_ccw:
                c.pts = c.pts[::-1]
                if c.dense is not None:
                    c.dense = c.dense[::-1]
                if c.bulges:
                    n = len(c.bulges)
                    c.bulges = [-c.bulges[(n - 2 - j) % n] for j in range(n)]
            c.depth = depth
    return contours


# ---------------------------------------------------------------- DXF writing
def write_dxf(path, contours, wmm, hmm, dxfversion, layer_colors, hatch_layers,
              fill_layer=None, outline_layers=None, outline_state='on',
              skip_outlines=()):
    import ezdxf
    from ezdxf import const
    doc = ezdxf.new(dxfversion, setup=True)
    doc.header['$INSUNITS'] = 4        # millimetres
    doc.header['$MEASUREMENT'] = 1     # metric
    doc.header['$LUNITS'] = 2
    msp = doc.modelspace()

    outline_layers = outline_layers or {}
    for lay in sorted({c.layer for c in contours}):
        if lay not in doc.layers:
            doc.layers.add(lay, color=layer_colors.get(lay, 7))
    for src, dst in outline_layers.items():
        if dst not in doc.layers:
            doc.layers.add(dst, color=layer_colors.get(src, 7))

    def target(lay):
        return outline_layers.get(lay, lay)

    for c in contours:
        if c.layer in skip_outlines:
            continue                              # HATCH boundaries carry the geometry
        if c.circle is not None:
            cx, cy, r = c.circle
            msp.add_circle((cx, cy), r,
                           dxfattribs={'layer': target(c.layer), 'lineweight': 0})
            continue
        pts = c.pts
        if c.bulges:
            data = [(p[0], p[1], 0.0, 0.0, b) for p, b in zip(pts, c.bulges)]
            msp.add_lwpolyline(data, format='xyseb', close=c.closed,
                               dxfattribs={'layer': target(c.layer), 'lineweight': 0})
        else:
            msp.add_lwpolyline([(p[0], p[1]) for p in pts], format='xy', close=c.closed,
                               dxfattribs={'layer': target(c.layer), 'lineweight': 0})

    # solid fill so the print house sees ink, not wireframe
    for lay in hatch_layers:
        cs = [c for c in contours if c.layer == lay and c.closed and len(c.geom()) >= 3]
        if not cs:
            continue
        fill_lay = fill_layer or lay
        if fill_lay not in doc.layers:
            doc.layers.add(fill_lay, color=layer_colors.get(lay, 7))

        def add_boundary(h, c, flags):
            """Boundary path that keeps the curves: bulges for arcs, and a circle
            as the two-vertex bulge form CAD expects."""
            if c.circle is not None:
                cx, cy, r = c.circle
                h.paths.add_polyline_path(
                    [(cx - r, cy, 1.0), (cx + r, cy, 1.0)],
                    is_closed=True, flags=flags)
            elif c.bulges:
                h.paths.add_polyline_path(
                    [(p[0], p[1], b) for p, b in zip(c.pts, c.bulges)],
                    is_closed=True, flags=flags)
            else:
                h.paths.add_polyline_path(
                    [(p[0], p[1]) for p in c.pts], is_closed=True, flags=flags)

        def new_hatch():
            h = msp.add_hatch(color=layer_colors.get(lay, 7),
                              dxfattribs={'layer': fill_lay})
            h.set_solid_fill()
            h.dxf.hatch_style = 0       # normal: honour the island flags below
            h.dxf.solid_fill = 1
            return h

        islands = {}
        for c in cs:
            islands.setdefault(c.poly_id, []).append(c)

        if list(islands) == [None]:
            # raw mode: no island structure, fall back to odd-parity over everything
            h = new_hatch()
            h.dxf.hatch_style = 0
            for c in cs:
                add_boundary(h, c, const.BOUNDARY_PATH_DEFAULT)
            continue

        # one HATCH per resolved island: far friendlier to older shop CAM than a
        # single entity carrying hundreds of boundary paths
        for pid in sorted(k for k in islands if k is not None):
            group = islands[pid]
            outers = [c for c in group if not c.is_hole]
            holes = [c for c in group if c.is_hole]
            if not outers:
                continue
            h = new_hatch()
            for c in outers:
                add_boundary(h, c,
                             const.BOUNDARY_PATH_EXTERNAL | const.BOUNDARY_PATH_OUTERMOST)
            for c in holes:
                add_boundary(h, c, const.BOUNDARY_PATH_DEFAULT)
    for lay in doc.layers:
        lay.dxf.lineweight = 0                    # 0.00 mm: never fatten the artwork
    for dst in outline_layers.values():
        if outline_state == 'off':
            doc.layers.get(dst).off()

    # ezdxf drops $EXTMIN/$EXTMAX unless the modelspace layout carries them, and
    # treats an exact (0,0,0) minimum as "unset" -- hence the epsilon.
    msp.dxf.extmin = (-1e-9, -1e-9, 0.0)
    msp.dxf.extmax = (float(wmm), float(hmm), 0.0)
    doc.header['$LIMMIN'] = (0.0, 0.0)
    doc.header['$LIMMAX'] = (float(wmm), float(hmm))
    try:
        from ezdxf import zoom
        zoom.window(msp, (0, 0), (wmm, hmm))
    except Exception:
        pass
    doc.saveas(path)


# ---------------------------------------------------------------- PDF artwork
def write_pdf(path, contours, wmm, hmm, ink_layer, bg='#101010', ink='#ffffff'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.path import Path as MP
    from matplotlib.patches import PathPatch
    fig = plt.figure(figsize=(wmm / 25.4, hmm / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, wmm); ax.set_ylim(0, hmm); ax.axis('off')
    ax.add_patch(plt.Rectangle((0, 0), wmm, hmm, facecolor=bg, edgecolor='none'))
    verts, codes = [], []
    for c in contours:
        if c.layer != ink_layer or not c.closed or len(c.geom()) < 3:
            continue
        G = c.geom()
        P = np.vstack([G, G[:1]])
        verts.extend(P.tolist())
        codes.extend([MP.MOVETO] + [MP.LINETO] * (len(P) - 2) + [MP.CLOSEPOLY])
    if verts:
        ax.add_patch(PathPatch(MP(verts, codes), facecolor=ink, edgecolor='none'))
    fig.savefig(path, format='pdf')
    plt.close(fig)


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('svg')
    ap.add_argument('prefix')
    ap.add_argument('--tol', type=float, default=0.0005, help='flattening tolerance in mm')
    ap.add_argument('--layer-map', default='')
    ap.add_argument('--silk-layer', default='SILKSCREEN')
    ap.add_argument('--no-hatch', action='store_true')
    ap.add_argument('--silk-mode', choices=['resolve', 'raw'], default='resolve',
                    help="resolve = composite ink/knockout with boolean ops (default); "
                         "raw = export every contour as drawn")
    ap.add_argument('--ink-color', default='#ffffff')
    ap.add_argument('--silk-outlines', choices=['off', 'on', 'none'], default='off',
                    help="what to do with the polyline copies of the silkscreen "
                         "contours: 'off' (default) = own layer, switched off so "
                         "viewers cannot stroke them over the fill; 'on' = own layer, "
                         "visible; 'none' = omit entirely (HATCH boundaries still "
                         "carry the geometry)")
    ap.add_argument('--fill-layer', default=None,
                    help='put the solid ink HATCHes on their own layer '
                         '(e.g. SILKSCREEN_FILL) instead of on the silkscreen layer')
    ap.add_argument('--dxf-version', default='AC1015')
    args = ap.parse_args()

    lm = {}
    for kv in args.layer_map.split(','):
        if '=' in kv:
            k, v = kv.split('=', 1)
            lm[k.strip()] = v.strip()

    contours, wmm, hmm, stats = extract(args.svg, args.tol, lm, 'DEFAULT')
    raw_n = sum(1 for c in contours if c.layer == args.silk_layer)
    if args.silk_mode == 'resolve':
        contours = resolve_ink(contours, args.silk_layer,
                               {args.ink_color.lower(), 'white'})
        res_n = sum(1 for c in contours if c.layer == args.silk_layer)
        st = getattr(resolve_ink, 'stats', {})
        print(f'silkscreen resolved: {raw_n} drawn contours -> {res_n} ink/knockout '
              f'rings ({st.get("restored", 0)} kept exact curves, '
              f'{st.get("flattened", 0)} re-cut by the booleans)')
        fix_winding(contours, skip_layers=(args.silk_layer,))
    else:
        fix_winding(contours)

    colors = {'EDGE_CUTS': 1, args.silk_layer: 7}
    ol, skip = {}, ()
    if not args.no_hatch:
        if args.silk_outlines == 'none':
            skip = (args.silk_layer,)
        else:
            ol = {args.silk_layer: args.silk_layer + '_OUTLINE'}
    write_dxf(args.prefix + '.dxf', contours, wmm, hmm, args.dxf_version, colors,
              [] if args.no_hatch else [args.silk_layer], fill_layer=args.fill_layer,
              outline_layers=ol, outline_state=args.silk_outlines, skip_outlines=skip)
    write_dxf(args.prefix + '-outlines-only.dxf', contours, wmm, hmm, 'AC1015', colors, [])
    write_pdf(args.prefix + '-silkscreen.pdf', contours, wmm, hmm, args.silk_layer)

    print(f'panel {wmm:.3f} x {hmm:.3f} mm')
    print(f'contours: {len(contours)}   circles: {stats["circles"]}   '
          f'arc-bulge polylines: {stats["arcs"]} arcs   plain polylines: {stats["polys"]}')
    for lay in sorted({c.layer for c in contours}):
        n = sum(1 for c in contours if c.layer == lay)
        print(f'  layer {lay}: {n} entities')


if __name__ == '__main__':
    main()
