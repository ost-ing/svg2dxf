#!/usr/bin/env python3
"""
panel2dxf.py - flatten an Inkscape panel SVG into cutout/silkscreen layers of plain
filled paths, run svg2dxf.py on it, and check the DXF against the source artwork.

Usage:
  python3 panel2dxf.py panel.svg out_prefix
      [--cutout outline,edge-cuts] [--silkscreen silkscreen]
      [--ink-color #ffffff] [--tol 0.0005] [--check-tol 0.015] [--inkscape inkscape]

Inkscape converts every object to a path and every silkscreen stroke to a filled
path. Strokes on the cutout layer are dropped: a cutout is its fill geometry.

Nothing is written unless the DXF matches the source: every cutout shape, and the
silkscreen ink composited in document order with each stroke modelled as its
outline buffered by half the stroke width. Any region of disagreement wider than
--check-tol fails the run.

On success writes out_prefix-flattened.svg, out_prefix.dxf,
out_prefix-outlines-only.dxf and out_prefix-silkscreen.pdf.
"""
import argparse, math, os, re, shutil, subprocess, sys, tempfile
import numpy as np
import ezdxf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from scipy import ndimage
from ezdxf import path as dxfpath
from lxml import etree
from shapely.affinity import affine_transform
from shapely.geometry import LinearRing, LineString, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import polygonize, unary_union
from svgpathtools import Line, parse_path
from svg2dxf import parse_transform

SVG = '{http://www.w3.org/2000/svg}'
INK = '{http://www.inkscape.org/namespaces/inkscape}'

NON_GRAPHIC = {'defs', 'namedview', 'metadata', 'title', 'desc', 'style'}
SHAPES = {'path', 'rect', 'circle', 'ellipse', 'line', 'polyline', 'polygon', 'text'}
INHERITED = {
    'fill', 'fill-opacity', 'fill-rule', 'stroke', 'stroke-width', 'stroke-opacity',
    'stroke-linecap', 'stroke-linejoin', 'stroke-miterlimit', 'stroke-dasharray',
    'stroke-dashoffset', 'visibility', 'font-style', 'font-variant', 'font-weight',
    'font-stretch', 'font-size', 'font-family', 'line-height', 'letter-spacing',
    'word-spacing', 'text-anchor', 'writing-mode', 'direction',
    '-inkscape-font-specification',
}
PATH_DRAW = re.compile(r'[LlHhVvCcSsQqTtAa]|[Mm]\s*[-\d.eE+]+[\s,]*[-\d.eE+]+[\s,]+[-\d.eE+]')
CAPS = {'round': 'round', 'butt': 'flat', 'square': 'square'}
JOINS = {'round': 'round', 'miter': 'mitre', 'bevel': 'bevel'}
SAMPLE_STEP = 0.01
UNSUPPORTED_EFFECTS = ('clip-path', 'mask', 'filter', 'marker', 'marker-start', 'marker-mid',
                       'marker-end')
RENDER_PX_PER_UNIT = 26
RENDER_TILE = 160
MM_PER_PX = 25.4 / 96
CUTOUT_BBOX_TOL = 0.005
CSS_TO_FC_WEIGHT = ([100, 200, 300, 350, 380, 400, 500, 600, 700, 800, 900],
                    [0, 40, 50, 55, 75, 80, 100, 180, 200, 205, 210])
STYLE_WEIGHTS = {'thin': 100, 'extralight': 200, 'ultralight': 200, 'light': 300,
                 'normal': 400, 'regular': 400, 'book': 400, 'medium': 500, 'semibold': 600,
                 'demibold': 600, 'bold': 700, 'extrabold': 800, 'ultrabold': 800,
                 'black': 900, 'heavy': 900}
STYLE_SLANTS = {'italic': 100, 'oblique': 110}
STYLE_IGNORED = {'condensed', 'semicondensed', 'extracondensed', 'ultracondensed',
                 'expanded', 'semiexpanded', 'extraexpanded', 'ultraexpanded'}


class PanelError(Exception):
    pass


def local(el):
    return etree.QName(el).localname


def style_of(el):
    d = {k: el.get(k) for k in INHERITED if el.get(k) is not None}
    for kv in (el.get('style') or '').split(';'):
        if ':' in kv:
            k, v = kv.split(':', 1)
            d[k.strip()] = v.strip()
    return d


def write_style(el, d):
    for k in INHERITED:
        if k in el.attrib:
            del el.attrib[k]
    el.set('style', ';'.join(f'{k}:{v}' for k, v in d.items()))


def hidden(d):
    return d.get('display') == 'none' or d.get('visibility') == 'hidden'


def number(v):
    return float(re.match(r'\s*[-\d.eE+]+', v).group())


def colour(v):
    v = (v or '').strip().lower()
    if v in ('white', '#fff', '#ffffff'):
        return '#ffffff'
    if v in ('black', '#000', '#000000'):
        return '#000000'
    return v


def matrix_attr(M):
    return 'matrix(%s)' % ','.join(
        repr(float(v)) for v in (M[0, 0], M[1, 0], M[0, 1], M[1, 1], M[0, 2], M[1, 2]))


def flatten_into(src, M, inherited, out, path):
    own = style_of(src)
    if hidden(own):
        return
    M = M @ parse_transform(src.get('transform'))
    tag = local(src)
    if tag in NON_GRAPHIC:
        return
    for k in UNSUPPORTED_EFFECTS:
        if src.get(k) or own.get(k, 'none') != 'none':
            raise PanelError(f'<{tag}> id={src.get("id")} uses {k}, which is not supported')
    if tag in ('g', 'a'):
        merged = dict(inherited)
        merged.update({k: v for k, v in own.items() if k in INHERITED})
        for c in src:
            if isinstance(c.tag, str):
                flatten_into(c, M, merged, out, path + [src.get('id')])
        return
    if tag not in SHAPES:
        raise PanelError(f'unsupported <{tag}> id={src.get("id")} under {"/".join(str(i) for i in path)}')
    el = etree.fromstring(etree.tostring(src))
    merged = dict(inherited)
    merged.update(own)
    write_style(el, merged)
    if 'transform' in el.attrib:
        del el.attrib['transform']
    if not np.allclose(M, np.eye(3)):
        el.set('transform', matrix_attr(M))
    out.append(el)


def new_layer(root, name):
    g = etree.SubElement(root, SVG + 'g')
    g.set('id', 'layer-' + name)
    g.set(INK + 'groupmode', 'layer')
    g.set(INK + 'label', name)
    return g


def layers_of(root):
    return {g.get(INK + 'label'): g for g in root
            if isinstance(g.tag, str) and g.get(INK + 'groupmode') == 'layer'}


def build_layers(src_file, mapping):
    tree = etree.parse(src_file)
    root = tree.getroot()
    sources = []
    for c in list(root):
        if not isinstance(c.tag, str) or local(c) in NON_GRAPHIC:
            continue
        lab = c.get(INK + 'label')
        if lab not in mapping:
            raise PanelError(f'top-level <{local(c)}> label={lab!r} id={c.get("id")} '
                             f'is in neither --cutout nor --silkscreen')
        sources.append((c, mapping[lab]))
        root.remove(c)
    missing = set(mapping) - {c.get(INK + 'label') for c, _ in sources}
    if missing:
        raise PanelError(f'no top-level group labelled {sorted(missing)}')
    layers = {name: new_layer(root, name) for name in dict.fromkeys(mapping.values())}
    for c, name in sources:
        leaves = []
        flatten_into(c, np.eye(3), {}, leaves, [])
        layers[name].extend(leaves)
    return tree


def regroup(tree):
    for name, layer in layers_of(tree.getroot()).items():
        for g in [g for g in layer if isinstance(g.tag, str) and local(g) == 'g']:
            leaves = []
            flatten_into(g, np.eye(3), {}, leaves, [name])
            at = layer.index(g)
            layer.remove(g)
            for k, el in enumerate(leaves):
                layer.insert(at + k, el)


def drop_degenerate(tree):
    dropped = []
    for layer in layers_of(tree.getroot()).values():
        for el in list(layer):
            if local(el) == 'path' and not PATH_DRAW.search(el.get('d') or ''):
                dropped.append(el.get('id'))
                layer.remove(el)
    return dropped


def has_stroke(d):
    return d.get('stroke', 'none') != 'none' and number(d.get('stroke-width', '1')) != 0 \
        and number(d.get('stroke-opacity', '1')) != 0


def has_fill(d):
    return d.get('fill', '#000000') != 'none'


def strip_strokes(tree, layer_names):
    for name, layer in layers_of(tree.getroot()).items():
        for el in layer:
            d = style_of(el)
            if name in layer_names or not has_stroke(d):
                d['stroke'] = 'none'
                write_style(el, d)


def similarity_scale(M):
    a, b, c, e = M[0, 0], M[1, 0], M[0, 1], M[1, 1]
    sx, sy = math.hypot(a, b), math.hypot(c, e)
    if abs(sx - sy) > 1e-9 * max(sx, sy) or abs(a * c + b * e) > 1e-9 * sx * sy:
        return None
    return sx


def check_paintable(tree, ink_colour):
    allowed = {ink_colour, '#000000'}
    for name, layer in layers_of(tree.getroot()).items():
        for el in layer:
            d = style_of(el)
            where = f'{name} id={el.get("id")}'
            if local(el) != 'path':
                raise PanelError(f'{where} is <{local(el)}>, not a path')
            if el.get('class'):
                raise PanelError(f'{where} is styled by class {el.get("class")}')
            if el.get(INK + 'path-effect'):
                raise PanelError(f'{where} still has a live path effect')
            if not has_fill(d) and not has_stroke(d):
                raise PanelError(f'{where} has neither fill nor stroke')
            for k in ('opacity', 'fill-opacity', 'stroke-opacity'):
                if k in d and number(d[k]) != 1:
                    raise PanelError(f'{where} has {k}:{d[k]}')
            if name == 'cutout':
                continue
            paints = []
            if has_fill(d):
                paints.append(colour(d.get('fill', '#000000')))
            if has_stroke(d):
                paints.append(colour(d['stroke']))
            for p in paints:
                if p not in allowed:
                    raise PanelError(f'{where} is painted {p}; silkscreen allows '
                                     f'{sorted(allowed)}')
            if len(set(paints)) > 1:
                raise PanelError(f'{where} has fill {paints[0]} and stroke {paints[1]}')
            if has_stroke(d):
                if d.get('stroke-dasharray', 'none') != 'none':
                    raise PanelError(f'{where} has a dashed stroke')
                if d.get('stroke-linecap', 'butt') not in CAPS:
                    raise PanelError(f'{where} has stroke-linecap {d["stroke-linecap"]}')
                if d.get('stroke-linejoin', 'miter') not in JOINS:
                    raise PanelError(f'{where} has stroke-linejoin {d["stroke-linejoin"]}')
                if similarity_scale(parse_transform(el.get('transform'))) is None:
                    raise PanelError(f'{where} has a stroke under a non-uniform transform')


def check_flattened(tree, expected_layers):
    root = tree.getroot()
    layers = layers_of(root)
    if set(layers) != set(expected_layers):
        raise PanelError(f'layers {sorted(layers)} != {sorted(expected_layers)}')
    for c in root:
        if isinstance(c.tag, str) and local(c) not in NON_GRAPHIC \
                and c.get(INK + 'groupmode') != 'layer':
            raise PanelError(f'stray top-level <{local(c)}> id={c.get("id")}')
    for name, layer in layers.items():
        for el in layer.iter():
            if el is layer or not isinstance(el.tag, str):
                continue
            where = f'{name} id={el.get("id")}'
            if local(el) != 'path' or el.getparent() is not layer:
                raise PanelError(f'{where} is a nested <{local(el)}>')
            d = style_of(el)
            if has_stroke(d):
                raise PanelError(f'{where} still has stroke {d["stroke"]}')
            if not has_fill(d):
                raise PanelError(f'{where} has no fill')


def subpaths(d):
    out = []
    for sp in parse_path(d).continuous_subpaths():
        pts = []
        for seg in sp:
            if isinstance(seg, Line):
                pts.append(seg.start)
            else:
                n = max(4, math.ceil(seg.length() / SAMPLE_STEP))
                pts += [seg.point(t) for t in np.linspace(0, 1, n, endpoint=False)]
        pts.append(sp[-1].end)
        xy = []
        for z in pts:
            if not xy or abs(complex(*xy[-1]) - z) > 1e-9:
                xy.append((z.real, z.imag))
        closed = abs(sp[0].start - sp[-1].end) < 1e-9
        if closed and len(xy) > 1 and math.dist(xy[0], xy[-1]) <= 1e-9:
            xy.pop()
        out.append((xy, closed))
    return out


def winding(q, xy):
    P = np.asarray(xy)
    a, b = P, np.roll(P, -1, axis=0)
    cross = (b[:, 0] - a[:, 0]) * (q[1] - a[:, 1]) - (q[0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    up = (a[:, 1] <= q[1]) & (b[:, 1] > q[1]) & (cross > 0)
    down = (a[:, 1] > q[1]) & (b[:, 1] <= q[1]) & (cross < 0)
    return int(up.sum()) - int(down.sum())


def fill_region(rings, rule):
    rings = [xy for xy, _ in rings if len(xy) >= 3]
    if not rings:
        return Polygon()
    faces = polygonize(unary_union([LinearRing(xy) for xy in rings]))
    kept = []
    for f in faces:
        q = f.representative_point()
        w = sum(winding((q.x, q.y), xy) for xy in rings)
        if (w % 2) if rule == 'evenodd' else w:
            kept.append(f)
    return unary_union(kept)


def stroke_region(subs, width, cap, join, miterlimit):
    r = width / 2
    parts = []
    for xy, closed in subs:
        if closed and len(xy) >= 3:
            parts.append(LinearRing(xy).buffer(r, join_style=JOINS[join], mitre_limit=miterlimit))
        elif len(set(xy)) >= 2:
            parts.append(LineString(xy).buffer(r, cap_style=CAPS[cap], join_style=JOINS[join],
                                               mitre_limit=miterlimit))
        elif cap == 'round':
            parts.append(Point(xy[0]).buffer(r))
        elif cap == 'square':
            x, y = xy[0]
            parts.append(Polygon([(x - r, y - r), (x + r, y - r), (x + r, y + r), (x - r, y + r)]))
    return unary_union(parts)


def element_region(el):
    d = style_of(el)
    subs = subpaths(el.get('d'))
    region = Polygon()
    if has_fill(d):
        region = fill_region([s for s in subs if s[1] or len(s[0]) >= 3],
                             d.get('fill-rule', 'nonzero'))
    if has_stroke(d):
        region = region.union(stroke_region(
            subs, number(d.get('stroke-width', '1')), d.get('stroke-linecap', 'butt'),
            d.get('stroke-linejoin', 'miter'), number(d.get('stroke-miterlimit', '4'))))
    M = parse_transform(el.get('transform'))
    return affine_transform(region, [M[0, 0], M[0, 1], M[1, 0], M[1, 1], M[0, 2], M[1, 2]])


def page(root):
    def mm(s):
        m = re.fullmatch(r'\s*([-\d.eE+]+)\s*mm\s*', s or '')
        if not m:
            raise PanelError(f'page size {s!r} is not in mm')
        return float(m.group(1))
    w, h = mm(root.get('width')), mm(root.get('height'))
    vx, vy, vw, vh = [float(v) for v in re.split(r'[,\s]+', root.get('viewBox').strip())]
    sx, sy = w / vw, h / vh
    return [sx, 0, 0, -sy, -vx * sx, h + vy * sy]


def expected_layers(tree, ink_colour):
    root = tree.getroot()
    to_dxf = page(root)
    layers = layers_of(root)
    cutouts = [(el.get('id'), affine_transform(element_region(el), to_dxf))
               for el in layers['cutout']]
    ink = Polygon()
    for el in layers['silkscreen']:
        d = style_of(el)
        paint = colour(d.get('fill', '#000000')) if has_fill(d) else colour(d['stroke'])
        region = affine_transform(element_region(el), to_dxf)
        ink = ink.union(region) if paint == ink_colour else ink.difference(region)
    return cutouts, ink


def dxf_layers(dxf_file, cutout_layer, silk_layer):
    msp = ezdxf.readfile(dxf_file).modelspace()
    cutouts = []
    for e in msp.query(f'*[layer=="{cutout_layer}"]'):
        if e.dxftype() == 'CIRCLE':
            c = e.dxf.center
            cutouts.append(Point(c.x, c.y).buffer(e.dxf.radius, quad_segs=256))
        elif e.dxftype() == 'LWPOLYLINE' and e.closed:
            pts = [(v.x, v.y) for v in dxfpath.make_path(e).flattening(0.0002)]
            cutouts.append(Polygon(pts))
        else:
            raise PanelError(f'unexpected {e.dxftype()} on {cutout_layer}')
    ink = []
    for h in msp.query(f'HATCH[layer=="{silk_layer}"]'):
        region = Polygon()
        for bp in h.paths:
            pts = [(v.x, v.y) for v in dxfpath.from_hatch_boundary_path(bp).flattening(0.0002)]
            ring = Polygon(pts)
            region = region.symmetric_difference(ring if ring.is_valid else ring.buffer(0))
        ink.append(region)
    return cutouts, unary_union(ink)


def disagreement(a, b, tol):
    diff = a.symmetric_difference(b)
    return [g for g in getattr(diff, 'geoms', [diff])
            if not g.is_empty and not g.buffer(-tol / 2).is_empty]


def check_dxf(tree, dxf_file, ink_colour, tol):
    want_cut, want_ink = expected_layers(tree, ink_colour)
    got_cut, got_ink = dxf_layers(dxf_file, 'CUTOUT', 'SILKSCREEN')
    problems = []
    if len(want_cut) != len(got_cut):
        problems.append(f'cutout: {len(want_cut)} shapes in the SVG, {len(got_cut)} in the DXF')
    unmatched = list(got_cut)
    for ident, w in want_cut:
        match = next((g for g in unmatched
                      if max(abs(p - q) for p, q in zip(g.bounds, w.bounds)) <= tol
                      and not disagreement(w, g, tol)), None)
        if match is None:
            problems.append(f'cutout id={ident} at {fmt(w.bounds)} has no matching DXF shape')
        else:
            unmatched.remove(match)
    for g in unmatched:
        problems.append(f'cutout: DXF shape at {fmt(g.bounds)} matches nothing in the SVG')
    for piece in disagreement(want_ink, got_ink, tol):
        problems.append(f'silkscreen differs by more than {tol} mm at {fmt(piece.bounds)}')
    if problems:
        raise PanelError('DXF does not match the source artwork:\n  ' + '\n  '.join(problems))


def visible_tree(svg_file, keep_labels):
    tree = etree.parse(svg_file)
    for c in tree.getroot():
        if isinstance(c.tag, str) and local(c) not in NON_GRAPHIC \
                and c.get(INK + 'label') not in keep_labels:
            c.set('style', 'display:none')
    return tree


def visible_leaves(el, inherited=None):
    own = style_of(el)
    if hidden(own) or local(el) in NON_GRAPHIC:
        return
    merged = dict(inherited or {})
    merged.update(own)
    if local(el) in ('g', 'a', 'svg'):
        for c in el:
            if isinstance(c.tag, str):
                yield from visible_leaves(c, {k: v for k, v in merged.items() if k in INHERITED})
    else:
        yield el, merged


def unquote(v):
    return v.strip().strip('\'"').strip()


def requested_faces(style):
    family = unquote(style.get('font-family', '').split(',')[0])
    if not family or family.lower() in ('serif', 'sans-serif', 'monospace', 'cursive', 'fantasy'):
        raise PanelError(f'font-family {style.get("font-family")!r} does not name one font')
    css = style.get('font-weight', 'normal')
    css = {'normal': '400', 'bold': '700'}.get(css, css)
    if not css.isdigit():
        raise PanelError(f'font-weight {css!r} is relative')
    weights = {int(css)}
    slant = STYLE_SLANTS.get(style.get('font-style', 'normal'), 0)
    spec = unquote(style.get('-inkscape-font-specification', ''))
    if spec:
        spec_family, _, rest = spec.partition(',')
        if spec_family.strip().lower() != family.lower():
            raise PanelError(f'font specification {spec!r} names a different family '
                             f'from font-family {family!r}')
        for word in rest.split():
            w = word.lower()
            axis = re.fullmatch(r'@wght=([\d.]+)', w)
            if axis:
                weights.add(float(axis.group(1)))
            elif w in STYLE_WEIGHTS:
                weights.add(STYLE_WEIGHTS[w])
            elif w in STYLE_SLANTS:
                slant = STYLE_SLANTS[w]
            elif w not in STYLE_IGNORED:
                raise PanelError(f'font specification {spec!r} has unknown style {word!r}')
    return [(family, w, slant) for w in sorted(weights)]


def fontconfig_env(inkscape_bin):
    env = dict(os.environ)
    r = subprocess.run([inkscape_bin, '--system-data-directory'], capture_output=True, text=True)
    conf = os.path.normpath(os.path.join(r.stdout.strip(), '..', '..', 'etc', 'fonts', 'fonts.conf'))
    if r.returncode == 0 and os.path.exists(conf):
        env['FONTCONFIG_FILE'] = conf
    return env


def text_parts(el, inherited):
    style = dict(inherited)
    style.update(style_of(el))
    if hidden(style):
        return
    if (el.text or '').strip():
        yield style
    for c in el:
        if isinstance(c.tag, str):
            yield from text_parts(c, {k: v for k, v in style.items() if k in INHERITED})
            if (c.tail or '').strip():
                yield style


def check_fonts(svg_file, inkscape_bin):
    if not shutil.which('fc-list'):
        raise PanelError('fc-list is not installed, so the fonts cannot be checked')
    faces = set()
    for el, style in visible_leaves(etree.parse(svg_file).getroot()):
        if local(el) == 'text':
            for part_style in text_parts(el, style):
                faces.update(requested_faces(part_style))
    env = fontconfig_env(inkscape_bin)
    missing = []
    for family, weight, slant in sorted(faces):
        fc_weight = float(np.interp(weight, *CSS_TO_FC_WEIGHT))
        name = re.sub(r'([\\\-:,])', r'\\\1', family)
        r = subprocess.run(['fc-list', f'{name}:weight={fc_weight:g}:slant={slant}', 'family'],
                           capture_output=True, text=True, env=env)
        if r.returncode != 0 or not r.stdout.strip():
            missing.append(f'{family} weight {weight:g}' + (' italic' if slant else ''))
    if missing:
        raise PanelError('fonts not installed, Inkscape would substitute: ' + ', '.join(missing))
    return sorted(faces)


def check_cutouts_against_source(svg_file, cutout_labels, dxf_file, inkscape_bin, tmp, tol):
    tree = visible_tree(svg_file, cutout_labels)
    root = tree.getroot()
    leaves = []
    for c in root:
        if isinstance(c.tag, str) and c.get(INK + 'label') in cutout_labels:
            leaves += [el for el, _ in visible_leaves(c)]
    for n, el in enumerate(leaves):
        if not el.get('id'):
            el.set('id', f'panel2dxf-cutout-{n}')
    for g in root.iter():
        if isinstance(g.tag, str) and local(g) not in NON_GRAPHIC:
            d = style_of(g)
            if 'stroke' in d or g in leaves:
                d['stroke'] = 'none'
                write_style(g, d)
    src = os.path.join(tmp, 'cutouts-source.svg')
    tree.write(src)
    r = subprocess.run([inkscape_bin, '--query-all', src], capture_output=True, text=True)
    if r.returncode != 0:
        raise PanelError(f'inkscape --query-all failed\n{r.stderr}')
    boxes = {}
    for line in r.stdout.splitlines():
        parts = line.split(',')
        if len(parts) == 5:
            boxes[parts[0]] = [float(v) * MM_PER_PX for v in parts[1:]]
    page_h = float(re.match(r'\s*([-\d.eE+]+)\s*mm', root.get('height')).group(1))
    want = []
    for el in leaves:
        if el.get('id') not in boxes:
            raise PanelError(f'Inkscape reports no bounding box for cutout id={el.get("id")}')
        x, y, w, h = boxes[el.get('id')]
        if w > 0 or h > 0:
            want.append((el.get('id'), (x, page_h - y - h, x + w, page_h - y)))
    got, _ = dxf_layers(dxf_file, 'CUTOUT', 'SILKSCREEN')
    unmatched = [g.bounds for g in got]
    problems = []
    for ident, b in want:
        match = next((u for u in unmatched if max(abs(p - q) for p, q in zip(u, b)) <= tol), None)
        if match is None:
            problems.append(f'cutout id={ident} at {fmt(b)} in the source has no DXF shape there')
        else:
            unmatched.remove(match)
    for u in unmatched:
        problems.append(f'DXF cutout at {fmt(u)} matches no cutout in the source')
    if problems:
        raise PanelError('DXF cutouts do not match the source SVG:\n  ' + '\n  '.join(problems))


def ink_path(ink):
    verts, codes = [], []
    for g in getattr(ink, 'geoms', [ink]):
        if g.is_empty:
            continue
        g = orient(g, 1.0)
        for ring in [g.exterior] + list(g.interiors):
            c = np.asarray(ring.coords)
            verts += c.tolist()
            codes += [MplPath.MOVETO] + [MplPath.LINETO] * (len(c) - 2) + [MplPath.CLOSEPOLY]
    return MplPath(verts, codes) if verts else None


def raster_of(path, x0, y0, x1, y1, page_h):
    W, H = (x1 - x0) * RENDER_PX_PER_UNIT, (y1 - y0) * RENDER_PX_PER_UNIT
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    try:
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        fig.patch.set_facecolor('black')
        ax.set_xlim(x0 * MM_PER_PX, x1 * MM_PER_PX)
        ax.set_ylim(page_h - y1 * MM_PER_PX, page_h - y0 * MM_PER_PX)
        if path is not None:
            ax.add_patch(PathPatch(path, facecolor='white', edgecolor='none', antialiased=True))
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[..., :3].mean(axis=2) > 127
    finally:
        plt.close(fig)


def check_render(svg_file, silk_labels, dxf_file, inkscape_bin, tmp):
    tree = visible_tree(svg_file, silk_labels)
    root = tree.getroot()
    src = os.path.join(tmp, 'silkscreen-source.svg')
    tree.write(src)
    mm = lambda a: float(re.match(r'\s*([-\d.eE+]+)\s*mm', root.get(a)).group(1))
    page_w, page_h = mm('width'), mm('height')
    wpx, hpx = math.ceil(page_w / MM_PER_PX), math.ceil(page_h / MM_PER_PX)
    tiles = [(x, y, min(x + RENDER_TILE, wpx), min(y + RENDER_TILE, hpx))
             for y in range(0, hpx, RENDER_TILE) for x in range(0, wpx, RENDER_TILE)]
    acts = [f'export-dpi:{96 * RENDER_PX_PER_UNIT}', 'export-background:#000000',
            'export-background-opacity:1']
    for i, (x0, y0, x1, y1) in enumerate(tiles):
        acts += [f'export-area:{x0}:{y0}:{x1}:{y1}',
                 f'export-filename:{os.path.join(tmp, f"tile{i}.png")}', 'export-do']
    r = subprocess.run([inkscape_bin, '--actions=' + ';'.join(acts), src],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise PanelError(f'inkscape render failed\n{r.stderr}')
    _, ink = dxf_layers(dxf_file, 'CUTOUT', 'SILKSCREEN')
    path = ink_path(ink)
    problems = []
    for i, (x0, y0, x1, y1) in enumerate(tiles):
        png = os.path.join(tmp, f'tile{i}.png')
        if not os.path.exists(png):
            raise PanelError(f'inkscape did not render tile {i}')
        ref = imread(png)[..., :3].mean(axis=2) > 0.5
        got = raster_of(path, x0, y0, x1, y1, page_h)
        if ref.shape != got.shape:
            raise PanelError(f'tile {i} rendered at {ref.shape}, expected {got.shape}')
        bad = ndimage.binary_erosion(ref ^ got)
        labels, _ = ndimage.label(bad)
        for sl in ndimage.find_objects(labels):
            r0, r1 = sl[0].start, sl[0].stop
            c0, c1 = sl[1].start, sl[1].stop
            px = MM_PER_PX / RENDER_PX_PER_UNIT
            problems.append('x %.3f..%.3f y %.3f..%.3f mm' % (
                x0 * MM_PER_PX + c0 * px, x0 * MM_PER_PX + c1 * px,
                page_h - (y0 * MM_PER_PX + r1 * px), page_h - (y0 * MM_PER_PX + r0 * px)))
    if problems:
        raise PanelError('DXF silkscreen differs from Inkscape\'s rendering of the source at:\n  '
                         + '\n  '.join(problems[:20])
                         + (f'\n  ... and {len(problems) - 20} more' if len(problems) > 20 else ''))


def fmt(bounds):
    return 'x %.3f..%.3f y %.3f..%.3f mm' % (bounds[0], bounds[2], bounds[1], bounds[3])


def inkscape(binary, src, dst, actions):
    acts = ';'.join(actions + [f'export-filename:{dst}', 'export-overwrite', 'export-do'])
    r = subprocess.run([binary, f'--actions={acts}', src], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(dst):
        raise PanelError(f'inkscape failed\n{r.stderr}')
    return etree.parse(dst)


def prepare_artwork(svg, mapping, ink_colour, inkscape_bin, tmp):
    start, done = os.path.join(tmp, 'layers.svg'), os.path.join(tmp, 'artwork.svg')
    build_layers(svg, mapping).write(start)
    artwork = inkscape(inkscape_bin, start, done, ['select-all:no-groups', 'object-to-path'])
    regroup(artwork)
    for i in drop_degenerate(artwork):
        print(f'dropped empty path id={i}', flush=True)
    strip_strokes(artwork, {'cutout'})
    check_paintable(artwork, ink_colour)
    artwork.write(done)
    return artwork


def strokes_to_paths(artwork_file, layer_names, inkscape_bin, tmp):
    flat = inkscape(inkscape_bin, artwork_file, os.path.join(tmp, 'stroked.svg'),
                    ['select-all:no-groups', 'object-stroke-to-path'])
    regroup(flat)
    drop_degenerate(flat)
    check_flattened(flat, layer_names)
    return flat


def svg2dxf(flattened, prefix, ink_colour, tol):
    here = os.path.dirname(os.path.abspath(__file__))
    r = subprocess.run([sys.executable, os.path.join(here, 'svg2dxf.py'), flattened, prefix,
                        '--silk-outlines', 'none', '--ink-color', ink_colour, '--tol', repr(tol)],
                       capture_output=True, text=True)
    print(r.stdout, end='', flush=True)
    if r.returncode != 0:
        raise PanelError(f'svg2dxf failed\n{r.stderr}')


def convert(svg, prefix, mapping, ink_colour, tol, check_tol, inkscape_bin):
    outputs = ['-flattened.svg', '.dxf', '-outlines-only.dxf', '-silkscreen.pdf']
    cutout_labels = {k for k, v in mapping.items() if v == 'cutout'}
    silk_labels = {k for k, v in mapping.items() if v == 'silkscreen'}
    with tempfile.TemporaryDirectory() as tmp:
        staged = os.path.join(tmp, 'out')
        check_fonts(svg, inkscape_bin)
        artwork = prepare_artwork(svg, mapping, ink_colour, inkscape_bin, tmp)
        flat = strokes_to_paths(os.path.join(tmp, 'artwork.svg'), set(mapping.values()),
                                inkscape_bin, tmp)
        flat.write(staged + outputs[0], xml_declaration=True, encoding='UTF-8')
        svg2dxf(staged + outputs[0], staged, ink_colour, tol)
        check_dxf(artwork, staged + '.dxf', ink_colour, check_tol)
        check_cutouts_against_source(svg, cutout_labels, staged + '.dxf', inkscape_bin, tmp,
                                     CUTOUT_BBOX_TOL)
        check_render(svg, silk_labels, staged + '.dxf', inkscape_bin, tmp)
        for suffix in outputs:
            shutil.move(staged + suffix, prefix + suffix)
    return [prefix + s for s in outputs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('svg')
    ap.add_argument('prefix')
    ap.add_argument('--cutout', default='outline,edge-cuts')
    ap.add_argument('--silkscreen', default='silkscreen')
    ap.add_argument('--ink-color', default='#ffffff')
    ap.add_argument('--tol', type=float, default=0.0005, help='svg2dxf flattening tolerance in mm')
    ap.add_argument('--check-tol', type=float, default=0.015,
                    help='widest disagreement in mm allowed between the DXF and the source')
    ap.add_argument('--inkscape', default='inkscape')
    args = ap.parse_args()

    mapping = {lab: 'cutout' for lab in args.cutout.split(',')}
    mapping.update({lab: 'silkscreen' for lab in args.silkscreen.split(',')})
    try:
        written = convert(args.svg, args.prefix, mapping, colour(args.ink_color),
                          args.tol, args.check_tol, args.inkscape)
    except PanelError as e:
        sys.exit(f'panel2dxf: {e}')
    print('checked against the source artwork; wrote ' + ', '.join(written))


if __name__ == '__main__':
    main()
