"""Geometric 1:1 check.  Builds the true ink region straight from the SVG at
0.2 um flattening, builds the region the DXF actually encodes, and compares
them by mis-inked area and by Hausdorff distance."""
import numpy as np, sys, ezdxf, math
from lxml import etree
from svgpathtools import parse_path
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
sys.path.insert(0,'.')
from svg2panel import parse_transform, apply, subpath_points
SVG='{http://www.w3.org/2000/svg}'; INK='{http://www.inkscape.org/namespaces/inkscape}'
H=128.5; FINE=0.0002

def truth():
    r=etree.parse('/mnt/user-data/uploads/mikado-panel-rev1.svg').getroot()
    ink=Polygon()
    for g in r.iter(SVG+'g'):
        if g.get(INK+'label')!='silksceren': continue
        for p in g.iter(SVG+'path'):
            st={k.split(':')[0].strip():k.split(':',1)[1].strip()
                for k in (p.get('style') or '').split(';') if ':' in k}
            fill=(st.get('fill') or p.get('fill') or '#000000').lower()
            M=parse_transform(p.get('transform')); reg=Polygon()
            for sp in parse_path(p.get('d')).continuous_subpaths():
                P=subpath_points(sp,FINE); P=apply(M,P); P[:,1]=H-P[:,1]
                if len(P)<3: continue
                q=Polygon(P)
                if not q.is_valid: q=q.buffer(0)
                if not q.is_empty: reg=reg.symmetric_difference(q)
            if reg.is_empty: continue
            ink = ink.union(reg) if fill in ('#ffffff','white') else ink.difference(reg)
            if not ink.is_valid: ink=ink.buffer(0)
    return ink

def bulge_arc(p0,p1,b,n=256):
    if abs(b)<1e-12: return [p0]
    th=4*math.atan(b); d=np.hypot(*(np.array(p1)-np.array(p0)))
    if d<1e-15: return [p0]
    r=d/(2*math.sin(abs(th)/2)); mx,my=(p0[0]+p1[0])/2,(p0[1]+p1[1])/2
    h=math.sqrt(max(r*r-(d/2)**2,0)); ux,uy=(p1[0]-p0[0])/d,(p1[1]-p0[1])/d
    sgn=1 if th>0 else -1
    cx,cy=mx-sgn*h*uy*np.sign(math.cos(th/2)) if False else (mx - sgn*uy*h*(1 if abs(th)<math.pi else -1),
                                                            my + sgn*ux*h*(1 if abs(th)<math.pi else -1))
    a0=math.atan2(p0[1]-cy,p0[0]-cx)
    return [(cx+r*math.cos(a0+th*i/n), cy+r*math.sin(a0+th*i/n)) for i in range(n)]

def got(fn):
    d=ezdxf.readfile(fn); islands=[]
    for e in d.modelspace():
        if e.dxftype()!='HATCH': continue
        rings=[]
        for bp in e.paths:
            vs=list(bp.vertices); pts=[]
            for i,v in enumerate(vs):
                p0=(v[0],v[1]); b=v[2] if len(v)>2 else 0.0
                p1=(vs[(i+1)%len(vs)][0], vs[(i+1)%len(vs)][1])
                pts.extend(bulge_arc(p0,p1,b) if abs(b)>1e-12 else [p0])
            if len(pts)>=3: rings.append((Polygon(pts), bp.path_type_flags))
        if not rings: continue
        outer=[g for g,f in rings if f & 1]          # EXTERNAL
        holes=[g for g,f in rings if not (f & 1)]
        if not outer: outer=[max((g for g,_ in rings), key=lambda g:g.area)]
        reg=unary_union([g.buffer(0) for g in outer])
        for h in holes: reg=reg.difference(h.buffer(0))
        islands.append(reg)
    return unary_union(islands)

T=truth(); G=got(sys.argv[1])
sd=T.symmetric_difference(G)
print('true ink area      %.6f mm^2'%T.area)
print('dxf  ink area      %.6f mm^2'%G.area)
print('mis-inked area     %.6f mm^2  (%.5f%% of ink)'%(sd.area,100*sd.area/T.area))
print('Hausdorff boundary %.6f mm'%T.boundary.hausdorff_distance(G.boundary))
