# svg2dxf / svg2panel

1. flatten the svg objects into layers and convert everything to paths
2. run the svg2panel.py. See file header for detail
3. view the output in [dxf-viewer](https://github.com/vagran/dxf-viewer)

`panel2dxf.py panel.svg out_prefix` does steps 1 and 2 in one go: it flattens the
panel into `cutout` and `silkscreen` layers, has Inkscape convert every object and
every silkscreen stroke to filled paths, and runs svg2dxf.py with
`--silk-outlines none`. Nothing is written unless every check passes:

- every font the text uses is installed in exactly the requested weight and style
- the DXF matches a shapely model of the artwork (strokes as outlines buffered by
  half their width), to `--check-tol` (default 0.015 mm)
- every cutout in the source, measured by Inkscape, has one DXF shape within 0.005 mm
- the DXF silkscreen matches Inkscape's own rendering of the source at 0.0102 mm per
  pixel; any disagreement 3 pixels wide or more fails

It needs `inkscape` and `fc-list` on the PATH.

Tests: `venv/bin/python -m unittest discover -s tests -t .` (needs Inkscape and
the Ubuntu Mono font).
