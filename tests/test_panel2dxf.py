import math, os, tempfile, unittest
from unittest import mock
import ezdxf
from ezdxf import bbox
from lxml import etree
import panel2dxf as p

SVG_NS = 'http://www.w3.org/2000/svg'
INK_NS = 'http://www.inkscape.org/namespaces/inkscape'
MAPPING = {'edge-cuts': 'cutout', 'silkscreen': 'silkscreen'}

PANEL = f'''<svg xmlns="{SVG_NS}" xmlns:inkscape="{INK_NS}"
     width="40mm" height="20mm" viewBox="0 0 40 20">
  <g inkscape:label="edge-cuts">
    <rect id="outline" x="0" y="0" width="40" height="20" style="fill:#000000"/>
    <circle id="hole" cx="5" cy="10" r="2" style="fill:#ffffff;stroke:#000000;stroke-width:0.1"/>
    <g transform="translate(36,4) rotate(45)">
      <rect id="slot" x="-1" y="-0.5" width="2" height="1" style="fill:#ffffff"/>
    </g>
  </g>
  <g inkscape:label="silkscreen">
    <rect id="pill" x="10" y="5" width="20" height="6" rx="1" style="fill:#ffffff"/>
    <path id="wave" d="M 13,8 C 15,6 17,10 19,8 S 23,6 25,8"
          style="fill:none;stroke:#000000;stroke-width:0.6;stroke-linecap:round;stroke-linejoin:round"/>
    <path id="bar" d="M 14,9.5 h 3 v 0.8 h -3 z"
          style="fill:#000000;stroke:#000000;stroke-width:0.04;stroke-linejoin:round"/>
    <path id="rule" d="M 10,15 H 30"
          style="fill:none;stroke:#ffffff;stroke-width:0.3;stroke-linecap:round"/>
    <g id="moved" transform="translate(32,2) rotate(90)" style="stroke-width:0.5">
      <path id="tick" d="M 0,0 H 3" style="fill:none;stroke:#ffffff;stroke-linecap:butt"/>
    </g>
    <g style="display:none">
      <rect id="hidden" x="1" y="1" width="5" height="5" style="fill:#ffffff"/>
    </g>
    <rect id="under" x="32" y="12" width="6" height="5" style="fill:#ffffff"/>
    <rect id="knock" x="33" y="13" width="4" height="3" style="fill:#000000"/>
    <rect id="island" x="34" y="14" width="2" height="1" style="fill:#ffffff"/>
    <text id="label" x="12" y="18.5"
          style="font-family:'Ubuntu Mono';font-weight:bold;font-size:2.8px;fill:#ffffff;
                 -inkscape-font-specification:'Ubuntu Mono, Bold'">SYS</text>
  </g>
</svg>'''


def path(d, style, transform=None, ident='p'):
    el = etree.Element(f'{{{SVG_NS}}}path', d=d, style=style, id=ident)
    if transform:
        el.set('transform', transform)
    return el


def layer_tree(*paths, label='silkscreen'):
    root = etree.Element(f'{{{SVG_NS}}}svg', nsmap={None: SVG_NS, 'inkscape': INK_NS})
    layer = etree.SubElement(root, f'{{{SVG_NS}}}g')
    layer.set(f'{{{INK_NS}}}groupmode', 'layer')
    layer.set(f'{{{INK_NS}}}label', label)
    layer.extend(paths)
    return etree.ElementTree(root)


def write(directory, name, text):
    f = os.path.join(directory, name)
    with open(f, 'w') as fh:
        fh.write(text)
    return f


def panel_with(body, width='40mm', height='20mm', viewbox='0 0 40 20'):
    return (f'<svg xmlns="{SVG_NS}" xmlns:inkscape="{INK_NS}" width="{width}" '
            f'height="{height}" viewBox="{viewbox}">{body}</svg>')


def ink_bounds(dxf):
    _, ink = p.dxf_layers(dxf, 'CUTOUT', 'SILKSCREEN')
    return [g.bounds for g in getattr(ink, 'geoms', [ink])]


def has_bounds(dxf, want, tol=0.002):
    return any(max(abs(a - b) for a, b in zip(got, want)) <= tol for got in ink_bounds(dxf))


class StrokeGeometry(unittest.TestCase):
    def test_round_capped_line_is_length_times_width_plus_one_disc(self):
        r = p.element_region(path('M 0,0 H 10', 'fill:none;stroke:#fff;stroke-width:1;'
                                                'stroke-linecap:round'))
        self.assertAlmostEqual(r.area, 10 + math.pi * 0.25, delta=0.005)

    def test_butt_capped_line_is_length_times_width(self):
        r = p.element_region(path('M 0,0 H 10', 'fill:none;stroke:#fff;stroke-width:1'))
        self.assertAlmostEqual(r.area, 10.0, places=9)

    def test_square_capped_line_adds_half_width_at_each_end(self):
        r = p.element_region(path('M 0,0 H 10', 'fill:none;stroke:#fff;stroke-width:1;'
                                                'stroke-linecap:square'))
        self.assertAlmostEqual(r.area, 11.0, places=9)

    def test_filled_square_grows_by_half_the_stroke_width(self):
        r = p.element_region(path('M 0,0 H 10 V 10 H 0 Z', 'fill:#000;stroke:#000;'
                                                          'stroke-width:1;stroke-linejoin:round'))
        self.assertAlmostEqual(r.area, 100 + 4 * 10 * 0.5 + math.pi * 0.25, delta=0.005)
        self.assertEqual(tuple(round(v, 9) for v in r.bounds), (-0.5, -0.5, 10.5, 10.5))

    def test_stroke_scales_with_a_uniform_transform(self):
        r = p.element_region(path('M 0,0 H 10', 'fill:none;stroke:#fff;stroke-width:1',
                                  'matrix(0,2,-2,0,5,5)'))
        self.assertAlmostEqual(r.area, 20 * 2, places=9)
        self.assertEqual(tuple(round(v, 9) for v in r.bounds), (4.0, 5.0, 6.0, 25.0))


class FillRule(unittest.TestCase):
    OUTER = 'M 0,0 H 10 V 10 H 0 Z '
    SAME_WAY = 'M 2,2 H 8 V 8 H 2 Z'
    OTHER_WAY = 'M 2,2 V 8 H 8 V 2 Z'

    def area(self, d, rule):
        return p.element_region(path(d, f'fill:#000;fill-rule:{rule}')).area

    def test_evenodd_punches_a_hole_whatever_the_winding(self):
        self.assertAlmostEqual(self.area(self.OUTER + self.SAME_WAY, 'evenodd'), 64)
        self.assertAlmostEqual(self.area(self.OUTER + self.OTHER_WAY, 'evenodd'), 64)

    def test_nonzero_fills_a_same_way_inner_ring(self):
        self.assertAlmostEqual(self.area(self.OUTER + self.SAME_WAY, 'nonzero'), 100)

    def test_nonzero_punches_an_opposite_way_inner_ring(self):
        self.assertAlmostEqual(self.area(self.OUTER + self.OTHER_WAY, 'nonzero'), 64)

    def test_self_crossing_figure_eight_keeps_both_lobes(self):
        d = 'M 0,0 L 4,4 L 4,0 L 0,4 Z'
        self.assertAlmostEqual(self.area(d, 'nonzero'), 8)
        self.assertAlmostEqual(self.area(d, 'evenodd'), 8)


class Flattening(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def flatten(self, body, **page):
        svg = write(self.tmp.name, 'in.svg', panel_with(body, **page))
        tree = p.build_layers(svg, {'silkscreen': 'silkscreen'})
        return list(p.layers_of(tree.getroot())['silkscreen'])

    def test_nested_transforms_compose_outer_first(self):
        [el] = self.flatten('<g inkscape:label="silkscreen" transform="translate(10,0)">'
                            '<g transform="rotate(90)"><g transform="scale(2)">'
                            '<path d="M 0,0 H 1 V 1 Z" style="fill:#fff"/></g></g></g>')
        r = p.element_region(el)
        self.assertEqual(tuple(round(v, 9) for v in r.bounds), (8.0, 0.0, 10.0, 2.0))

    def test_group_style_is_inherited_and_the_child_wins(self):
        a, b = self.flatten('<g inkscape:label="silkscreen" style="stroke:#fff;stroke-width:0.5">'
                            '<path id="a" d="M 0,0 H 1" style="fill:none"/>'
                            '<path id="b" d="M 0,0 H 1" style="fill:none;stroke-width:0.25"/></g>')
        self.assertEqual(p.style_of(a)['stroke-width'], '0.5')
        self.assertEqual(p.style_of(b)['stroke-width'], '0.25')
        self.assertEqual(p.style_of(b)['stroke'], '#fff')

    def test_non_inherited_group_properties_do_not_leak(self):
        [el] = self.flatten('<g inkscape:label="silkscreen" style="opacity:1;display:inline">'
                            '<path d="M 0,0 H 1 V 1 Z" style="fill:#fff"/></g>')
        self.assertNotIn('display', p.style_of(el))

    def test_hidden_groups_and_shapes_are_dropped(self):
        els = self.flatten('<g inkscape:label="silkscreen">'
                           '<g style="display:none"><path id="g" d="M 0,0 H 1 V 1 Z"/></g>'
                           '<path id="h" d="M 0,0 H 1 V 1 Z" style="visibility:hidden"/>'
                           '<path id="v" d="M 0,0 H 1 V 1 Z" style="fill:#fff"/></g>')
        self.assertEqual([e.get('id') for e in els], ['v'])

    def test_drawing_order_survives_nesting(self):
        els = self.flatten('<g inkscape:label="silkscreen"><path id="1" d="M 0,0 H 1 V 1 Z"/>'
                           '<g><path id="2" d="M 0,0 H 1 V 1 Z"/><g><path id="3" '
                           'd="M 0,0 H 1 V 1 Z"/></g><path id="4" d="M 0,0 H 1 V 1 Z"/></g>'
                           '<path id="5" d="M 0,0 H 1 V 1 Z"/></g>')
        self.assertEqual([e.get('id') for e in els], ['1', '2', '3', '4', '5'])

    def test_unassigned_top_level_group_is_refused(self):
        with self.assertRaises(p.PanelError) as cm:
            self.flatten('<g inkscape:label="silkscreen"/><g inkscape:label="notes"/>')
        self.assertIn("'notes'", str(cm.exception))

    def test_unsupported_elements_and_effects_are_refused(self):
        for body in ['<use href="#x"/>', '<image href="x.png" width="1" height="1"/>',
                     '<path d="M 0,0 H 1" clip-path="url(#c)"/>',
                     '<path d="M 0,0 H 1" style="mask:url(#m)"/>',
                     '<g style="filter:url(#f)"><path d="M 0,0 H 1"/></g>',
                     '<path d="M 0,0 H 1" style="marker-end:url(#a)"/>']:
            with self.subTest(body=body), self.assertRaises(p.PanelError):
                self.flatten(f'<g inkscape:label="silkscreen">{body}</g>')

    def test_page_mapping_handles_an_offset_scaled_viewbox(self):
        svg = write(self.tmp.name, 'in.svg',
                    panel_with('', width='80mm', height='40mm', viewbox='10 5 40 20'))
        a, b, d, e, xoff, yoff = p.page(etree.parse(svg).getroot())
        to_dxf = lambda x, y: (a * x + b * y + xoff, d * x + e * y + yoff)
        self.assertEqual(to_dxf(10, 5), (0.0, 40.0))
        self.assertEqual(to_dxf(50, 25), (80.0, 0.0))


class Refusals(unittest.TestCase):
    def refused(self, style, transform=None, label='silkscreen'):
        tree = layer_tree(path('M 0,0 H 10', style, transform), label=label)
        with self.assertRaises(p.PanelError) as cm:
            p.check_paintable(tree, '#ffffff')
        return str(cm.exception)

    def test_dashed_stroke(self):
        self.assertIn('dashed', self.refused('fill:none;stroke:#fff;stroke-dasharray:1,1'))

    def test_stroke_under_non_uniform_scale(self):
        self.assertIn('non-uniform', self.refused('fill:none;stroke:#fff', 'scale(1,2)'))

    def test_fill_and_stroke_in_different_colours(self):
        self.assertIn('fill #ffffff and stroke #000000', self.refused('fill:#fff;stroke:#000'))

    def test_colour_that_is_neither_ink_nor_black(self):
        self.assertIn('painted #808080', self.refused('fill:#808080'))

    def test_gradient_fill(self):
        self.assertIn('painted url(#g)', self.refused('fill:url(#g)'))

    def test_partial_opacity(self):
        self.assertIn('fill-opacity', self.refused('fill:#fff;fill-opacity:0.5'))

    def test_shape_with_neither_fill_nor_stroke(self):
        self.assertIn('neither fill nor stroke', self.refused('fill:none;stroke:none'))

    def test_accepts_white_and_black(self):
        p.check_paintable(layer_tree(path('M 0,0 H 1 V 1 Z', 'fill:white'),
                                     path('M 0,0 H 1 V 1 Z', 'fill:#000')), '#ffffff')


class FontRequests(unittest.TestCase):
    def test_bold_from_css_and_specification(self):
        self.assertEqual(p.requested_faces({'font-family': "'Ubuntu Mono'", 'font-weight': 'bold',
                                            '-inkscape-font-specification': "'Ubuntu Mono, Bold'"}),
                         [('Ubuntu Mono', 700, 0)])

    def test_variable_axis_weight_is_checked_alongside_css_weight(self):
        self.assertEqual(p.requested_faces({'font-family': 'Roboto Mono', 'font-weight': 'normal',
                                            '-inkscape-font-specification':
                                                "'Roboto Mono, @wght=300'"}),
                         [('Roboto Mono', 300, 0), ('Roboto Mono', 400, 0)])

    def test_italic(self):
        self.assertEqual(p.requested_faces({'font-family': 'X', 'font-style': 'italic'}),
                         [('X', 400, 100)])

    def test_generic_family_is_refused(self):
        with self.assertRaises(p.PanelError):
            p.requested_faces({'font-family': 'sans-serif'})

    def test_relative_weight_is_refused(self):
        with self.assertRaises(p.PanelError):
            p.requested_faces({'font-family': 'X', 'font-weight': 'bolder'})

    def test_specification_naming_another_family_is_refused(self):
        with self.assertRaises(p.PanelError):
            p.requested_faces({'font-family': 'X', '-inkscape-font-specification': "'Y, Bold'"})

    def test_unknown_style_word_is_refused(self):
        with self.assertRaises(p.PanelError):
            p.requested_faces({'font-family': 'X', '-inkscape-font-specification': "'X, Wobbly'"})


class InstalledFonts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def check(self, style):
        svg = write(self.tmp.name, 'in.svg', panel_with(
            f'<g inkscape:label="silkscreen"><text style="{style}">A</text></g>'))
        return p.check_fonts(svg, 'inkscape')

    def test_installed_face_passes(self):
        self.assertEqual(self.check("font-family:'Ubuntu Mono';font-weight:bold"),
                         [('Ubuntu Mono', 700, 0)])

    def test_missing_family_is_refused(self):
        with self.assertRaises(p.PanelError) as cm:
            self.check('font-family:NoSuchFontAnywhere')
        self.assertIn('NoSuchFontAnywhere weight 400', str(cm.exception))

    def test_missing_weight_of_an_installed_family_is_refused(self):
        with self.assertRaises(p.PanelError) as cm:
            self.check("font-family:'Ubuntu Mono';font-weight:300")
        self.assertIn('Ubuntu Mono weight 300', str(cm.exception))

    def test_tspan_face_is_checked(self):
        svg = write(self.tmp.name, 'in.svg', panel_with(
            '<g inkscape:label="silkscreen"><text style="font-family:\'Ubuntu Mono\'">A'
            '<tspan style="font-family:NoSuchFontAnywhere">B</tspan></text></g>'))
        with self.assertRaises(p.PanelError):
            p.check_fonts(svg, 'inkscape')

    def test_hidden_text_is_not_checked(self):
        self.assertEqual(self.check('font-family:NoSuchFontAnywhere;display:none'), [])


class Pipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.svg = write(self.dir, 'panel.svg', PANEL)

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, mutate_flat=None, mutate_dxf=None):
        artwork = p.prepare_artwork(self.svg, MAPPING, '#ffffff', 'inkscape', self.dir)
        flat = p.strokes_to_paths(os.path.join(self.dir, 'artwork.svg'),
                                  set(MAPPING.values()), 'inkscape', self.dir)
        if mutate_flat:
            mutate_flat(flat, artwork)
        flat.write(os.path.join(self.dir, 'out-flattened.svg'))
        p.svg2dxf(os.path.join(self.dir, 'out-flattened.svg'), os.path.join(self.dir, 'out'),
                  '#ffffff', 0.0005)
        dxf = os.path.join(self.dir, 'out.dxf')
        if mutate_dxf:
            doc = ezdxf.readfile(dxf)
            mutate_dxf(doc.modelspace())
            doc.saveas(dxf)
        return artwork, dxf

    def vector(self, artwork, dxf):
        p.check_dxf(artwork, dxf, '#ffffff', 0.015)

    def cutouts(self, dxf):
        p.check_cutouts_against_source(self.svg, {'edge-cuts'}, dxf, 'inkscape', self.dir,
                                       p.CUTOUT_BBOX_TOL)

    def render(self, dxf):
        p.check_render(self.svg, {'silkscreen'}, dxf, 'inkscape', self.dir)

    def assert_all_pass(self, artwork, dxf):
        self.vector(artwork, dxf)
        self.cutouts(dxf)
        self.render(dxf)


class GoodPanel(Pipeline):
    def test_convert_writes_every_output(self):
        written = p.convert(self.svg, os.path.join(self.dir, 'final'), MAPPING, '#ffffff',
                            0.0005, 0.015, 'inkscape')
        self.assertEqual([os.path.basename(f) for f in written],
                         ['final-flattened.svg', 'final.dxf', 'final-outlines-only.dxf',
                          'final-silkscreen.pdf'])
        for f in written:
            self.assertGreater(os.path.getsize(f), 0, f)

    def test_every_check_passes(self):
        self.assert_all_pass(*self.build())

    def test_cutout_strokes_do_not_grow_the_hole(self):
        _, dxf = self.build()
        circles = list(ezdxf.readfile(dxf).modelspace().query('CIRCLE'))
        self.assertEqual(len(circles), 1)
        self.assertAlmostEqual(circles[0].dxf.radius, 2.0, places=6)
        self.assertAlmostEqual(circles[0].dxf.center.x, 5.0, places=6)
        self.assertAlmostEqual(circles[0].dxf.center.y, 10.0, places=6)

    def test_white_stroke_becomes_ink_of_its_full_width(self):
        _, dxf = self.build()
        self.assertTrue(has_bounds(dxf, (9.85, 4.85, 30.15, 5.15)))

    def test_inherited_width_and_nested_transform_reach_the_dxf(self):
        _, dxf = self.build()
        self.assertTrue(has_bounds(dxf, (31.75, 15.0, 32.25, 18.0)))

    def test_hidden_artwork_is_absent(self):
        _, dxf = self.build()
        self.assertFalse([b for b in ink_bounds(dxf) if b[0] < 7 and b[3] > 13])

    def test_knockout_with_an_island_keeps_drawing_order(self):
        _, dxf = self.build()
        _, ink = p.dxf_layers(dxf, 'CUTOUT', 'SILKSCREEN')
        self.assertTrue(ink.contains(p.Point(32.5, 3.5)))
        self.assertFalse(ink.contains(p.Point(33.5, 4.5)))
        self.assertTrue(ink.contains(p.Point(35.0, 5.5)))

    def test_offset_scaled_viewbox_converts_and_checks(self):
        svg = write(self.dir, 'offset.svg', panel_with(
            '<g inkscape:label="edge-cuts"><rect x="10" y="5" width="40" height="20" '
            'style="fill:#000"/><circle cx="20" cy="15" r="2" style="fill:#fff"/></g>'
            '<g inkscape:label="silkscreen"><path d="M 30,15 H 45" '
            'style="fill:none;stroke:#fff;stroke-width:0.3"/></g>',
            width='80mm', height='40mm', viewbox='10 5 40 20'))
        prefix = os.path.join(self.dir, 'offset')
        p.convert(svg, prefix, MAPPING, '#ffffff', 0.0005, 0.015, 'inkscape')
        circles = list(ezdxf.readfile(prefix + '.dxf').modelspace().query('CIRCLE'))
        self.assertAlmostEqual(circles[0].dxf.radius, 4.0, places=6)
        self.assertAlmostEqual(circles[0].dxf.center.x, 20.0, places=6)
        self.assertAlmostEqual(circles[0].dxf.center.y, 20.0, places=6)
        self.assertTrue(has_bounds(prefix + '.dxf', (40.0, 19.7, 70.0, 20.3)))


class FaultsInOwnFlattening(Pipeline):
    def test_lost_transform_is_caught_only_by_the_source_checks(self):
        with mock.patch.object(p, 'matrix_attr', lambda M: 'matrix(1,0,0,1,0,0)'):
            artwork, dxf = self.build()
        self.vector(artwork, dxf)
        with self.assertRaises(p.PanelError):
            self.cutouts(dxf)
        with self.assertRaises(p.PanelError):
            self.render(dxf)

    def test_reversed_drawing_order_is_caught_by_the_render_check(self):
        def reverse(flat, artwork):
            for tree in (flat, artwork):
                layer = p.layers_of(tree.getroot())['silkscreen']
                layer[:] = list(reversed(list(layer)))
        artwork, dxf = self.build(mutate_flat=reverse)
        self.vector(artwork, dxf)
        with self.assertRaises(p.PanelError):
            self.render(dxf)

    def test_ignoring_hidden_is_caught_by_the_render_check(self):
        with mock.patch.object(p, 'hidden', lambda d: False):
            artwork = p.prepare_artwork(self.svg, MAPPING, '#ffffff', 'inkscape', self.dir)
            flat = p.strokes_to_paths(os.path.join(self.dir, 'artwork.svg'),
                                      set(MAPPING.values()), 'inkscape', self.dir)
        flat.write(os.path.join(self.dir, 'h-flattened.svg'))
        p.svg2dxf(os.path.join(self.dir, 'h-flattened.svg'), os.path.join(self.dir, 'h'),
                  '#ffffff', 0.0005)
        dxf = os.path.join(self.dir, 'h.dxf')
        self.vector(artwork, dxf)
        with self.assertRaises(p.PanelError) as cm:
            self.render(dxf)
        self.assertIn('x 1.0', str(cm.exception))

    def test_lost_style_inheritance_is_caught_by_the_render_check(self):
        with mock.patch.object(p, 'INHERITED', p.INHERITED - {'stroke-width'}):
            artwork, dxf = self.build()
        self.vector(artwork, dxf)
        with self.assertRaises(p.PanelError):
            self.render(dxf)


class FaultsAfterInkscape(Pipeline):
    def drop(self, ident):
        def mutate(flat, artwork):
            for el in list(flat.iter()):
                if el.get('id') == ident:
                    el.getparent().remove(el)
        return mutate

    def test_lost_stroke_is_caught_by_vector_and_render_checks(self):
        artwork, dxf = self.build(mutate_flat=self.drop('rule'))
        with self.assertRaises(p.PanelError) as cm:
            self.vector(artwork, dxf)
        self.assertIn('x 9.850..30.150', str(cm.exception))
        with self.assertRaises(p.PanelError):
            self.render(dxf)

    def test_knockout_missing_its_0_04_stroke_is_caught_by_the_vector_check(self):
        artwork, dxf = self.build()
        for el in artwork.iter():
            if el.get('id') == 'bar':
                d = p.style_of(el)
                d['stroke'] = 'none'
                p.write_style(el, d)
        with self.assertRaises(p.PanelError):
            self.vector(artwork, dxf)


class FaultsInTheDxf(Pipeline):
    def test_hole_0_01_mm_too_big_is_caught_by_the_source_cutout_check(self):
        def grow(msp):
            for c in msp.query('CIRCLE'):
                c.dxf.radius += 0.01
        artwork, dxf = self.build(mutate_dxf=grow)
        with self.assertRaises(p.PanelError) as cm:
            self.cutouts(dxf)
        self.assertIn('id=hole', str(cm.exception))

    def test_missing_cutout_is_caught_by_both_cutout_checks(self):
        def remove(msp):
            for c in msp.query('CIRCLE'):
                msp.delete_entity(c)
        artwork, dxf = self.build(mutate_dxf=remove)
        with self.assertRaises(p.PanelError):
            self.vector(artwork, dxf)
        with self.assertRaises(p.PanelError):
            self.cutouts(dxf)

    def test_missing_silkscreen_shape_is_caught_by_vector_and_render_checks(self):
        def remove(msp):
            for h in msp.query('HATCH'):
                b = bbox.extents([h])
                if b.extmin.y > 4.8 and b.extmax.y < 5.2:
                    msp.delete_entity(h)
        artwork, dxf = self.build(mutate_dxf=remove)
        with self.assertRaises(p.PanelError):
            self.vector(artwork, dxf)
        with self.assertRaises(p.PanelError):
            self.render(dxf)


class NothingWrittenOnFailure(Pipeline):
    def test_failed_check_leaves_no_output(self):
        prefix = os.path.join(self.dir, 'final')
        with mock.patch.object(p, 'hidden', lambda d: False):
            with self.assertRaises(p.PanelError):
                p.convert(self.svg, prefix, MAPPING, '#ffffff', 0.0005, 0.015, 'inkscape')
        self.assertFalse([f for f in os.listdir(self.dir) if f.startswith('final')])


if __name__ == '__main__':
    unittest.main()
