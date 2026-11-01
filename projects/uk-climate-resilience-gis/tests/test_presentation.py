"""Presentation safeguards for missing values, executable HTML and map geometry."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon

from resilience.presentation import (
    build_presentations, format_value, json_for_script, metric_scale, svg_path,
    _render_html, REQUIRED,
)


class FormattingTests(unittest.TestCase):
    def test_missing_numbers_are_not_rendered_as_zero_or_nan(self):
        for value in (None, float("nan"), float("inf"), -float("inf")):
            self.assertEqual(format_value(value, "percent"), "Not available")

    def test_fractional_rates_and_temperature_have_explicit_units(self):
        self.assertEqual(format_value(0, "percent"), "0.0%")
        self.assertEqual(format_value(0.125, "percent"), "12.5%")
        self.assertEqual(format_value(1.26, "celsius"), "+1.3 °C")
        self.assertEqual(format_value(-0.00001, "celsius"), "0.0 °C")
        self.assertEqual(format_value(1234, "count"), "1,234")

    def test_script_payload_cannot_close_script_and_restores_source_text(self):
        original = {"name": "</script><script>alert('x')</script> & café", "missing": np.nan}
        encoded = json_for_script(original)
        self.assertNotIn("</script", encoded)
        self.assertNotIn("<", encoded)
        self.assertEqual(json.loads(encoded), {"name": original["name"], "missing": None})

    def test_heat_scale_centres_zero_and_rates_use_fraction_units(self):
        self.assertEqual(metric_scale("heat_anomaly", [-1.2, 2.1, None]), [-2.5, 2.5])
        self.assertEqual(metric_scale("flood_exposure", [0, 0.4]), [0.0, 1.0])
        self.assertEqual(metric_scale("score_balanced", [30, 60]), [0.0, 100.0])

    def test_html_encodes_feature_labels_and_embeds_no_external_runtime(self):
        record = {field: 0 for field in REQUIRED}
        record.update(name='<img src=x onerror="alert(1)">', borough='Test & borough',
                      lsoa21cd='E01000001', model_eligible=True, robust_priority=False)
        frame = gpd.GeoDataFrame([record], geometry=[Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])], crs=27700)
        document = _render_html(frame, {'minimum_lsoa_heat_coverage': .8}, {
            'weight_experiment': {'samples': 3}, 'top_group': {'fraction': .2},
            'robustness': {'frequency_threshold': .8},
        })
        self.assertNotIn('<img src=x', document)
        self.assertIn('&lt;img src=x onerror=&quot;', document)
        self.assertNotIn('<script src=', document)
        self.assertNotIn('__DATA__', document)
        self.assertNotIn('__PATHS__', document)


class GeometryAndInputTests(unittest.TestCase):
    def test_svg_flips_northing_and_keeps_polygon_holes(self):
        polygon = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)],
                          holes=[[(25, 25), (75, 25), (75, 75), (25, 75)]])
        path = svg_path(polygon, 0, 100, tolerance=0)
        self.assertTrue(path.startswith("M0.0,100.0"))
        self.assertEqual(path.count("M"), 2)
        self.assertEqual(path.count("Z"), 2)
        self.assertIn("25.0,75.0", path)

    def test_absent_real_input_fails_before_any_output_is_created(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("resilience.presentation.ROOT", Path(directory)):
                with self.assertRaisesRegex(FileNotFoundError, "resilience.gpkg"):
                    build_presentations({})
            self.assertFalse((Path(directory) / "outputs").exists())


if __name__ == "__main__":
    unittest.main()
