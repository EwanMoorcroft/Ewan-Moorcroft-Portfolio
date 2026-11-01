"""Exercise the actual Processing algorithm with hand-calculated scores."""
import math
import os
import unittest

from qgis.core import (
    QgsApplication, QgsFeature, QgsField, QgsGeometry,
    QgsProcessingException, QgsVariantUtils, QgsVectorLayer,
)
from qgis.PyQt.QtCore import QMetaType

from scripts.qgis.resilience_processing import ResilienceProvider


class WeightedNeighbourhoodAlgorithmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QgsApplication.instance()
        if cls.app is None:
            cls.app = QgsApplication([], False, os.environ.get("QGIS_CUSTOM_CONFIG_PATH", ""))
            cls.app.initQgis()
        from processing.core.Processing import Processing
        Processing.initialize()
        cls.provider = ResilienceProvider()
        QgsApplication.processingRegistry().addProvider(cls.provider)

    @classmethod
    def tearDownClass(cls):
        QgsApplication.processingRegistry().removeProvider(cls.provider)

    def source(self, values, extra_field=False):
        layer = QgsVectorLayer("Point?crs=EPSG:27700", "hand_known", "memory")
        fields = [QgsField("name", QMetaType.Type.QString)]
        fields += [QgsField(name, QMetaType.Type.Double) for name in ("flood", "heat", "access", "income", "age")]
        if extra_field:
            fields.append(QgsField("weighted_score", QMetaType.Type.Double))
        layer.dataProvider().addAttributes(fields)
        layer.updateFields()
        for index, row in enumerate(values):
            feature = QgsFeature(layer.fields())
            attributes = [f"Area {index}"] + list(row)
            if extra_field:
                attributes.append(0)
            feature.setAttributes(attributes)
            feature.setGeometry(QgsGeometry.fromWkt(f"POINT({335000+index} 390000)"))
            layer.dataProvider().addFeatures([feature])
        return layer

    def run_algorithm(self, layer, weights="1,1,1,1,1", **overrides):
        import processing
        parameters = {
            "INPUT": layer, "FLOOD": "flood", "HEAT": "heat", "ACCESS": "access",
            "INCOME": "income", "AGE": "age", "WEIGHTS": weights, "OUTPUT": "memory:",
        }
        parameters.update(overrides)
        return processing.run("resilience:weightedneighbourhoodscores", parameters)["OUTPUT"]

    def test_unequal_weights_match_hand_calculation_and_preserve_geometry(self):
        source = self.source([[0, 25, 50, 75, 100]])
        result = self.run_algorithm(source, "1,2,3,4,5")
        feature = next(result.getFeatures())
        self.assertAlmostEqual(feature["weighted_score"], 1000 / 15)
        self.assertTrue(feature["score_valid"])
        self.assertEqual(feature["score_issue"], "")
        self.assertEqual(feature["name"], "Area 0")
        self.assertEqual(feature.geometry().asWkt(), next(source.getFeatures()).geometry().asWkt())
        self.assertEqual(result.crs().authid(), "EPSG:27700")

    def test_missing_active_dimension_preserves_feature_with_null_score(self):
        result = self.run_algorithm(self.source([[10, None, 30, 40, 50], [10, 20, 30, 40, 50]]))
        features = list(result.getFeatures())
        self.assertEqual(len(features), 2)
        self.assertTrue(QgsVariantUtils.isNull(features[0]["weighted_score"]))
        self.assertFalse(features[0]["score_valid"])
        self.assertIn("heat", features[0]["score_issue"])
        self.assertAlmostEqual(features[1]["weighted_score"], 30)

    def test_zero_weight_dimension_does_not_make_a_score_missing(self):
        result = self.run_algorithm(self.source([[10, None, None, None, None]]), "1,0,0,0,0")
        feature = next(result.getFeatures())
        self.assertAlmostEqual(feature["weighted_score"], 10)
        self.assertTrue(feature["score_valid"])

    def test_out_of_range_and_nonfinite_active_values_are_not_scored(self):
        for value in [-1, 101, math.inf]:
            with self.subTest(value=value):
                feature = next(self.run_algorithm(self.source([[value, 20, 30, 40, 50]])).getFeatures())
                self.assertTrue(QgsVariantUtils.isNull(feature["weighted_score"]))
                self.assertFalse(feature["score_valid"])

    def test_invalid_weights_are_rejected(self):
        source = self.source([[10, 20, 30, 40, 50]])
        for weights in ["1,2,3,4", "1,-1,1,1,1", "0,0,0,0,0", "nan,1,1,1,1", "inf,1,1,1,1", "x,1,1,1,1", "1e308,1e308,1e308,1e308,1e308"]:
            with self.subTest(weights=weights), self.assertRaises(QgsProcessingException):
                self.run_algorithm(source, weights)

    def test_duplicate_dimensions_and_output_name_conflicts_are_rejected(self):
        with self.assertRaises(QgsProcessingException):
            self.run_algorithm(self.source([[10, 20, 30, 40, 50]]), HEAT="flood")
        with self.assertRaises(QgsProcessingException):
            self.run_algorithm(self.source([[10, 20, 30, 40, 50]], extra_field=True))


if __name__ == "__main__":
    unittest.main()
