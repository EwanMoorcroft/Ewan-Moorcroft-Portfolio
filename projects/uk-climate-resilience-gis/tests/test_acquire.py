"""Source-contract checks: missing areas must never masquerade as zero risk."""
import unittest

from resilience.acquire import combine_arcgis_pages, region_imd_codes


def polygon(code):
    return {
        "type": "Feature",
        "properties": {"LSOA21CD": code},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
        },
    }


class RegionSelectionTests(unittest.TestCase):
    def test_selects_only_requested_boroughs_by_exact_name(self):
        rows = [
            {"LSOA code (2021)": "E01000003", "Local Authority District name (2024)": "Salford"},
            {"LSOA code (2021)": "E01000001", "Local Authority District name (2024)": "Manchester"},
            {"LSOA code (2021)": "E01000002", "Local Authority District name (2024)": "Oldham"},
        ]
        self.assertEqual(region_imd_codes(rows, ["Manchester", "Salford"]), ["E01000001", "E01000003"])

    def test_missing_requested_borough_is_an_error(self):
        rows = [{"LSOA code (2021)": "E01000001", "Local Authority District name (2024)": "Manchester"}]
        with self.assertRaisesRegex(ValueError, "Salford"):
            region_imd_codes(rows, ["Manchester", "Salford"])

    def test_duplicate_lsoa_is_an_error(self):
        row = {"LSOA code (2021)": "E01000001", "Local Authority District name (2024)": "Manchester"}
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            region_imd_codes([row, row], ["Manchester"])


class ArcGISCoverageTests(unittest.TestCase):
    def test_combines_pages_and_preserves_source_geometry(self):
        first, second = polygon("E01000001"), polygon("E01000002")
        result = combine_arcgis_pages(
            [{"type": "FeatureCollection", "features": [first]},
             {"type": "FeatureCollection", "features": [second]}],
            "LSOA21CD", {"E01000001", "E01000002"},
        )
        self.assertEqual(result["features"], [first, second])
        self.assertEqual(result["type"], "FeatureCollection")

    def test_silent_missing_record_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "coverage"):
            combine_arcgis_pages(
                [{"type": "FeatureCollection", "features": [polygon("E01000001")]}],
                "LSOA21CD", {"E01000001", "E01000002"},
            )

    def test_overlapping_pages_are_rejected(self):
        page = {"type": "FeatureCollection", "features": [polygon("E01000001")]}
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            combine_arcgis_pages([page, page], "LSOA21CD", {"E01000001"})

    def test_partial_service_response_is_rejected(self):
        page = {"type": "FeatureCollection", "features": [polygon("E01000001")], "exceededTransferLimit": True}
        with self.assertRaisesRegex(ValueError, "transfer"):
            combine_arcgis_pages([page], "LSOA21CD", {"E01000001"})

    def test_api_error_cannot_be_treated_as_empty_data(self):
        with self.assertRaisesRegex(ValueError, "ArcGIS"):
            combine_arcgis_pages([{"error": {"code": 400, "message": "Invalid query"}}], "LSOA21CD", set())

    def test_missing_geometry_is_rejected(self):
        feature = polygon("E01000001")
        feature["geometry"] = None
        with self.assertRaisesRegex(ValueError, "geometry"):
            combine_arcgis_pages([{"type": "FeatureCollection", "features": [feature]}], "LSOA21CD", {"E01000001"})


if __name__ == "__main__":
    unittest.main()
