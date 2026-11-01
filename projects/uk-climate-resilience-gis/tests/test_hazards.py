import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import geopandas as gpd
from shapely.geometry import box, mapping
from resilience.hazards import (
    FLOOD_API, COLLECTIONS, _page_chain, area_overlap, audit_flood_pages,
    inspect_flood_page,
)


class OverlayTests(unittest.TestCase):
    def test_overlapping_flood_features_do_not_double_count(self):
        areas = gpd.GeoDataFrame({'lsoa21cd': ['A', 'B']}, geometry=[box(0,0,100,100),box(100,0,200,100)], crs=27700)
        hazard = gpd.GeoDataFrame(geometry=[box(0,0,50,100),box(25,0,75,100)], crs=27700)
        result = area_overlap(areas, hazard)
        self.assertAlmostEqual(result['A'], .75)
        self.assertEqual(result['B'], 0)

    def test_geographic_area_calculation_is_rejected(self):
        areas = gpd.GeoDataFrame({'lsoa21cd':['A']}, geometry=[box(0,0,1,1)], crs=4326)
        with self.assertRaises(ValueError):
            area_overlap(areas, areas)

    def test_empty_complete_extent_is_zero_without_losing_neighbourhoods(self):
        areas = gpd.GeoDataFrame({'lsoa21cd': ['A', 'B']},
                                geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10)], crs=27700)
        hazard = gpd.GeoDataFrame(geometry=[], crs=27700)
        result = area_overlap(areas, hazard)
        self.assertEqual(result.to_dict(), {'A': 0.0, 'B': 0.0})

    def test_feature_extending_outside_study_is_clipped_before_fraction(self):
        areas = gpd.GeoDataFrame({'lsoa21cd': ['A']}, geometry=[box(0, 0, 10, 10)], crs=27700)
        hazard = gpd.GeoDataFrame(geometry=[box(-100, 0, 5, 100)], crs=27700)
        self.assertAlmostEqual(area_overlap(areas, hazard)['A'], 0.5)


def feature(identifier, shift=0):
    return {'type': 'Feature', 'id': identifier,
            'properties': {'label': 'An escaped "label" and café'},
            'geometry': mapping(box(shift, 0, shift + 1, 1))}


class FloodPaginationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def page(self, name, features, matched, next_url=None, returned=None):
        document = {'type': 'FeatureCollection', 'features': features,
                    'numberMatched': matched,
                    'numberReturned': len(features) if returned is None else returned,
                    'crs': {'type': 'name', 'properties': {'name': 'urn:ogc:def:crs:EPSG::4326'}}}
        if next_url:
            document['links'] = [{'rel': 'next', 'href': next_url}]
        path = self.root / name
        path.write_text(json.dumps(document, ensure_ascii=False))
        return path

    def test_unfollowed_next_link_is_a_real_incomplete_acquisition_error(self):
        first = self.page('first.geojson', [feature('A')], 2, 'https://example.org/page2')
        with self.assertRaisesRegex(ValueError, 'incomplete|next', msg='A first page must never count as complete'):
            audit_flood_pages([first])

    def test_complete_pages_reconcile_unique_features_and_report_duplicates(self):
        first = self.page('first.geojson', [feature('A'), feature('B', 2)], 3,
                          'https://example.org/page2')
        second = self.page('second.geojson', [feature('B', 2), feature('C', 4)], 3)
        audit = audit_flood_pages([first, second])
        self.assertEqual(audit['number_matched'], 3)
        self.assertEqual(audit['features_returned'], 4)
        self.assertEqual(audit['unique_features'], 3)
        self.assertEqual(audit['duplicate_features'], 1)
        self.assertEqual(audit['pages'][1]['new_feature_mask'], [False, True])
        self.assertEqual(len(audit['pages'][0]['sha256']), 64)

    def test_same_id_with_changed_content_cannot_be_silently_deduplicated(self):
        first = self.page('first.geojson', [feature('A')], 1, 'https://example.org/page2')
        second = self.page('second.geojson', [feature('A', 2)], 1)
        with self.assertRaisesRegex(ValueError, 'conflict|changed'):
            audit_flood_pages([first, second])

    def test_final_page_with_missing_records_is_rejected(self):
        page = self.page('first.geojson', [feature('A')], 2)
        with self.assertRaisesRegex(ValueError, 'reconcile|missing'):
            audit_flood_pages([page])

    def test_returned_count_must_match_actual_feature_array(self):
        page = self.page('first.geojson', [feature('A')], 2, returned=2)
        with self.assertRaisesRegex(ValueError, 'numberReturned|returned'):
            inspect_flood_page(page)

    def test_changed_matched_count_between_pages_is_rejected(self):
        first = self.page('first.geojson', [feature('A')], 2, 'https://example.org/page2')
        second = self.page('second.geojson', [feature('B')], 3)
        with self.assertRaisesRegex(ValueError, 'numberMatched|matched'):
            audit_flood_pages([first, second])

    def test_incremental_parser_handles_small_chunks_and_escaped_unicode(self):
        page = self.page('first.geojson', [feature('A'), feature('B', 3)], 2)
        info = inspect_flood_page(page, chunk_size=17)
        self.assertEqual(info['feature_ids'], ['A', 'B'])
        self.assertEqual(info['number_returned'], 2)

    def test_truncated_json_is_rejected(self):
        page = self.page('first.geojson', [feature('A')], 1)
        page.write_text(page.read_text()[:-7])
        with self.assertRaises(ValueError):
            inspect_flood_page(page)

    def test_missing_feature_geometry_cannot_masquerade_as_no_exposure(self):
        bad = feature('A')
        bad['geometry'] = None
        page = self.page('first.geojson', [bad], 1)
        with self.assertRaisesRegex(ValueError, 'geometry'):
            inspect_flood_page(page)

    def test_processing_cache_rejects_changed_source_checksum(self):
        raw = self.root / 'data/raw/ea'
        raw.mkdir(parents=True)
        source = self.page('fixture.geojson', [feature('A')], 1)
        page = raw / 'defended_1pct.geojson'
        source.replace(page)
        page.with_name(page.name + '.source.json').write_text(json.dumps({
            'source_url': FLOOD_API + '/' + COLLECTIONS['defended_1pct'] + '/items',
            'parameters': {'f': 'json', 'bbox': '0.0,0.0,1.0,1.0', 'limit': 10000},
            'sha256': '0' * 64,
        }))
        with patch('resilience.hazards.ROOT', self.root):
            with self.assertRaisesRegex(ValueError, 'checksum'):
                _page_chain('defended_1pct', [0.0, 0.0, 1.0, 1.0], acquire=False)
