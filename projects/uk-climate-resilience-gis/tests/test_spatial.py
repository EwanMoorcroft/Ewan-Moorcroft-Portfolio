"""Independent small-graph checks for spatial association and its null."""

import json
import unittest

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import sparse
from shapely.geometry import Polygon, box

from resilience.spatial import global_moran, spatial_diagnostics


def settings(permutations=199):
    return {"seed": 20260929, "moran_permutations": permutations}


def chain():
    return gpd.GeoDataFrame(
        {"lsoa21cd": ["A", "B", "C"], "score_balanced": [0.0, 1.0, 3.0]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)],
        crs=27700,
    )


class SpatialTests(unittest.TestCase):
    def test_chain_matches_hand_calculated_row_standardised_statistic_and_lag(self):
        table, diagnostic = spatial_diagnostics(chain(), settings())
        primary = diagnostic["primary"]
        self.assertAlmostEqual(primary["statistic"], -1 / 28)
        self.assertAlmostEqual(primary["expected"], -0.5)
        self.assertEqual(primary["n"], 3)
        self.assertEqual(primary["s0"], 3)
        self.assertEqual(primary["neighbour_counts"], {"A": 1, "B": 2, "C": 1})
        self.assertEqual(primary["row_weight_sums"], {"A": 1.0, "B": 1.0, "C": 1.0})
        np.testing.assert_allclose(table.z_score, np.array([-4, -1, 5]) / np.sqrt(14))
        np.testing.assert_allclose(table.spatial_lag_z, np.array([-1, 0.5, -1]) / np.sqrt(14))
        self.assertEqual(table.quadrant.tolist(), ["LL", "LH", "HL"])

    def test_queen_includes_corner_contacts_and_rook_requires_a_shared_edge(self):
        areas = gpd.GeoDataFrame(
            {"lsoa21cd": list("ABCD"), "score_balanced": [0, 0, 1, 1]},
            geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1),
                      box(0, 1, 1, 2), box(1, 1, 2, 2)], crs=27700,
        )
        _, diagnostic = spatial_diagnostics(areas, settings())
        self.assertEqual(diagnostic["primary"]["undirected_edges"], 6)
        self.assertEqual(diagnostic["rook_sensitivity"]["undirected_edges"], 4)
        self.assertAlmostEqual(diagnostic["primary"]["statistic"], -1 / 3)
        self.assertAlmostEqual(diagnostic["rook_sensitivity"]["statistic"], 0)
        self.assertAlmostEqual(diagnostic["primary"]["p_two_sided"], 1)
        self.assertAlmostEqual(diagnostic["primary"]["permutation_sd"], 0, places=12)

    def test_island_is_retained_in_n_without_a_fabricated_nearest_neighbour(self):
        areas = chain()
        areas.loc[2, "geometry"] = box(5, 0, 6, 1)
        table, diagnostic = spatial_diagnostics(areas, settings())
        primary = diagnostic["primary"]
        self.assertEqual(primary["n"], 3)
        self.assertEqual(primary["s0"], 2)
        self.assertEqual(primary["island_ids"], ["C"])
        self.assertEqual(primary["row_weight_sums"]["C"], 0)
        self.assertAlmostEqual(primary["statistic"], 2 / 7)
        self.assertAlmostEqual(primary["expected"], -0.5)
        self.assertEqual(table.quadrant.iloc[2], "isolated")
        self.assertTrue(np.isnan(table.spatial_lag_z.iloc[2]))
        self.assertTrue(np.isfinite(table.z_score.iloc[2]))

    def test_two_sided_tail_uses_analytic_expectation_and_plus_one_correction(self):
        table, diagnostic = spatial_diagnostics(chain(), settings(999))
        primary = diagnostic["primary"]
        distribution = np.asarray(primary["permutation_statistics"])
        # All six arrangements of [0, 1, 3] on a three-node path yield one of
        # these values. Only -1/28 is as far from E[I]=-1/2 as the observation.
        possible = np.array([-1 / 28, -4 / 7, -25 / 28])
        self.assertTrue(np.isclose(distribution[:, None], possible).any(axis=1).all())
        extreme = int(np.isclose(distribution, -1 / 28).sum())
        self.assertEqual(primary["extreme_permutations"], extreme)
        self.assertAlmostEqual(primary["p_two_sided"], (extreme + 1) / 1000)
        self.assertAlmostEqual(primary["permutation_mean"], -0.5, delta=0.04)
        self.assertGreater(primary["permutation_sd"], 0)
        self.assertNotIn("p_local", table)

    def test_fixed_seed_and_sorted_identifiers_make_every_result_row_order_invariant(self):
        areas = chain()
        areas.index = [3, 3, 9]
        first, diagnostic = spatial_diagnostics(areas, settings())
        repeated, repeated_diagnostic = spatial_diagnostics(areas, settings())
        shuffled, shuffled_diagnostic = spatial_diagnostics(areas.iloc[[2, 0, 1]], settings())
        pd.testing.assert_frame_equal(first, repeated)
        self.assertEqual(diagnostic, repeated_diagnostic)
        self.assertEqual(diagnostic, shuffled_diagnostic)
        self.assertEqual(shuffled.lsoa21cd.tolist(), ["C", "A", "B"])
        pd.testing.assert_frame_equal(
            first.sort_values("lsoa21cd").reset_index(drop=True),
            shuffled.sort_values("lsoa21cd").reset_index(drop=True),
        )
        json.dumps(diagnostic, allow_nan=False)

    def test_excluded_middle_polygon_does_not_create_a_new_edge_across_the_gap(self):
        for score, eligible in [(np.inf, True), (1, False), (None, True)]:
            with self.subTest(score=score, eligible=eligible):
                areas = chain()
                areas.loc[1, "score_balanced"] = score
                areas["model_eligible"] = [True, eligible, True]
                original = areas.copy(deep=True)
                table, diagnostic = spatial_diagnostics(areas, settings())
                pd.testing.assert_frame_equal(areas, original)
                self.assertEqual(table.spatial_eligible.tolist(), [True, False, True])
                self.assertTrue(np.isnan(table.z_score.iloc[1]))
                self.assertTrue(pd.isna(table.quadrant.iloc[1]))
                self.assertEqual(diagnostic["primary"]["island_ids"], ["A", "C"])
                self.assertEqual(diagnostic["primary"]["s0"], 0)
                self.assertIsNone(diagnostic["primary"]["statistic"])

    def test_zero_variance_does_not_manufacture_a_test_or_quadrant(self):
        areas = chain()
        areas["score_balanced"] = 7
        table, diagnostic = spatial_diagnostics(areas, settings())
        self.assertEqual(diagnostic["primary"]["status"], "zero_variance")
        self.assertIsNone(diagnostic["primary"]["statistic"])
        self.assertIsNone(diagnostic["primary"]["p_two_sided"])
        self.assertEqual(diagnostic["primary"]["permutation_statistics"], [])
        self.assertTrue(table.z_score.isna().all())
        self.assertTrue(table.quadrant.isna().all())
        json.dumps(diagnostic, allow_nan=False)

    def test_exact_mean_axis_is_not_mislabelled_as_high_or_low(self):
        areas = chain()
        areas["score_balanced"] = [0, 1, 2]
        table, diagnostic = spatial_diagnostics(areas, settings())
        self.assertTrue(table.quadrant.isna().all())
        self.assertEqual(diagnostic["quadrants"]["axis_count"], 3)

    def test_gaps_are_not_closed_by_snapping_or_buffering(self):
        areas = chain().iloc[:2].copy()
        areas.loc[1, "geometry"] = box(1.000001, 0, 2.000001, 1)
        table, diagnostic = spatial_diagnostics(areas, settings())
        self.assertEqual(diagnostic["primary"]["status"], "no_neighbours")
        self.assertEqual(table.quadrant.tolist(), ["isolated", "isolated"])

    def test_overlap_invalid_and_missing_scored_geometries_are_rejected(self):
        bad_geometries = [
            box(0.5, 0, 1.5, 1),
            Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)]),
            None,
        ]
        for geometry in bad_geometries:
            with self.subTest(geometry=geometry):
                areas = chain()
                areas.loc[1, "geometry"] = geometry
                with self.assertRaises(ValueError):
                    spatial_diagnostics(areas, settings())

    def test_duplicate_and_missing_ids_are_rejected(self):
        for ids in [["A", "A", "C"], ["A", None, "C"]]:
            with self.subTest(ids=ids):
                areas = chain()
                areas["lsoa21cd"] = ids
                with self.assertRaises(ValueError):
                    spatial_diagnostics(areas, settings())

    def test_empty_and_singleton_cohorts_have_explicit_undefined_statistics(self):
        for areas in [chain().iloc[:0], chain().iloc[:1]]:
            with self.subTest(n=len(areas)):
                table, diagnostic = spatial_diagnostics(areas, settings())
                self.assertEqual(len(table), len(areas))
                self.assertIsNone(diagnostic["primary"]["statistic"])
                self.assertEqual(diagnostic["primary"]["status"], "insufficient_units")
                json.dumps(diagnostic, allow_nan=False)

    def test_public_statistic_rejects_negative_nonfinite_and_self_weights(self):
        for weights in [
            [[0, -1], [1, 0]], [[0, np.nan], [1, 0]],
            [[1, 1], [1, 0]], [[0, 1, 0], [1, 0, 0]],
        ]:
            with self.subTest(weights=weights):
                with self.assertRaises(ValueError):
                    global_moran(np.array([0, 1]), sparse.csr_matrix(weights), 19, 1)

    def test_chunk_size_does_not_change_the_seeded_permutation_experiment(self):
        weights = sparse.csr_matrix([[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]])
        small = global_moran(np.array([0, 1, 3]), weights, 199, 9, chunk_size=7)
        large = global_moran(np.array([0, 1, 3]), weights, 199, 9, chunk_size=128)
        self.assertEqual(small, large)

    def test_invalid_permutation_settings_are_rejected(self):
        for overrides in [{"moran_permutations": 0}, {"moran_permutations": 3.5}, {"seed": -1}]:
            with self.subTest(overrides=overrides):
                config = settings()
                config.update(overrides)
                with self.assertRaises(ValueError):
                    spatial_diagnostics(chain(), config)


if __name__ == "__main__":
    unittest.main()
