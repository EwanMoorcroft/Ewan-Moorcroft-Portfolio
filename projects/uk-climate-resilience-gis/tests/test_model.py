"""Decision-model contracts with hand-calculated and adversarial fixtures."""

import copy
import json
import unittest

import numpy as np
import pandas as pd

from resilience.model import fit_model


INDICATORS = [
    "flood_exposure", "heat_anomaly", "access_deficit", "income_rate",
    "age_sensitive_share",
]
DIMENSIONS = ["dim_flood", "dim_heat", "dim_access", "dim_income", "dim_age"]
SCENARIOS = ["balanced", "hazard", "equity"]


def configuration():
    return {
        "indicators": INDICATORS.copy(),
        "scenarios": {
            "balanced": [0.2, 0.2, 0.2, 0.2, 0.2],
            "hazard": [0.3, 0.3, 0.1, 0.15, 0.15],
            "equity": [0.15, 0.15, 0.15, 0.3, 0.25],
        },
        "seed": 20260929,
        "top_fraction": 0.2,
        "sensitivity_samples": 31,
        "dirichlet_concentration": 50,
        "robust_frequency_threshold": 0.8,
    }


def observations(rows):
    return pd.DataFrame(rows, columns=["lsoa21cd"] + INDICATORS)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.config = configuration()
        self.frame = observations([
            ["A", 0, 100, 0, 0, 100],
            ["B", 10, 50, 10, 10, 50],
            ["C", 20, 0, 20, 20, 0],
        ])

    def test_each_scenario_uses_its_configured_weights_and_descending_ranks(self):
        result, diagnostics = fit_model(self.frame, self.config)
        np.testing.assert_allclose(result[DIMENSIONS], [
            [0, 100, 0, 0, 100], [50, 50, 50, 50, 50],
            [100, 0, 100, 100, 0],
        ])
        np.testing.assert_allclose(result.score_balanced, [40, 50, 60])
        np.testing.assert_allclose(result.score_hazard, [45, 50, 55])
        np.testing.assert_allclose(result.score_equity, [40, 50, 60])
        for scenario in SCENARIOS:
            np.testing.assert_allclose(result["rank_" + scenario], [3, 2, 1])
        self.assertEqual(result.scenario_top_count.tolist(), [0, 0, 3])
        self.assertEqual(diagnostics["cohort"]["eligible_count"], 3)

    def test_midrank_ties_and_inclusive_cutoff_keep_equal_neighbourhoods_equal(self):
        frame = observations([[str(i), value, value, value, value, value]
                              for i, value in enumerate([0, 0, 5, 10, 10])])
        result, diagnostics = fit_model(frame, self.config)
        np.testing.assert_allclose(result.dim_flood, [0, 0, 50, 100, 100])
        np.testing.assert_allclose(result.rank_balanced, [4.5, 4.5, 3, 1.5, 1.5])
        np.testing.assert_allclose(result.top20_frequency, [0, 0, 0, 1, 1])
        self.assertEqual(result.robust_priority.tolist(), [False, False, False, True, True])
        self.assertEqual(diagnostics["scenarios"]["balanced"]["top_count"], 2)
        self.assertEqual(diagnostics["scenarios"]["balanced"]["top_threshold"], 100)

    def test_nonfinite_and_missing_rows_do_not_change_the_complete_case_cohort(self):
        extra = observations([
            ["D", 9999, np.nan, 9999, 9999, 9999],
            ["E", np.inf, 1, 1, 1, 1],
            ["F", 1, 1, -np.inf, 1, 1],
            ["G", 1, 1, 1, None, 1],
            ["H", 1, 1, 1, 1, "unknown"],
        ])
        frame = pd.concat([self.frame, extra], ignore_index=True)
        original = frame.copy(deep=True)
        result, diagnostics = fit_model(frame, self.config)
        pd.testing.assert_frame_equal(frame, original)
        self.assertEqual(result.lsoa21cd.tolist(), list("ABCDEFGH"))
        self.assertEqual(result.model_eligible.tolist(), [True] * 3 + [False] * 5)
        np.testing.assert_allclose(result.loc[:2, "dim_flood"], [0, 50, 100])
        unscored_columns = DIMENSIONS + ["score_" + x for x in SCENARIOS]
        unscored_columns += ["rank_" + x for x in SCENARIOS]
        unscored_columns += ["scenario_top_count", "top20_frequency", "rank_p05",
                             "rank_p95", "rank_span_fraction"]
        self.assertTrue(result.loc[3:, unscored_columns].isna().all().all())
        self.assertFalse(result.loc[3:, "robust_priority"].any())
        self.assertIn("heat_anomaly", result.loc[3, "model_missing_indicators"])
        self.assertEqual(diagnostics["cohort"]["excluded_count"], 5)

    def test_constant_dimension_is_neutral_without_reweighting_other_dimensions(self):
        self.frame["heat_anomaly"] = 6
        result, diagnostics = fit_model(self.frame, self.config)
        np.testing.assert_allclose(result.dim_heat, [50, 50, 50])
        np.testing.assert_allclose(result.score_balanced, [30, 50, 70])
        self.assertTrue(diagnostics["normalization"]["dimensions"]["heat_anomaly"]["constant"])
        self.assertIsNone(diagnostics["indicator_spearman"]["heat_anomaly"]["income_rate"])

    def test_all_tied_scores_are_explicitly_reported_as_non_discriminating(self):
        frame = observations([[str(i), 7, 7, 7, 7, 7] for i in range(5)])
        result, diagnostics = fit_model(frame, self.config)
        np.testing.assert_allclose(result[DIMENSIONS], 50)
        np.testing.assert_allclose(result.score_balanced, 50)
        np.testing.assert_allclose(result.rank_balanced, 3)
        np.testing.assert_allclose(result.rank_span_fraction, 0)
        self.assertTrue(diagnostics["scenarios"]["balanced"]["all_scores_tied"])
        self.assertEqual(diagnostics["scenarios"]["balanced"]["top_count"], 5)

    def test_robust_priority_requires_the_top_group_in_all_three_scenarios(self):
        frame = observations([
            ["A", 100, 100, 0, 0, 0],
            ["B", 0, 0, 0, 100, 100],
            ["C", 50, 50, 100, 50, 50],
        ])
        self.config["robust_frequency_threshold"] = 0
        result, _ = fit_model(frame, self.config)
        self.assertEqual(result.scenario_top_count.tolist(), [1, 0, 2])
        self.assertFalse(result.robust_priority.any())

    def test_monotone_dominance_has_certain_experiment_selection_and_zero_rank_span(self):
        frame = observations([[str(i), i, i, i, i, i] for i in range(5)])
        self.config["robust_frequency_threshold"] = 1
        result, _ = fit_model(frame, self.config)
        np.testing.assert_allclose(result.top20_frequency, [0, 0, 0, 0, 1])
        np.testing.assert_allclose(result.rank_p05, [5, 4, 3, 2, 1])
        np.testing.assert_allclose(result.rank_p95, [5, 4, 3, 2, 1])
        np.testing.assert_allclose(result.rank_span_fraction, 0)
        self.assertEqual(result.robust_priority.tolist(), [False] * 4 + [True])

    def test_alternative_normalisation_and_leave_one_out_report_real_tie_changes(self):
        _, diagnostics = fit_model(self.frame, self.config)
        alt = diagnostics["alternative_minmax"]["balanced"]
        self.assertAlmostEqual(alt["spearman"], 1)
        self.assertAlmostEqual(alt["top_jaccard"], 1)
        omitted = diagnostics["leave_one_dimension_out"]["flood_exposure"]["balanced"]
        self.assertIsNone(omitted["spearman"])
        self.assertEqual(omitted["top_count"], 3)
        self.assertAlmostEqual(omitted["top_jaccard"], 1 / 3)
        self.assertAlmostEqual(diagnostics["indicator_spearman"]["flood_exposure"]["heat_anomaly"], -1)
        self.assertAlmostEqual(diagnostics["scenario_rank_spearman"]["balanced"]["equity"], 1)

    def test_seeded_experiment_and_diagnostics_are_invariant_to_input_row_order(self):
        self.frame.index = [9, 9, 2]
        first, diagnostics = fit_model(self.frame, self.config)
        repeat, repeated_diagnostics = fit_model(self.frame, self.config)
        shuffled_frame = self.frame.iloc[[2, 0, 1]]
        shuffled, shuffled_diagnostics = fit_model(shuffled_frame, self.config)
        pd.testing.assert_frame_equal(first, repeat)
        self.assertEqual(shuffled.index.tolist(), [2, 9, 9])
        self.assertEqual(shuffled.lsoa21cd.tolist(), ["C", "A", "B"])
        pd.testing.assert_frame_equal(
            first.sort_values("lsoa21cd").reset_index(drop=True),
            shuffled.sort_values("lsoa21cd").reset_index(drop=True),
        )
        self.assertEqual(diagnostics, repeated_diagnostics)
        self.assertEqual(diagnostics, shuffled_diagnostics)
        json.dumps(diagnostics, allow_nan=False)

    def test_all_2000_weight_draws_and_integer_allocation_are_auditable(self):
        self.config["sensitivity_samples"] = 2000
        _, diagnostics = fit_model(self.frame, self.config)
        experiment = diagnostics["weight_experiment"]
        self.assertEqual(experiment["draws_by_scenario"], {"balanced": 667, "hazard": 667, "equity": 666})
        weights = np.asarray(experiment["weight_draws"])
        self.assertEqual(weights.shape, (2000, 5))
        self.assertTrue((weights > 0).all())
        np.testing.assert_allclose(weights.sum(axis=1), 1)
        np.testing.assert_allclose(experiment["alpha_by_scenario"]["balanced"], [10] * 5)
        self.assertEqual(experiment["seed"], 20260929)
        self.assertEqual(experiment["concentration"], 50)

    def test_invalid_weights_fail_instead_of_silently_changing_the_policy(self):
        invalid = [
            [0.2] * 4,
            [0.3] * 5,
            [-0.1, 0.3, 0.3, 0.3, 0.2],
            [np.nan, 0.2, 0.2, 0.2, 0.2],
            [0] * 5,
        ]
        for weights in invalid:
            with self.subTest(weights=weights):
                config = copy.deepcopy(self.config)
                config["scenarios"]["balanced"] = weights
                with self.assertRaises(ValueError):
                    fit_model(self.frame, config)

    def test_zero_policy_weights_stay_zero_during_dirichlet_experiment(self):
        self.config["scenarios"] = {name: [1, 0, 0, 0, 0] for name in SCENARIOS}
        result, diagnostics = fit_model(self.frame, self.config)
        np.testing.assert_allclose(result.score_balanced, [0, 50, 100])
        np.testing.assert_allclose(diagnostics["weight_experiment"]["weight_draws"],
                                   np.tile([1, 0, 0, 0, 0], (31, 1)))
        omitted = diagnostics["leave_one_dimension_out"]["flood_exposure"]["balanced"]
        self.assertFalse(omitted["assessed"])

    def test_duplicate_or_missing_identifiers_are_rejected(self):
        for ids in [["A", "A", "C"], ["A", None, "C"], ["A", " ", "C"]]:
            with self.subTest(ids=ids):
                frame = self.frame.copy()
                frame["lsoa21cd"] = ids
                with self.assertRaises(ValueError):
                    fit_model(frame, self.config)

    def test_invalid_experiment_settings_are_rejected(self):
        for key, value in [
            ("top_fraction", 0), ("top_fraction", 1.1),
            ("sensitivity_samples", 2), ("sensitivity_samples", 3.5),
            ("dirichlet_concentration", 0), ("dirichlet_concentration", np.inf),
            ("robust_frequency_threshold", -0.1), ("robust_frequency_threshold", np.nan),
            ("seed", -1),
        ]:
            with self.subTest(key=key, value=value):
                config = copy.deepcopy(self.config)
                config[key] = value
                with self.assertRaises(ValueError):
                    fit_model(self.frame, config)

    def test_empty_and_entirely_unobserved_cohorts_return_no_artificial_scores(self):
        for frame in [self.frame.iloc[:0], observations([["A", None, None, None, None, None]])]:
            with self.subTest(rows=len(frame)):
                result, diagnostics = fit_model(frame, self.config)
                self.assertEqual(len(result), len(frame))
                self.assertFalse(result.model_eligible.any())
                self.assertTrue(result.score_balanced.isna().all())
                self.assertEqual(diagnostics["cohort"]["eligible_count"], 0)
                json.dumps(diagnostics, allow_nan=False)

    def test_single_complete_neighbourhood_has_defined_rank_span(self):
        result, _ = fit_model(self.frame.iloc[:1], self.config)
        self.assertEqual(result.rank_p05.iloc[0], 1)
        self.assertEqual(result.rank_p95.iloc[0], 1)
        self.assertEqual(result.rank_span_fraction.iloc[0], 0)


if __name__ == "__main__":
    unittest.main()
