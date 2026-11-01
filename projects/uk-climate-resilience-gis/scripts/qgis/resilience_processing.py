"""Reusable QGIS Processing algorithm for transparent five-dimension scores.

Load this file as a QGIS Processing script, or register ResilienceProvider in a
standalone PyQGIS application. It does not install a plugin or change a profile.
"""
from __future__ import annotations

import math

from qgis.core import (
    Qgis, QgsFeature, QgsFeatureSink, QgsField, QgsFields,
    QgsProcessingAlgorithm, QgsProcessingException, QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource, QgsProcessingParameterField,
    QgsProcessingParameterString, QgsProcessingProvider,
)
from qgis.PyQt.QtCore import QMetaType


DIMENSIONS = (
    ("FLOOD", "Flood exposure", "dim_flood"),
    ("HEAT", "Surface heat", "dim_heat"),
    ("ACCESS", "Green-space access deficit", "dim_access"),
    ("INCOME", "Income deprivation", "dim_income"),
    ("AGE", "Age sensitivity", "dim_age"),
)
OUTPUT_FIELDS = ("weighted_score", "score_valid", "score_issue")


def parse_weights(value: str) -> list[float]:
    """Validate five explicit weights and normalise their positive total."""
    try:
        weights = [float(part.strip()) for part in value.split(",")]
    except (TypeError, ValueError) as exc:
        raise QgsProcessingException("Weights must be five comma-separated numbers.") from exc
    if len(weights) != 5 or any(not math.isfinite(w) or w < 0 for w in weights):
        raise QgsProcessingException("Provide five finite, nonnegative weights in flood, heat, access, income, age order.")
    try:
        total = math.fsum(weights)
    except OverflowError as exc:
        raise QgsProcessingException("The sum of weights must be finite and greater than zero.") from exc
    if not math.isfinite(total) or total <= 0:
        raise QgsProcessingException("The sum of weights must be finite and greater than zero.")
    return [weight / total for weight in weights]


class WeightedNeighbourhoodScores(QgsProcessingAlgorithm):
    def name(self):
        return "weightedneighbourhoodscores"

    def displayName(self):
        return "Weighted neighbourhood scores (five dimensions)"

    def group(self):
        return "Climate resilience"

    def groupId(self):
        return "climateresilience"

    def createInstance(self):
        return WeightedNeighbourhoodScores()

    def shortHelpString(self):
        return (
            "Combine five distinct, already-normalised numeric fields on a 0–100 scale, "
            "where higher values are adverse. Weights are supplied in flood, heat, green-space "
            "access deficit, income deprivation, age sensitivity order and are divided by their sum. "
            "This algorithm does not rank, rescale raw indicators, or estimate intervention benefits.\n\n"
            "Input attributes, geometry, CRS and feature count are retained. Outputs add weighted_score, "
            "score_valid and score_issue. A missing, nonfinite or out-of-range value in any positively "
            "weighted field produces a NULL score, false validity and named invalid fields; remaining "
            "weights are never reallocated around missing data. Zero-weight fields are ignored. "
            "Five different numeric fields are required. Existing output field names are rejected."
        )

    def initAlgorithm(self, configuration=None):
        self.addParameter(QgsProcessingParameterFeatureSource(
            "INPUT", "Neighbourhoods with normalised dimensions",
            [Qgis.ProcessingSourceType.VectorAnyGeometry],
        ))
        for key, label, default in DIMENSIONS:
            self.addParameter(QgsProcessingParameterField(
                key, f"{label} field (0–100)", defaultValue=default,
                parentLayerParameterName="INPUT",
                type=QgsProcessingParameterField.DataType.Numeric,
            ))
        self.addParameter(QgsProcessingParameterString(
            "WEIGHTS", "Weights: flood, heat, access, income, age",
            defaultValue="0.2,0.2,0.2,0.2,0.2",
        ))
        self.addParameter(QgsProcessingParameterFeatureSink("OUTPUT", "Scored neighbourhoods"))

    def checkParameterValues(self, parameters, context):
        valid, message = super().checkParameterValues(parameters, context)
        if not valid:
            return valid, message
        try:
            parse_weights(self.parameterAsString(parameters, "WEIGHTS", context))
        except QgsProcessingException as exc:
            return False, str(exc)
        selected = [self.parameterAsString(parameters, key, context) for key, _, _ in DIMENSIONS]
        if len(set(selected)) != 5:
            return False, "Select five different dimension fields; duplicate indicators would be double-counted."
        return True, ""

    def processAlgorithm(self, parameters, context, feedback):
        weights = parse_weights(self.parameterAsString(parameters, "WEIGHTS", context))
        source = self.parameterAsSource(parameters, "INPUT", context)
        if source is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, "INPUT"))
        selected = [self.parameterAsString(parameters, key, context) for key, _, _ in DIMENSIONS]
        if len(set(selected)) != 5:
            raise QgsProcessingException("Select five different dimension fields.")
        for field in selected:
            index = source.fields().indexFromName(field)
            if index < 0 or not source.fields()[index].isNumeric():
                raise QgsProcessingException(f"Selected dimension is missing or nonnumeric: {field}")
        conflicts = set(OUTPUT_FIELDS) & set(source.fields().names())
        if conflicts:
            raise QgsProcessingException("Output fields already exist: " + ", ".join(sorted(conflicts)))
        fields = QgsFields(source.fields())
        fields.append(QgsField("weighted_score", QMetaType.Type.Double, len=20, prec=8))
        fields.append(QgsField("score_valid", QMetaType.Type.Bool))
        fields.append(QgsField("score_issue", QMetaType.Type.QString))
        sink, destination = self.parameterAsSink(
            parameters, "OUTPUT", context, fields, source.wkbType(), source.sourceCrs(),
        )
        if sink is None:
            raise QgsProcessingException(self.invalidSinkError(parameters, "OUTPUT"))
        count = source.featureCount()
        invalid_count = 0
        written = 0
        for index, feature in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break
            contributions, invalid = [], []
            for field, weight in zip(selected, weights):
                if weight == 0:
                    continue
                try:
                    value = float(feature[field])
                except (TypeError, ValueError, OverflowError):
                    invalid.append(field)
                    continue
                if not math.isfinite(value) or not 0 <= value <= 100:
                    invalid.append(field)
                else:
                    contributions.append(value * weight)
            score = None if invalid else math.fsum(contributions)
            invalid_count += bool(invalid)
            output = QgsFeature(fields)
            output.setGeometry(feature.geometry())
            output.setAttributes(feature.attributes() + [score, not invalid, ", ".join(invalid)])
            if not sink.addFeature(output, QgsFeatureSink.Flag.FastInsert):
                raise QgsProcessingException("Could not write an output feature.")
            written += 1
            if count > 0:
                feedback.setProgress(100 * (index + 1) / count)
        sink.finalize()
        feedback.pushInfo("Normalised weights: " + ", ".join(f"{w:.6g}" for w in weights))
        feedback.pushInfo(f"Wrote {written} features; {invalid_count} have NULL scores from invalid active dimensions.")
        return {"OUTPUT": destination}


class ResilienceProvider(QgsProcessingProvider):
    def id(self):
        return "resilience"

    def name(self):
        return "Climate resilience"

    def longName(self):
        return "Transparent neighbourhood adaptation screening"

    def loadAlgorithms(self):
        self.addAlgorithm(WeightedNeighbourhoodScores())
