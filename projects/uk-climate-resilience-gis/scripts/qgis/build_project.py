"""Build the portable Greater Manchester QGIS project and six A3 map layouts.

Run through scripts/run_python.sh. This consumes processed evidence only and
never creates demonstration data when an input is missing. Source captions can
be overridden with --notes config/map_notes.json without changing map logic.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import shutil
import tempfile
import zipfile
from xml.etree import ElementTree

import numpy as np
from osgeo import gdal
from qgis.core import (
    Qgis, QgsApplication, QgsCategorizedSymbolRenderer, QgsCoordinateReferenceSystem,
    QgsExpression, QgsExpressionContext, QgsExpressionContextUtils,
    QgsFillSymbol, QgsLayerMetadata, QgsLayerTreeModel, QgsLayoutExporter,
    QgsLayoutItemLabel, QgsLayoutItemLegend, QgsLayoutItemMap, QgsLayoutItemPage,
    QgsLayoutItemScaleBar, QgsLayoutItemShape, QgsLayoutPoint, QgsLayoutSize,
    QgsMapThemeCollection, QgsMarkerSymbol, QgsPalLayerSettings, QgsPrintLayout,
    QgsProject, QgsRasterLayer, QgsRasterShader, QgsColorRampShader,
    QgsSingleBandPseudoColorRenderer, QgsRectangle, QgsReferencedRectangle,
    QgsRendererCategory, QgsRuleBasedRenderer, QgsSingleSymbolRenderer,
    QgsTextBufferSettings, QgsTextFormat, QgsVariantUtils, QgsVectorLayer,
    QgsVectorLayerSimpleLabeling,
)
from qgis.PyQt.QtCore import QDateTime, Qt
from qgis.PyQt.QtGui import QColor, QFont, QFontDatabase, QFontInfo

ROOT = Path(__file__).resolve().parents[2]
PROJECT_PATH = ROOT / "qgis/project/gm_climate_resilience.qgz"
REPORT_PATH = ROOT / "metadata/qgis_build.json"
STYLE_DIR = ROOT / "qgis/styles"
MAP_DIR = ROOT / "outputs/maps"
INK, TEAL, OCHRE = "#263f48", "#2d746e", "#b27b35"
PAPER, MUTED, LINE = "#fafbf9", "#586c70", "#cbd5d2"
MISSING = "#e1e5e2"
CRS = None  # Initialised after QgsApplication, avoiding pre-application CRS lookups.
LAYOUT_NAMES = ["01_overview", "02_flood", "03_heat", "04_accessibility", "05_priorities", "06_robustness"]

# Short editorial source text only. Exact dates/licences belong to retained
# acquisition sidecars. An optional JSON file may override any of these strings.
DEFAULT_MAP_NOTES = {
    "edition": "GREATER MANCHESTER / ADAPTATION SCREENING",
    "source_line": "ONS 2021 · EA present-day extents (2026) · selected USGS Landsat dates · Defra 2024 (corrected 2025) · MHCLG IoD2025 · OS/OSM 2026.\n© OpenStreetMap contributors / ODbL. Source licences and exact versions: project source register.",
    "portfolio_note": "Independent portfolio case study • not an official GMCA product",
    "overview_limit": "Relative indicator positions identify questions for investigation; they do not establish intervention suitability or benefit.",
    "flood_limit": "LSOA polygon-area share in the defended 1% river / 0.5% sea scenario; retained inland water is included. It is not a count of people affected.",
    "heat_limit": "Clear-sky daytime land-surface temperature is not air temperature or individual heat dose. Low-coverage estimates are excluded.",
    "access_limit": "2026 park case: OA population-weighted share beyond 1 km or with no mapped route to parks ≥10 ha. This differs from the official 2024 household measure.",
    "priority_limit": "Scores compare five dimensions within this study. A high score calls for appraisal; it is not a benefit, cost or delivery assessment.",
    "robustness_limit": "Selection frequency measures sensitivity to sampled weights. It is not a probability that a neighbourhood is at risk or that the model is correct.",
}

ALIASES = {
    "lsoa21cd": "LSOA 2021 code", "name": "Neighbourhood name", "borough": "Borough",
    "population": "Resident population", "income_rate": "Income deprivation rate",
    "age_sensitive_share": "Age-sensitive population share", "flood_exposure": "Defended flood footprint / neighbourhood area",
    "flood_01pct": "0.1% alternative footprint / neighbourhood area",
    "heat_anomaly": "Surface-temperature anomaly (°C)", "heat_coverage": "Valid surface-heat coverage (share)",
    "access_deficit": "Official 2024 household access deficit (Scenario 5)",
    "case_network_deficit": "2026 population-weighted park walking deficit (share)", "model_eligible": "Complete evidence for scoring",
    "dim_flood": "Flood dimension (0–100)", "dim_heat": "Surface-heat dimension (0–100)",
    "dim_access": "Access dimension (0–100)", "dim_income": "Income dimension (0–100)",
    "dim_age": "Age dimension (0–100)", "score_balanced": "Equal-weight score (0–100)",
    "score_hazard": "Hazard-emphasis score (0–100)", "score_equity": "Social-emphasis score (0–100)",
    "rank_balanced": "Equal-weight rank (1 = highest score)", "rank_hazard": "Hazard-emphasis rank",
    "rank_equity": "Social-emphasis rank", "scenario_top_count": "Scenarios selecting this area",
    "top20_frequency": "Highest-fifth selection frequency", "rank_p05": "5th percentile rank",
    "rank_p95": "95th percentile rank", "rank_span_fraction": "Rank interval as share of comparison set",
    "robust_priority": "Stable priority under configured rule", "function": "Green-space function",
    "area_ha": "Green-space area (ha)", "site_id": "Park identifier", "origin_id": "Population origin",
    "eligible": "Qualifies as a routing destination",
    "distance_m": "Shortest mapped walk to eligible park (m)", "snap_m": "Origin-to-network snap (m)",
    "known": "Origin meets network snapping rules", "reachable": "Mapped route to an eligible park exists", "hazard": "Flood hazard category",
}
REQUIRED_FIELDS = {
    "lsoa21cd", "name", "borough", "population", "model_eligible",
    "flood_exposure", "heat_anomaly", "heat_coverage", "access_deficit", "income_rate", "age_sensitive_share",
    "dim_flood", "dim_heat", "dim_access", "dim_income", "dim_age",
    "score_balanced", "rank_balanced", "top20_frequency", "robust_priority",
}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256_file(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def normalize_layout_sources(project_path, project):
    """QGIS 4.2 writes absolute fallback sources inside layout layer references."""
    with zipfile.ZipFile(project_path) as archive:
        entries = [(item, archive.read(item.filename)) for item in archive.infolist()]
    changed = 0
    rewritten = []
    for item, content in entries:
        if item.filename.endswith(".qgs"):
            xml = ElementTree.fromstring(content)
            for node in xml.findall(".//Layouts//Layer[@source]"):
                source = node.attrib["source"]
                relative = project.pathResolver().writePath(source)
                if Path(relative.split("|", 1)[0]).is_absolute():
                    raise ValueError(f"Layout source cannot be made portable: {source}")
                changed += relative != source
                node.set("source", relative)
            content = ElementTree.tostring(xml, encoding="utf-8", xml_declaration=True)
        rewritten.append((item, content))
    temporary = project_path.with_suffix(".qgz.part")
    with zipfile.ZipFile(temporary, "w") as archive:
        for item, content in rewritten:
            archive.writestr(item, content)
    temporary.replace(project_path)
    return changed


def clean(value):
    return None if QgsVariantUtils.isNull(value) else value


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def truth(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def values(rows, field):
    return [value for row in rows if (value := number(row.get(field))) is not None]


def percent(value, digits=0):
    return "—" if value is None else f"{100 * value:.{digits}f}%"


def text_format(font, size=10, color=INK, bold=False):
    fmt = QgsTextFormat()
    face = QFont(font)
    face.setBold(bold)
    fmt.setFont(face)
    fmt.setSize(size)
    fmt.setColor(QColor(color))
    return fmt


def add_label(layout, text, x, y, width, height, font, size=10, color=INK, bold=False):
    label = QgsLayoutItemLabel(layout)
    label.setText(str(text))
    label.setTextFormat(text_format(font, size, color, bold))
    label.setMargin(0)
    layout.addLayoutItem(label)
    label.attemptMove(QgsLayoutPoint(x, y))
    label.attemptResize(QgsLayoutSize(width, height))
    return label


def panel(layout, x, y, width, height, color=PAPER, outline=None):
    item = QgsLayoutItemShape(layout)
    item.setShapeType(QgsLayoutItemShape.Shape.Rectangle)
    item.setSymbol(QgsFillSymbol.createSimple({
        "color": color, "outline_style": "solid" if outline else "no",
        "outline_color": outline or color, "outline_width": "0.15",
    }))
    layout.addLayoutItem(item)
    item.attemptMove(QgsLayoutPoint(x, y))
    item.attemptResize(QgsLayoutSize(width, height))
    return item


def fill(color, outline="#ffffff", width="0.055", style="solid"):
    return QgsFillSymbol.createSimple({"color": color, "style": style, "outline_color": outline, "outline_width": width})


def rules(layer, specifications):
    root_rule = QgsRuleBasedRenderer.Rule(None)
    context = QgsExpressionContext()
    context.appendScopes(QgsExpressionContextUtils.globalProjectLayerScopes(layer))
    first = next(layer.getFeatures(), None)
    if first is not None:
        context.setFeature(first)
    for expression, label, symbol in specifications:
        parsed = QgsExpression(expression)
        if parsed.hasParserError() or not parsed.prepare(context):
            raise ValueError(f"Invalid cartographic rule for {layer.name()} / {label}: {parsed.parserErrorString()} {parsed.evalErrorString()}")
        if first is not None:
            parsed.evaluate(context)
            if parsed.hasEvalError():
                raise ValueError(f"Cannot evaluate cartographic rule for {layer.name()} / {label}: {parsed.evalErrorString()}")
        root_rule.appendChild(QgsRuleBasedRenderer.Rule(symbol, filterExp=expression, label=label))
    layer.setRenderer(QgsRuleBasedRenderer(root_rule))


def audit_theme(layer):
    """Require one visible class per feature, including an explicit missing class."""
    children = layer.renderer().rootRule().children()
    context = QgsExpressionContext()
    context.appendScopes(QgsExpressionContextUtils.globalProjectLayerScopes(layer))
    expressions = [QgsExpression(rule.filterExpression()) for rule in children]
    for expression in expressions:
        if not expression.prepare(context):
            raise ValueError(f"Cannot prepare thematic classification for {layer.name()}.")
    counts = [0] * len(children)
    checked = 0
    for feature in layer.getFeatures():
        context.setFeature(feature)
        matches = [index for index, expression in enumerate(expressions) if bool(clean(expression.evaluate(context)))]
        if any(expression.hasEvalError() for expression in expressions) or len(matches) != 1:
            identifier = feature["lsoa21cd"] if "lsoa21cd" in layer.fields().names() else feature.id()
            raise ValueError(f"{layer.name()}: feature {identifier} matches {len(matches)} classes; expected exactly one.")
        counts[matches[0]] += 1
        checked += 1
    return {"features": checked, "classes": {rule.label(): count for rule, count in zip(children, counts)}}


def metadata(layer, title, abstract, notes):
    details = QgsLayerMetadata()
    details.setTitle(title)
    details.setIdentifier("gm-climate-resilience:" + layer.name())
    details.setAbstract(abstract + "\n\n" + notes["source_line"])
    details.setLanguage("en-GB")
    details.setType("dataset")
    details.setCrs(layer.crs())
    details.setKeywords({"theme": ["adaptation screening", "Greater Manchester", "relative comparison"]})
    layer.setMetadata(details)
    if isinstance(layer, QgsVectorLayer):
        layer.setReadOnly(True)
        for field, alias in ALIASES.items():
            index = layer.fields().indexFromName(field)
            if index >= 0:
                layer.setFieldAlias(index, alias)
        if "lsoa21cd" in layer.fields().names() and "score_balanced" in layer.fields().names():
            layer.setDisplayExpression('"name" || \' • \' || "lsoa21cd"')
            layer.setMapTipTemplate(
                '<div style="font-family:Arial;color:#263f48;padding:8px;min-width:250px">'
                '<b>[% "name" %]</b><br>[% "lsoa21cd" %] · [% "borough" %]<hr>'
                'Population: [% format_number("population",0) %]<br>'
                'Equal-weight score: [% coalesce(format_number("score_balanced",1),\'Insufficient evidence\') %]<br>'
                'Highest-fifth selection: [% coalesce(format_number(100*"top20_frequency",0)||\'%\',\'Not scored\') %]<br>'
                '<small>Comparison evidence for investigation; not predicted losses.</small></div>'
            )


def clone_layer(project, base, name, group, notes, abstract="Derived cartographic view of processed evidence."):
    layer = base.clone()
    layer.setName(name)
    metadata(layer, name, abstract, notes)
    project.addMapLayer(layer, False)
    group.addLayer(layer)
    return layer


def numeric_theme(project, base, name, field, cuts, colors, labels, group, notes, valid_expression=None):
    layer = clone_layer(project, base, name, group, notes)
    valid = valid_expression or f'"{field}" IS NOT NULL'
    spec = []
    for index, (lower, upper, color, label) in enumerate(zip(cuts[:-1], cuts[1:], colors, labels)):
        lower_test = ">=" if index == 0 else ">"
        condition = f'({valid}) AND "{field}" {lower_test} {lower:.12g} AND "{field}" <= {upper:.12g}'
        spec.append((condition, label, fill(color)))
    spec.append((f'NOT coalesce(({valid}),false)', "Insufficient evidence", fill(MISSING)))
    rules(layer, spec)
    return layer


def label_boroughs(layer, font, size=9):
    settings = QgsPalLayerSettings()
    settings.fieldName = "borough"
    settings.placement = Qgis.LabelPlacement.OverPoint
    fmt = text_format(font, size, INK, True)
    buffer = QgsTextBufferSettings()
    buffer.setEnabled(True)
    buffer.setSize(0.75)
    buffer.setColor(QColor("#ffffff"))
    fmt.setBuffer(buffer)
    settings.setFormat(fmt)
    layer.setLabeling(QgsVectorLayerSimpleLabeling(settings))
    layer.setLabelsEnabled(True)


def add_map(layout, layers, extent, x, y, width, height, background="#f4f7f4"):
    item = QgsLayoutItemMap(layout)
    layout.addLayoutItem(item)
    item.attemptMove(QgsLayoutPoint(x, y))
    item.attemptResize(QgsLayoutSize(width, height))
    item.setCrs(CRS)
    item.setLayers(layers)
    item.setKeepLayerSet(True)
    item.setKeepLayerStyles(True)
    item.setBackgroundColor(QColor(background))
    item.setBackgroundEnabled(True)
    item.setFrameEnabled(True)
    item.setFrameStrokeColor(QColor(LINE))
    rectangle = QgsRectangle(extent)
    rectangle.scale(1.045)
    item.zoomToExtent(rectangle)
    return item


def add_legend(layout, map_item, layer, title, font):
    legend = QgsLayoutItemLegend(layout)
    legend.setTitle(title)
    legend.setLinkedMap(map_item)
    legend.setSyncMode(Qgis.LegendSyncMode.Manual)
    legend.model().rootGroup().clear()
    node = legend.model().rootGroup().addLayer(layer)
    node.setCustomProperty("legend/title-label", "")
    node.setCustomProperty("legend/title-style", "hidden")
    legend.setSymbolWidth(8)
    legend.setSymbolHeight(3.5)
    legend.setBoxSpace(0)
    legend.setColumnSpace(1)
    for component in Qgis.LegendComponent:
        if component in (Qgis.LegendComponent.Undefined, Qgis.LegendComponent.Hidden):
            continue
        legend.rstyle(component).setTextFormat(text_format(font, 8.3 if component != Qgis.LegendComponent.Title else 9, INK, component == Qgis.LegendComponent.Title))
    layout.addLayoutItem(legend)
    legend.attemptMove(QgsLayoutPoint(311, 96))
    legend.attemptResize(QgsLayoutSize(95, 71))
    return legend


def make_layout(project, definition, font, notes, context_layers, region_extent):
    layout = QgsPrintLayout(project)
    layout.initializeDefaults()
    layout.setName(definition["name"])
    layout.renderContext().setDpi(300)
    layout.pageCollection().pages()[0].setPageSize("A3", QgsLayoutItemPage.Orientation.Landscape)
    add_label(layout, notes["edition"] + "    /    " + definition["number"], 14, 9, 391, 6, font, 8.5, TEAL, True)
    add_label(layout, definition["title"], 14, 18, 391, 17, font, 25, INK, True)
    add_label(layout, definition["subtitle"], 14, 38, 391, 9, font, 10, MUTED)
    panel(layout, 14, 50, 392, 0.6, TEAL)
    map_item = add_map(layout, definition["layers"], definition["extent"], 14, 56, 285, 211)
    map_item.setId(definition["name"] + "_main_map")
    add_label(layout, "N\n↑", 285, 60, 9, 14, font, 11, INK, True)
    panel(layout, 20, 251, 78, 13, "#ffffff")
    scale = QgsLayoutItemScaleBar(layout)
    scale.setStyle("Single Box")
    scale.setLinkedMap(map_item)
    scale.setUnits(Qgis.DistanceUnit.Kilometers)
    scale.setUnitsPerSegment(2 if definition["name"] == "04_accessibility" else 5)
    scale.setNumberOfSegments(2)
    scale.setNumberOfSegmentsLeft(0)
    scale.setUnitLabel("km")
    scale.setTextFormat(text_format(font, 7.3))
    scale.setHeight(1.7)
    layout.addLayoutItem(scale)
    scale.attemptMove(QgsLayoutPoint(22, 252))
    add_label(layout, "READING THE EVIDENCE", 311, 56, 95, 7, font, 8.5, TEAL, True)
    add_label(layout, definition["reading"], 311, 66, 95, 27, font, 9.2, INK)
    add_legend(layout, map_item, definition["legend_layer"], definition["legend_title"], font)
    if definition.get("extra_key"):
        add_label(layout, definition["extra_key"], 311, 160, 95, 17, font, 7.7, MUTED)
    panel(layout, 311, 180, 95, 0.25, LINE)
    for index, (value, caption) in enumerate(definition["metrics"][:2]):
        x = 311 + 49 * index
        add_label(layout, value, x, 185, 46, 12, font, 19, TEAL, True)
        add_label(layout, caption, x, 198, 46, 15, font, 8, MUTED)
    if context_layers:
        add_label(layout, "Manchester + Salford / detailed access case", 311, 216, 95, 6, font, 7.5, MUTED, True)
        inset = add_map(layout, context_layers, region_extent, 317, 224, 85, 40, "#ffffff")
        inset.setId(definition["name"] + "_context_map")
    else:
        add_label(layout, "CONTEXT\nGreater Manchester\nBritish National Grid / metres", 311, 225, 95, 28, font, 8.5, MUTED)
    add_label(layout, definition["limit"], 14, 273, 392, 8, font, 7.7, INK)
    add_label(layout, notes["source_line"], 14, 282, 392, 6, font, 7.1, MUTED)
    add_label(layout, notes["portfolio_note"], 14, 290, 340, 4, font, 7.1, MUTED)
    add_label(layout, definition["number"] + " / 06", 385, 290, 21, 4, font, 7.1, MUTED)
    project.layoutManager().addLayout(layout)
    return layout


def raster_style(layer, minimum, maximum, colors):
    shader = QgsRasterShader()
    ramp = QgsColorRampShader()
    ramp.setColorRampType(QgsColorRampShader.Type.Interpolated)
    ramp.setClassificationMode(Qgis.ShaderClassificationMethod.Continuous)
    ramp.setColorRampItemList([
        QgsColorRampShader.ColorRampItem(float(value), QColor(color), f"{value:g}")
        for value, color in zip(np.linspace(minimum, maximum, len(colors)), colors)
    ])
    shader.setRasterShaderFunction(ramp)
    renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader)
    renderer.setClassificationMin(minimum)
    renderer.setClassificationMax(maximum)
    layer.setRenderer(renderer)


def build_project(input_path, project_path, notes, font):
    if not input_path.exists():
        raise FileNotFoundError(f"Required processed evidence is absent: {input_path}. Run data processing and analysis first.")
    config = json.loads((ROOT / "config/analysis.json").read_text())
    selection_path = ROOT / "metadata/landsat_selection.json"
    selected_scenes = json.loads(selection_path.read_text()).get("scenes", []) if selection_path.exists() else []
    selected_years = sorted({str(scene.get("date", ""))[:4] for scene in selected_scenes if scene.get("date")})
    heat_subtitle = "Summer clear-sky land-surface temperature • anomaly in °C, with coverage retained"
    if selected_scenes and selected_years:
        heat_subtitle = f"{len(selected_scenes)} selected clear-sky summer dates ({selected_years[0]}–{selected_years[-1]}) • 100 m surface-temperature anomaly • not a continuous series"
    project = QgsProject.instance()
    project.clear()
    project.setCrs(CRS)
    project.setFileName(str(project_path))
    project.setFilePathStorage(Qgis.FilePathType.Relative)
    details = project.metadata()
    details.setTitle("Greater Manchester • climate resilience investment screening")
    details.setAuthor("Ewan Moorcroft")
    details.setAbstract("Independent, evidence-led neighbourhood screening; five distinct dimensions, scenario comparisons and explicit uncertainty. " + notes["priority_limit"])
    details.setCreationDateTime(QDateTime.currentDateTimeUtc())
    project.setMetadata(details)
    project.setCustomVariables({"study_area": "Greater Manchester", "analysis_crs": "EPSG:27700", "source_note": notes["source_line"]})
    root = project.layerTreeRoot()
    screening = root.addGroup("01 • Screening maps")
    evidence = root.addGroup("02 • Source indicators and evidence")
    access_group = root.addGroup("03 • Manchester + Salford walking case")
    context = root.addGroup("04 • Administrative context")
    warnings = []
    loaded = {}

    def load_vector(key, name, group, required=False):
        options = QgsVectorLayer.LayerOptions()
        options.forceReadOnly = True
        options.loadDefaultStyle = False
        layer = QgsVectorLayer(str(input_path) + "|layername=" + key, name, "ogr", options)
        if layer.isValid() and layer.featureCount() < 0:
            raise RuntimeError(f"Provider cannot read a reliable feature count for {key}; build stopped.")
        if not layer.isValid() or layer.featureCount() == 0:
            message = f"{'Required' if required else 'Optional'} layer {key} is missing, empty or invalid."
            if required:
                raise ValueError(message)
            warnings.append(message)
            return None
        if layer.crs().authid() != "EPSG:27700":
            raise ValueError(f"{key} uses {layer.crs().authid()}; processed layers must use EPSG:27700.")
        metadata(layer, name, "Processed evidence retained for inspection; no provider data are edited by this project.", notes)
        project.addMapLayer(layer, False)
        group.addLayer(layer)
        loaded[key] = layer
        return layer

    neighbourhoods = load_vector("neighbourhoods", "Neighbourhood evidence • full attribute table", evidence, True)
    neighbourhoods.setRenderer(QgsSingleSymbolRenderer(fill("#e1e8e3", "#ffffff", "0.06")))
    missing = REQUIRED_FIELDS - set(neighbourhoods.fields().names())
    if missing:
        raise ValueError("Required neighbourhood fields missing: " + ", ".join(sorted(missing)))
    rows = [{field.name(): clean(feature[field.name()]) for field in neighbourhoods.fields()} for feature in neighbourhoods.getFeatures()]
    eligible = [row for row in rows if truth(row.get("model_eligible"))]
    if not eligible or not values(eligible, "score_balanced"):
        raise ValueError("No neighbourhood has usable model evidence; priority maps cannot be built.")
    region_extent = neighbourhoods.extent()
    boroughs = load_vector("boroughs", "Borough boundaries and names", context)
    borough_outline = None
    context_layers = []
    if boroughs:
        if "borough" not in boroughs.fields().names():
            raise ValueError("The boroughs layer requires a borough field.")
        boroughs.setRenderer(QgsSingleSymbolRenderer(fill("#edf1ee", "#8d9c99", "0.18")))
        borough_outline = clone_layer(project, boroughs, "Borough labels and outlines", context, notes)
        borough_outline.setRenderer(QgsSingleSymbolRenderer(fill("255,255,255,0", "#6d817f", "0.20")))
        label_boroughs(borough_outline, font)
        case_context = clone_layer(project, boroughs, "Detailed case location", context, notes)
        rules(case_context, [
            ('"borough" IN (\'Manchester\',\'Salford\')', "Manchester + Salford", fill("#4c9187", "#ffffff", "0.15")),
            ('"borough" NOT IN (\'Manchester\',\'Salford\')', "Other GM boroughs", fill("#e0e7e2", "#ffffff", "0.15")),
        ])
        context_layers = [case_context]

    flood_extent = load_vector("flood_extent", "Defended flood footprint • main scenario", evidence)
    if flood_extent:
        if "hazard" not in flood_extent.fields().names():
            warnings.append("flood_extent lacks hazard codes; its scenario cannot be verified and the overlay is excluded from map 02.")
            flood_extent = None
        else:
            alternative_flood = clone_layer(project, flood_extent, "Defended flood footprint • 0.1% alternative", evidence, notes, "Independently mapped 0.1% river and sea scenario. It is not a guaranteed spatial superset of the main defended footprint.")
            alternative_flood.setSubsetString('"hazard" = \'defended_01pct\'')
            alternative_flood.setRenderer(QgsSingleSymbolRenderer(fill("123,156,174,45", "#839fac", "0.10")))
            flood_extent.setSubsetString('"hazard" = \'defended_1pct\'')
            flood_extent.setRenderer(QgsSingleSymbolRenderer(fill("54,99,125,70", "54,99,125,100", "0.06")))
            if flood_extent.featureCount() == 0:
                warnings.append("No defended_1pct flood geometry is available; the main footprint overlay is excluded from map 02.")
                flood_extent = None
    all_greenspaces = load_vector("greenspaces", "All public parks • context inventory", access_group)
    greenspaces = None
    if all_greenspaces:
        all_greenspaces.setRenderer(QgsSingleSymbolRenderer(fill("#d1dfd6", "#ffffff", "0.08")))
        greenspaces = clone_layer(project, all_greenspaces, "Qualifying public parks ≥10 ha", access_group, notes, "Routing destinations only. Smaller parks remain in the separate context inventory and are not treated as eligible destinations.")
        if "eligible" in greenspaces.fields().names():
            greenspaces.setSubsetString('"eligible" = 1')
        elif "area_ha" in greenspaces.fields().names():
            greenspaces.setSubsetString(f'"area_ha" >= {config["minimum_public_park_ha"]}')
        else:
            raise ValueError("Green-space destinations require an eligible flag or area_ha field.")
        greenspaces.setRenderer(QgsSingleSymbolRenderer(fill("#488577", "#ffffff", "0.1")))
        greenspaces.setScaleBasedVisibility(True)
        greenspaces.setMinimumScale(600000)
    entrances = load_vector("entrances", "Mapped park entrances", access_group)
    if entrances:
        entrances.setRenderer(QgsSingleSymbolRenderer(QgsMarkerSymbol.createSimple({"name": "circle", "color": "#ffffff", "outline_color": "#226157", "outline_width": "0.18", "size": "1.2"})))
        entrances.setScaleBasedVisibility(True)
        entrances.setMinimumScale(100000)
    origins = load_vector("case_origins", "Walking-access population origins", access_group)
    if origins:
        if not {"known", "distance_m"} <= set(origins.fields().names()):
            warnings.append("case_origins lacks known/distance_m; distance symbols are unavailable.")
            origins.setRenderer(QgsSingleSymbolRenderer(QgsMarkerSymbol.createSimple({"color": "#617578", "size": "0.75"})))
        else:
            def dot(color, size="0.95"):
                return QgsMarkerSymbol.createSimple({"name": "circle", "color": color, "outline_style": "no", "size": size})
            threshold = config["walking_threshold_m"]
            finite_distance = '("distance_m" IS NOT NULL AND lower(to_string("distance_m")) NOT IN (\'inf\',\'-inf\',\'nan\',\'infinity\',\'-infinity\'))'
            route = 'coalesce("reachable",false)' if "reachable" in origins.fields().names() else finite_distance
            rules(origins, [
                (f'"known" AND ({route}) AND "distance_m" <= {threshold}', f"Mapped walk ≤ {threshold:g} m", dot("#28766c", "0.65")),
                (f'"known" AND ({route}) AND "distance_m" > {threshold}', f"Mapped walk > {threshold:g} m", dot("#a76b34")),
                (f'"known" AND NOT ({route})', "No mapped route", QgsMarkerSymbol.createSimple({"name": "cross2", "color": "#5c6261", "outline_color": "#5c6261", "size": "1.4", "outline_width": "0.22"})),
                ('NOT coalesce("known",false)', "Origin not snapped / unknown", dot("#929b97", "0.8")),
            ])
        origins.setScaleBasedVisibility(True)
        origins.setMinimumScale(220000)

    for filename, name, low, high, colors in [
        ("heat_anomaly.tif", "Surface-temperature anomaly • raster", -3, 3, ["#4f8590", "#eff2e8", "#ba7338"]),
        ("ndvi.tif", "Vegetation index • context only", -0.1, 0.8, ["#d6c6a7", "#eff2dc", "#337568"]),
        ("heat_observations.tif", "Valid heat observations • coverage", 0, len(selected_scenes) or config["landsat_max_dates"], ["#f0eee6", "#739b99", "#214f55"]),
    ]:
        path = input_path.parent / filename
        if not path.exists():
            warnings.append(f"Optional raster {filename} is absent.")
            continue
        layer = QgsRasterLayer(str(path), name, "gdal")
        if not layer.isValid():
            warnings.append(f"Optional raster {filename} is invalid.")
            continue
        raster_style(layer, low, high, colors)
        metadata(layer, name, "Processed raster evidence. Vegetation is contextual and is not an additional scoring dimension.", notes)
        project.addMapLayer(layer, False)
        evidence.addLayer(layer)
        loaded[filename] = layer

    # Each map owns a distinct thematic layer. No shared style is mutated while
    # exporting another map, and all styles remain inspectable in the layer tree.
    dim_fields = ["dim_flood", "dim_heat", "dim_access", "dim_income", "dim_age"]
    maximum = "array_max(array(" + ",".join('"' + field + '"' for field in dim_fields) + "))"
    ties = " + ".join(f'CASE WHEN "{field}" = {maximum} THEN 1 ELSE 0 END' for field in dim_fields)
    dominance = clone_layer(project, neighbourhoods, "Most adverse relative dimension", screening, notes)
    driver_names = ["Flood", "Surface heat", "Green-space access", "Income", "Age sensitivity"]
    driver_colors = ["#456d85", "#b87c35", "#508a78", "#805a6b", "#8d887b"]
    dominance_rules = []
    for field, name, color in zip(dim_fields, driver_names, driver_colors):
        dominance_rules.append((f'"model_eligible" AND ({ties}) = 1 AND "{field}" = {maximum}', name, fill(color)))
    dominance_rules += [
        (f'"model_eligible" AND ({ties}) > 1', "Joint highest dimensions", fill("#3d525a")),
        ('NOT coalesce("model_eligible",false)', "Insufficient evidence", fill(MISSING)),
    ]
    rules(dominance, dominance_rules)
    flood = numeric_theme(project, neighbourhoods, "Flood exposure • neighbourhood share", "flood_exposure", [0, .01, .05, .15, .3, 1], ["#edf3f3", "#c7dce0", "#8cb8c4", "#528b9f", "#28556c"], ["0–1%", ">1–5%", ">5–15%", ">15–30%", ">30%"], screening, notes)
    coverage = config["minimum_lsoa_heat_coverage"]
    heat = numeric_theme(project, neighbourhoods, "Daytime surface-heat anomaly", "heat_anomaly", [-100, -1.5, -.5, .5, 1.5, 100], ["#4e8793", "#abcacf", "#eff0df", "#d9b278", "#ae6c30"], ["≤ −1.5°C", "−1.5 to −0.5°C", "−0.5 to +0.5°C", "+0.5 to +1.5°C", "> +1.5°C"], screening, notes, f'"heat_anomaly" IS NOT NULL AND "heat_coverage" >= {coverage}')
    regional_access = numeric_theme(project, neighbourhoods, "Official household access deficit", "access_deficit", [0, .2, .4, .6, .8, 1], ["#eef2df", "#cbd9bd", "#94b8a5", "#5c948c", "#2c625f"], ["0–20%", ">20–40%", ">40–60%", ">60–80%", ">80%"], screening, notes)
    case_rows = [row for row in rows if row.get("borough") in config["case_study_boroughs"]]
    case_field = "case_network_deficit"
    has_case_metric = bool(values(case_rows, case_field))
    if not has_case_metric:
        warnings.append("Case network deficit is unavailable; map 04 explicitly shows the official regional household proxy within the two case boroughs.")
        case_field = "access_deficit"
    access = numeric_theme(project, neighbourhoods, "Case walking-access deficit" if has_case_metric else "Case boroughs • official household access proxy", case_field, [0, .2, .4, .6, .8, 1], ["#f4eee0", "#e2cfaa", "#c9ac79", "#ad8650", "#856037"], ["0–20%", ">20–40%", ">40–60%", ">60–80%", ">80%"], screening, notes)
    access.setSubsetString('"borough" IN (\'Manchester\',\'Salford\')')
    case_extent = access.extent()
    if case_extent.isEmpty():
        raise ValueError("Manchester and Salford cannot be located in the neighbourhood borough field.")
    scores = values(eligible, "score_balanced")
    inner = np.quantile(scores, [.2, .4, .6, .8]).tolist()
    cuts = [0] + inner + [100]
    if len(set(inner)) < 4:
        cuts = [0, 20, 40, 60, 80, 100]
        warnings.append("Score quantiles tied; priority legend uses fixed 20-point intervals.")
    score_labels = [f"{a:.1f}–{b:.1f}" for a, b in zip(cuts[:-1], cuts[1:])]
    priority = numeric_theme(project, neighbourhoods, "Equal-weight priority score", "score_balanced", cuts, ["#e6eeea", "#bfd5ca", "#85b7a7", "#4b9384", "#205f58"], score_labels, screening, notes, '"model_eligible" AND "score_balanced" IS NOT NULL')
    robust = numeric_theme(project, neighbourhoods, "Priority selection frequency", "top20_frequency", [0, .2, .5, .8, 1], ["#edf0e9", "#cedac3", "#91b8a0", "#32766d"], ["0–20% of samples", ">20–50%", ">50–80%", ">80–100%"], screening, notes, '"model_eligible" AND "top20_frequency" IS NOT NULL')
    stable_outline = clone_layer(project, neighbourhoods, "Stable-priority outline", screening, notes)
    rules(stable_outline, [('"robust_priority"', "Stable priority", fill("255,255,255,0", "#173f3c", "0.30"))])
    classification_checks = {layer.name(): audit_theme(layer) for layer in [dominance, flood, heat, regional_access, access, priority, robust]}

    def stack(theme, *extra):
        return [layer for layer in [borough_outline, *extra, theme] if layer is not None]

    valid_heat = [row for row in rows if number(row.get("heat_anomaly")) is not None and (number(row.get("heat_coverage")) or 0) >= coverage]
    flooded = sum((number(row.get("flood_exposure")) or 0) > 0 for row in rows)
    flood_area_km2 = sum((number(clean(feature["flood_exposure"])) or 0) * feature.geometry().area() / 1_000_000 for feature in neighbourhoods.getFeatures())
    stable_count = sum(truth(row.get("robust_priority")) for row in rows)
    top_threshold = sorted(scores, reverse=True)[max(0, math.ceil(len(scores) * config["top_fraction"]) - 1)]
    top_count = sum(score >= top_threshold for score in scores)
    case_known, case_reachable = [], []
    disconnected_count = 0
    known_population = deficit_population = 0.0
    if origins and "known" in origins.fields().names():
        for feature in origins.getFeatures():
            known = truth(clean(feature["known"]))
            reachable = truth(clean(feature["reachable"])) if "reachable" in origins.fields().names() else number(clean(feature["distance_m"])) is not None
            case_known.append(known)
            case_reachable.append(known and reachable)
            disconnected_count += known and not reachable
            population = number(clean(feature["population"])) if "population" in origins.fields().names() else None
            distance = number(clean(feature["distance_m"]))
            if known and population is not None:
                known_population += population
                if not reachable or (distance is not None and distance > config["walking_threshold_m"]):
                    deficit_population += population
    case_deficits = values(case_rows, case_field)
    metrics = {
        "overview": [(f"{len(eligible):,}", "areas with complete scoring evidence"), (f"{len(rows):,}", "neighbourhoods in the study")],
        "flood": [(f"{flooded:,}", "areas intersecting the flood footprint"), (f"{flood_area_km2:.1f}", "km² of neighbourhood area in the footprint")],
        "heat": [(f"{len(valid_heat):,}", "areas meeting heat coverage rule"), (percent(coverage), "minimum valid area coverage")],
        "access": [(percent(sum(case_reachable) / len(case_reachable)) if case_reachable else "—", "case origins with a mapped route"), (percent(deficit_population / known_population) if known_population else (percent(statistics.median(case_deficits)) if case_deficits else "—"), "residents beyond 1 km or no mapped route" if known_population else "median area access deficit")],
        "priorities": [(f"{top_count:,}", "areas reaching the highest-fifth threshold"), (f"{stable_count:,}", "areas meeting the stable-priority rule")],
        "robustness": [(f"{config['sensitivity_samples']:,}", "sampled weight combinations"), (percent(config["robust_frequency_threshold"]), "stable-priority frequency threshold")],
    }
    definitions = [
        dict(name="01_overview", number="01", title="Different places. Different pressures.", subtitle="The most adverse relative dimension in each neighbourhood • five dimensions, one comparison area", reading="Colour identifies the largest normalised dimension, not the largest physical hazard. Joint maxima are shown separately.", legend_layer=dominance, legend_title="Most adverse dimension", layers=stack(dominance), extent=region_extent, metrics=metrics["overview"], limit=notes["overview_limit"]),
        dict(name="02_flood", number="02", title="Where flood exposure concentrates", subtitle="LSOA polygon-area fraction • current EA defended 1% river / 0.5% sea scenario", reading="Darker areas have a larger share of their area within the flood footprint. Blue geometry shows the underlying mapped extent.", legend_layer=flood, legend_title="Share of neighbourhood area", layers=stack(flood, flood_extent), extent=region_extent, metrics=metrics["flood"], limit=notes["flood_limit"]),
        dict(name="03_heat", number="03", title="A warmer surface is a different signal", subtitle=heat_subtitle, reading="Warm colours show above-reference surface temperatures. Grey areas lack an estimate meeting the coverage rule.", legend_layer=heat, legend_title="Surface anomaly / °C", layers=stack(heat), extent=region_extent, metrics=metrics["heat"], limit=notes["heat_limit"]),
        dict(name="04_accessibility", number="04", title="A park nearby is not always a short walk", subtitle=(f"Manchester + Salford • public parks ≥10 ha • {config['walking_threshold_m']:,} m mapped walking threshold" if has_case_metric else "Manchester + Salford • official household access proxy; network deficit unavailable"), reading=("Ochre areas have greater population-weighted walking deficits. Green polygons are parks; origin dots show known and unknown routes." if has_case_metric else "Network deficit is unavailable. This view shows the separate official 2024 household proxy, with available park context."), legend_layer=access, legend_title="Access deficit / share", layers=stack(access, origins, entrances, greenspaces), extent=case_extent, metrics=metrics["access"], limit=notes["access_limit"]),
        dict(name="05_priorities", number="05", title="Where further appraisal has the strongest case", subtitle="Equal-weight screening • flood, surface heat, access, income and age • stable priorities outlined", reading="Darker areas score higher across the five dimensions. Outlines identify areas that also satisfy the configured stability rule.", legend_layer=priority, legend_title="Equal-weight score / 100", layers=stack(priority, stable_outline), extent=region_extent, metrics=metrics["priorities"], limit=notes["priority_limit"]),
        dict(name="06_robustness", number="06", title="Which priorities survive reweighting?", subtitle="Frequency of reaching the highest-fifth score threshold across sampled weights • ties included", reading="Darker areas are selected more consistently. Stable outlines also require selection in all three named scenarios; source adequacy is separate.", legend_layer=robust, legend_title="Highest-fifth frequency", layers=stack(robust, stable_outline), extent=region_extent, metrics=metrics["robustness"], limit=notes["robustness_limit"]),
    ]
    if has_case_metric:
        definitions[3]["extra_key"] = "Green: qualifying public parks ≥10 ha\nOrigins: teal ≤1 km · ochre >1 km · × no mapped route"
        definitions[3]["reading"] = "Ochre areas have more residents beyond 1 km or with no mapped route. Green polygons are qualifying parks; crosses show disconnected origins."
        definitions[3]["limit"] = f"OA population-weighted deficit: beyond 1 km or no mapped route to parks ≥10 ha (2026); {disconnected_count:,} disconnected origins. Official 2024 household access is different."
    for definition in definitions:
        make_layout(project, definition, font, notes, context_layers, region_extent)
    STYLE_DIR.mkdir(parents=True, exist_ok=True)
    for key, layer in [
        ("overview_drivers", dominance), ("flood_exposure", flood), ("heat_anomaly", heat),
        ("regional_access_deficit", regional_access), ("case_access_deficit", access),
        ("balanced_priorities", priority), ("robustness", robust), ("stable_priority_outline", stable_outline),
    ]:
        message, success = layer.saveNamedStyle(str(STYLE_DIR / (key + ".qml")))
        if not success:
            raise RuntimeError(f"Style export failed for {key}: {message}")
    tree_model = QgsLayerTreeModel(root)
    for definition in definitions:
        visible = {layer.id() for layer in definition["layers"]}
        for node in root.findLayers():
            node.setItemVisibilityChecked(node.layerId() in visible)
        project.mapThemeCollection().insert(definition["name"], QgsMapThemeCollection.createThemeFromCurrentState(root, tree_model))
    initial = {layer.id() for layer in definitions[4]["layers"]}
    for node in root.findLayers():
        node.setItemVisibilityChecked(node.layerId() in initial)
        node.setExpanded(False)
    for group in [evidence, access_group, context]:
        group.setExpanded(False)
    project.viewSettings().setDefaultViewExtent(QgsReferencedRectangle(region_extent, CRS))
    project_path.parent.mkdir(parents=True, exist_ok=True)
    if not project.write():
        raise RuntimeError("QGIS project could not be saved.")
    normalized_sources = normalize_layout_sources(project_path, project)
    return {
        "warnings": warnings, "neighbourhoods": len(rows), "eligible_neighbourhoods": len(eligible),
        "case_access_measure": case_field, "priority_legend_edges": cuts,
        "case_route_evidence": {"origins": len(case_known), "snapped_origins": sum(case_known), "reachable_origins": sum(case_reachable), "disconnected_snapped_origins": disconnected_count, "known_population": known_population, "deficit_population": deficit_population, "overall_deficit": deficit_population / known_population if known_population else None},
        "classification_checks": classification_checks,
        "heat_selection_context": {"scene_count": len(selected_scenes), "selected_years": selected_years, "subtitle": heat_subtitle},
        "flood_main_footprint_km2": flood_area_km2,
        "normalized_layout_fallback_sources": normalized_sources,
        "layout_details": [{"name": d["name"], "title": d["title"], "layer_ids": [layer.id() for layer in d["layers"]], "scale": project.layoutManager().layoutByName(d["name"]).itemById(d["name"] + "_main_map").scale()} for d in definitions],
    }


def export_layouts(project, png_dpi, results=None):
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    results = [] if results is None else results
    for name in LAYOUT_NAMES:
        layout = project.layoutManager().layoutByName(name)
        if layout is None:
            raise ValueError(f"Required layout is missing from the project: {name}")
        exporter = QgsLayoutExporter(layout)
        png_path, pdf_path = MAP_DIR / (name + ".png"), MAP_DIR / (name + ".pdf")
        # A presentation PNG needs pixels, not embedded GIS georeferencing.
        # renderPageToImage avoids the macOS PNG driver's unsupported update path.
        image = exporter.renderPageToImage(0, dpi=png_dpi)
        png_ok = not image.isNull() and image.save(str(png_path), "PNG")
        settings = QgsLayoutExporter.PdfExportSettings()
        settings.dpi = 300
        settings.forceVectorOutput = True
        settings.textRenderFormat = Qgis.TextRenderFormat.AlwaysText
        settings.exportMetadata = True
        # A fresh path avoids stale PDF content when QGIS post-processes an
        # existing target. Replace the previous export only after success.
        with tempfile.TemporaryDirectory(prefix=".qgis-pdf-", dir=MAP_DIR) as temporary:
            temporary_pdf = Path(temporary) / pdf_path.name
            status = exporter.exportToPdf(str(temporary_pdf), settings)
            pdf_ok = status == QgsLayoutExporter.ExportResult.Success and temporary_pdf.is_file()
            if pdf_ok:
                temporary_pdf.replace(pdf_path)
        results.append({
            "layout": name, "png_success": png_ok, "pdf_success": pdf_ok,
            "pdf_export_code": int(status), "png_pixels": [image.width(), image.height()],
            "png": str(png_path.relative_to(ROOT)), "pdf": str(pdf_path.relative_to(ROOT)),
            "png_bytes": png_path.stat().st_size if png_ok else None,
            "pdf_bytes": pdf_path.stat().st_size if pdf_ok else None,
        })
        print(f"{name}: PNG {'OK' if png_ok else 'FAILED'}; PDF {'OK' if pdf_ok else 'FAILED'}", flush=True)
        if not png_ok or not pdf_ok:
            raise RuntimeError(f"Export failed for {name}: {exporter.errorMessage()}")
    return results


def verify_relocation(project_path):
    """Copy only the saved project and referenced files, then reopen that copy."""
    destination = ROOT / "metadata/qgis_relocation.json"
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "status": "failed", "project": str(project_path.relative_to(ROOT))}

    original, copied = QgsProject(), QgsProject()
    try:
        with zipfile.ZipFile(project_path) as archive:
            project_entries = [name for name in archive.namelist() if name.endswith(".qgs")]
            if len(project_entries) != 1:
                raise ValueError("QGZ must contain exactly one project XML document.")
            xml = ElementTree.fromstring(archive.read(project_entries[0]))
        source_uris = [item.text for item in xml.findall(".//projectlayers/maplayer/datasource")]
        layout_uris = [item.attrib["source"] for item in xml.findall(".//Layouts//Layer[@source]")]
        if not source_uris:
            raise ValueError("The project has no persisted data sources.")
        files = {project_path.relative_to(ROOT): project_path}
        for uri in source_uris + layout_uris:
            path = Path(uri.split("|", 1)[0])
            if path.is_absolute():
                raise ValueError(f"Absolute data source is not portable: {uri}")
            resolved = (project_path.parent / path).resolve()
            relative = resolved.relative_to(ROOT)
            if not str(relative).startswith("data/processed/") or not resolved.is_file():
                raise ValueError(f"Expected an existing processed input within the project: {uri}")
            files[relative] = resolved
        if not original.read(str(project_path)):
            raise ValueError("Cannot reopen the source QGIS project.")
        original_counts = {layer.id(): layer.featureCount() for layer in original.mapLayers().values() if isinstance(layer, QgsVectorLayer)}
        if any(count < 0 for count in original_counts.values()):
            raise ValueError("Source project has an unreadable feature count; relocation cannot pass.")
        report["relative_datasource_uris"] = source_uris
        report["relative_layout_fallback_uris"] = layout_uris
        report["copied_files"] = []
        with tempfile.TemporaryDirectory(prefix="gm-qgis-relocation-") as temporary:
            temporary_root = Path(temporary).resolve()
            for relative, source in files.items():
                target = temporary_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                checksum = sha256_file(source)
                if sha256_file(target) != checksum or os.path.samefile(source, target):
                    raise ValueError(f"Relocation copy differs from its source: {relative}")
                report["copied_files"].append({"path": str(relative), "bytes": target.stat().st_size, "sha256": checksum})
            copied_path = temporary_root / project_path.relative_to(ROOT)
            if not copied.read(str(copied_path)):
                raise ValueError("Cannot reopen the relocated QGIS project.")
            layers = []
            for layer in copied.mapLayers().values():
                source_path = Path(layer.source().split("|", 1)[0]).resolve()
                if not source_path.is_relative_to(temporary_root) or not source_path.is_file() or not layer.isValid():
                    raise ValueError(f"Relocated layer does not resolve inside the copied hierarchy: {layer.name()}")
                count = layer.featureCount() if isinstance(layer, QgsVectorLayer) else None
                if count is not None and (count < 0 or count != original_counts[layer.id()]):
                    raise ValueError(f"Relocated feature count changed: {layer.name()}")
                layers.append({"name": layer.name(), "source": str(source_path.relative_to(temporary_root)), "valid": True, "feature_count": count})
            layout_names = sorted(layout.name() for layout in copied.layoutManager().layouts())
            if layout_names != sorted(LAYOUT_NAMES) or sorted(copied.mapThemeCollection().mapThemes()) != sorted(LAYOUT_NAMES):
                raise ValueError("Relocation did not preserve all six layouts and map themes.")
            render_checks = []
            for name in LAYOUT_NAMES:
                baseline = QgsLayoutExporter(original.layoutManager().layoutByName(name)).renderPageToImage(0, dpi=72)
                relocated = QgsLayoutExporter(copied.layoutManager().layoutByName(name)).renderPageToImage(0, dpi=72)
                if baseline.isNull() or relocated.isNull() or baseline != relocated:
                    raise ValueError(f"Relocated layout is not pixel-identical to its source at 72 dpi: {name}")
                raw = relocated.constBits().asstring(relocated.sizeInBytes())
                render_checks.append({"layout": name, "dpi": 72, "pixels": [relocated.width(), relocated.height()], "pixel_identical": True, "rgba_sha256": hashlib.sha256(raw).hexdigest()})
            for relative, source in files.items():
                if sha256_file(source) != sha256_file(temporary_root / relative):
                    raise ValueError(f"A source changed during relocation rendering: {relative}")
            report.update({"status": "passed", "all_sources_relative": True, "all_layout_fallback_sources_relative": True, "all_sources_resolve_inside_copied_hierarchy": True, "layers": layers, "layouts": layout_names, "layout_render_checks": render_checks, "map_themes": copied.mapThemeCollection().mapThemes(), "temporary_copy_retained": False})
            copied.clear()
    except Exception as exc:
        report["error"] = str(exc)
        save_json(destination, report)
        raise
    finally:
        original.clear()
        copied.clear()
    save_json(destination, report)
    return report


def run(args, font):
    # QGIS 4.2 can switch a GPKG to WAL when a cloned layer gets a subset;
    # its existing NOLOCK readers then fail. This process-only option preserves
    # the normal DELETE journal mode and avoids changing source file headers.
    gdal.SetConfigOption("OGR_SQLITE_JOURNAL", "DELETE")
    notes = dict(DEFAULT_MAP_NOTES)
    if args.notes:
        overrides = json.loads(args.notes.read_text())
        unknown = set(overrides) - set(notes)
        if unknown:
            raise ValueError("Unknown map-note keys: " + ", ".join(sorted(unknown)))
        if any(not isinstance(value, str) for value in overrides.values()):
            raise ValueError("Map-note values must be strings.")
        notes.update(overrides)
    do_build = args.build or not args.export
    do_export = args.export or not args.build
    previous = json.loads(REPORT_PATH.read_text()) if not do_build and REPORT_PATH.exists() else {}
    result = dict(previous)
    result.update({"checked_at_utc": datetime.now(timezone.utc).isoformat(), "qgis_version": Qgis.QGIS_VERSION, "font_requested": font, "font_resolved": QFontInfo(QFont(font)).family(), "project": str(args.project.relative_to(ROOT)) if args.project.is_relative_to(ROOT) else str(args.project), "input": str(args.input.relative_to(ROOT)) if args.input.is_relative_to(ROOT) else str(args.input), "exports": [], "visual_qa": "pending; export success alone does not confirm visual quality", "source_notes": notes if do_build else previous.get("source_notes", notes)})
    if do_build:
        result["built_at_utc"] = result["checked_at_utc"]
    input_checksum = sha256_file(args.input) if args.input.exists() else None
    result.pop("error", None)
    try:
        if do_build:
            result.update(build_project(args.input, args.project, notes, font))
        elif not QgsProject.instance().read(str(args.project)):
            raise ValueError(f"Cannot read QGIS project: {args.project}")
        project = QgsProject.instance()
        result["layers"] = [{"name": layer.name(), "valid": layer.isValid(), "provider": layer.providerType(), "source": project.pathResolver().writePath(layer.source()), "feature_count": layer.featureCount() if isinstance(layer, QgsVectorLayer) else None} for layer in project.mapLayers().values()]
        if any(not layer["valid"] for layer in result["layers"]):
            raise ValueError("One or more project layers are invalid; export cancelled.")
        if any(layer["feature_count"] is not None and layer["feature_count"] < 0 for layer in result["layers"]):
            raise ValueError("One or more project layers has an unreadable feature count; export cancelled.")
        result["layouts"] = [layout.name() for layout in project.layoutManager().layouts()]
        result["map_themes"] = project.mapThemeCollection().mapThemes()
        if do_export:
            export_layouts(project, args.dpi, result["exports"])
        relocated = verify_relocation(args.project)
        result["relocation"] = {"status": relocated["status"], "report": "metadata/qgis_relocation.json", "layer_count": len(relocated["layers"])}
        final_checksum = sha256_file(args.input)
        result["input_integrity"] = {"sha256_before": input_checksum, "sha256_after": final_checksum, "unchanged": input_checksum == final_checksum, "ogr_sqlite_journal": "DELETE"}
        if input_checksum != final_checksum:
            raise ValueError("Processed GeoPackage changed during project construction or export.")
        result["status"] = "passed"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        save_json(REPORT_PATH, result)
        raise
    save_json(REPORT_PATH, result)
    for warning in result.get("warnings", []):
        print("WARNING: " + warning, file=sys.stderr)
    print("Project report: " + str(REPORT_PATH.relative_to(ROOT)), flush=True)
    return result


def main():
    global CRS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="Build project and layouts; combine with --export to also render.")
    parser.add_argument("--export", action="store_true", help="Export saved layouts; combine with --build to rebuild first.")
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/resilience.gpkg")
    parser.add_argument("--project", type=Path, default=PROJECT_PATH)
    parser.add_argument("--notes", type=Path, help="JSON overrides for source and interpretation captions.")
    parser.add_argument("--dpi", type=int, default=180, help="PNG resolution; vector PDFs use 300 dpi for raster content.")
    args = parser.parse_args()
    if not 72 <= args.dpi <= 400:
        parser.error("--dpi must be between 72 and 400.")
    args.input, args.project = args.input.resolve(), args.project.resolve()
    profile = os.environ.get("QGIS_CUSTOM_CONFIG_PATH", str(ROOT / "work/qgis-profile"))
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QgsApplication([], False, profile)
    app.initQgis()
    CRS = QgsCoordinateReferenceSystem("EPSG:27700")
    available = set(QFontDatabase.families())
    font = next((candidate for candidate in ["Arial", "Helvetica Neue", "DejaVu Sans"] if candidate in available), QFont().family())
    try:
        run(args, font)
    finally:
        QgsProject.instance().clear()
        gc.collect()
        app.exitQgis()


if __name__ == "__main__":
    main()
