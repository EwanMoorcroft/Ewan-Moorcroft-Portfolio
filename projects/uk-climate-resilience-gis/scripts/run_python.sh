#!/bin/sh
# Run project Python with an isolated QGIS profile and the matching GIS libraries.
# macOS: set QGIS_APP=/path/to/QGIS.app to select a specific installation.
# Other platforms: configure PyQGIS normally; optionally set QGIS_PYTHON.
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export QGIS_CUSTOM_CONFIG_PATH="$project_root/work/qgis-profile"
export XDG_CONFIG_HOME="$project_root/work/cache/config"
export XDG_CACHE_HOME="$project_root/work/cache"
export MPLCONFIGDIR="$project_root/work/cache/matplotlib"
# QGIS/GDAL subset filters otherwise switch GeoPackage journal modes while
# earlier providers still hold connections. A fixed process-local mode keeps
# repeated read-only builds stable and leaves the analytical snapshot unchanged.
export OGR_SQLITE_JOURNAL=DELETE
mkdir -p "$QGIS_CUSTOM_CONFIG_PATH" "$XDG_CONFIG_HOME" "$MPLCONFIGDIR"

if [ "$(uname -s)" = Darwin ]; then
    qgis_app=${QGIS_APP:-}
    if [ -z "$qgis_app" ]; then
        for candidate in /Applications/QGIS*.app "$HOME"/Applications/QGIS*.app; do
            if [ -x "$candidate/Contents/MacOS/python" ]; then
                qgis_app=$candidate
                break
            fi
        done
    fi
    if [ -z "$qgis_app" ] || [ ! -x "$qgis_app/Contents/MacOS/python" ]; then
        printf '%s\n' 'No usable QGIS application found. Set QGIS_APP to the QGIS .app directory.' >&2
        exit 1
    fi

    qgis_contents="$qgis_app/Contents"
    # An active Conda environment otherwise supplies incompatible GDAL plugins
    # and PROJ databases to the signed QGIS application bundle.
    unset PYTHONHOME PROJ_LIB GDAL_DRIVER_PATH QGIS_PREFIX_PATH
    unset DYLD_LIBRARY_PATH DYLD_FALLBACK_LIBRARY_PATH QT_QPA_PLATFORM_PLUGIN_PATH
    export GDAL_DATA="$qgis_contents/Resources/qgis/gdal"
    export PROJ_DATA="$qgis_contents/Resources/qgis/proj"
    export QT_PLUGIN_PATH="$qgis_contents/PlugIns"
    export PYTHONPATH="$project_root/src:$qgis_contents/Resources/qgis/python:$qgis_contents/Resources/qgis/python/plugins"
    export PATH="$qgis_contents/MacOS:/usr/bin:/bin:/usr/sbin:/sbin"
    exec "$qgis_contents/MacOS/python" "$@"
fi

# Keep externally configured PyQGIS/GDAL/PROJ paths on Linux and other systems.
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${QGIS_PYTHON:-python3}" "$@"
