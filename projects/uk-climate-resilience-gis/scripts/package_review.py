"""Create a local review bundle with relative paths, verified members and no raw cache."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    validation = json.loads((ROOT/'outputs/qa/validation.json').read_text())
    if validation['status'] != 'passed':
        raise ValueError('A passing final real-data validation report is required before packaging')
    attested = validation.get('artifact_sha256', {})
    if not attested or not all(name in attested for name in
                               ['data/processed/resilience.gpkg',
                                'outputs/tables/priority_shortlist.csv',
                                'outputs/portfolio/interactive.html']):
        raise ValueError('The validation report lacks a complete delivery fingerprint')
    for name, expected in attested.items():
        path = ROOT/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Validated analytical artifact changed: '+name)
    export_qa = json.loads((ROOT/'metadata/qgis_pdf_qa.json').read_text())
    if export_qa.get('status') != 'passed' or export_qa.get('visual_review', {}).get('status') != 'passed':
        raise ValueError('Passing export checks and recorded visual review are required before packaging')
    reviewed = {'qgis/project/gm_climate_resilience.qgz': export_qa['project_sha256'],
                'data/processed/resilience.gpkg': export_qa['input_sha256']}
    for row in export_qa['pdfs']:
        reviewed[row['file']] = row['sha256']
        reviewed[row['png']['file']] = row['png']['sha256']
    for name, expected in reviewed.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Export review is stale for '+name)
    files = [ROOT/name for name in ['README.md', 'LICENSE', 'DATA_LICENSES.md', 'CITATION.cff', 'Makefile', 'pyproject.toml', '.gitignore', 'environment/detected.json']]
    files += [ROOT/'docs'/name for name in ['client_brief.md', 'data_dictionary.md', 'data_sources.md',
                                            'limitations.md', 'methodology.md', 'qgis_project.md',
                                            'reproduction.md', 'validation.md']]
    for directory in ['config', 'src', 'scripts', 'tests', 'metadata', 'qgis',
                      'data/processed', 'outputs/maps', 'outputs/figures', 'outputs/tables', 'outputs/portfolio', 'outputs/qa']:
        files += [path for path in (ROOT/directory).rglob('*') if path.is_file() and
                  '__pycache__' not in path.parts and path.suffix not in {'.log', '.pyc'} and
                  path.name not in {'.DS_Store', 'Thumbs.db'} and not path.name.startswith('._') and
                  not path.name.endswith('~')]
    files = sorted(set(files))
    destination = ROOT/'outputs/release/gm_climate_resilience_review.zip'
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, arcname='gm-climate-resilience/'+str(path.relative_to(ROOT)))
        archive.writestr('gm-climate-resilience/SHA256SUMS.txt', ''.join(f'{digest}  {name}\n' for name, digest in manifest.items()))
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError('Bundle CRC validation failed')
        for name, expected in manifest.items():
            if hashlib.sha256(archive.read('gm-climate-resilience/'+name)).hexdigest() != expected:
                raise ValueError('Bundled member differs from source: '+name)
    print(json.dumps({'archive': str(destination.relative_to(ROOT)), 'bytes': destination.stat().st_size,
                      'members':len(files)+1,'sha256':hashlib.sha256(destination.read_bytes()).hexdigest()},indent=2))


if __name__ == '__main__':
    main()
