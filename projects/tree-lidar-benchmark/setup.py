"""Build configuration for including verified retained evidence."""

import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as SetuptoolsBuildPy

SOURCE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE_ROOT))

from build_support import copy_retained_evidence  # noqa: E402


class BuildPy(SetuptoolsBuildPy):
    """Add manifest-verified evidence to the built package tree."""

    def run(self) -> None:
        super().run()
        package_root = Path(self.build_lib) / "tree_lidar_benchmark"
        copy_retained_evidence(SOURCE_ROOT, package_root)


setup(cmdclass={"build_py": BuildPy})
