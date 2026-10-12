from pathlib import Path

from setuptools import find_namespace_packages, setup


PACKAGE_ROOT = Path(__file__).resolve().parent
requirements = [
    line.strip()
    for line in (PACKAGE_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]

setup(
    name="tnet_analysis",
    version="1.0.2",
    description="Temporal-network generation, integration, dynamism and structure measures",
    long_description=(PACKAGE_ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    license_expression="MIT",
    license_files=["LICENSE"],
    python_requires=">=3.9",
    packages=find_namespace_packages(
        include=["tnet_analysis", "tnet_analysis.example"],
    ),
    include_package_data=False,
    install_requires=requirements,
    extras_require={
        "parallel": ["joblib", "tqdm"],
        "test": ["joblib", "networkx"],
        "docs": ["sphinx>=7,<8"],
        "release": ["build", "twine"],
    },
    classifiers=[
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
    ],
)
