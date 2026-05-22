"""Setup script for the HyMem package."""

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [
        line.strip() for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="hymem",
    version="0.2.0",
    description="HyMem: Hybrid Memory Architecture with Dynamic Retrieval Scheduling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["hymem", "hymem.*"]),
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="ai, memory, retrieval, llm, agent, hymem",
    entry_points={
        "console_scripts": [
            "hymem=hymem.cli:main",
        ],
    },
)