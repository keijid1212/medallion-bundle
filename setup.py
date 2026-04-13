from setuptools import setup, find_packages

setup(
    name="medallion",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pyspark>=3.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-spark>=0.6.0",
            "delta-spark>=2.4.0",
        ]
    },
)
