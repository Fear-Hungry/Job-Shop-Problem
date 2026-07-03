from setuptools import setup, find_packages

setup(
    name="jssp_solver",
    version="0.1.0",
    author="Marcus Vinicius, Eduardo Augusto Moreira",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "ortools",
        "pytest",
        "pytest-cov"
    ],
)
