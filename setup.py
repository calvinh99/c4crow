from setuptools import setup, find_packages

setup(
    name="c4crow",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        'numpy>=1.21.0,<2.0.0',
        'pandas',
        'matplotlib',
        'torch',
        'PyYAML>=6.0.0,<7.0.0',
    ],
    tests_require=['pytest'],
    setup_requires=['pytest-runner'],
    extras_require={
        'test': ['pytest'],
    },
)