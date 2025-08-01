from setuptools import setup, find_packages

setup(
    name="bactracker",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "trackpy",
        "opencv-python",
        "tifffile",
        "scikit-learn",
        "shapely",
        "ipywidgets"
    ],
    author="Robert Targ",
    description="Bacterial motility tracking and classification toolkit",
    license="FIGURE OUT",
)
