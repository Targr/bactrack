from setuptools import setup, find_packages

setup(
    name="bactracker",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "opencv-python", "numpy", "matplotlib", "trackpy",
        "pandas", "ipywidgets", "scikit-learn", "tifffile", "shapely"
    ],
    python_requires=">=3.7",
)
