from setuptools import setup, find_packages

VERSION = "1.3" 
DESCRIPTION = "Helpful modules for pytorch"
LONG_DESCRIPTION = "Mini package, machine learning help modules for pytorch"

# Setting up
setup(
        name="ml_modules", 
        version=VERSION,
        author="Davide Wiest",
        author_email="<jasnadw03@gmail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[
            "torch",
            "numpy",
            "pandas",
            "matplotlib",
            "torchvision",
            "tqdm",
        ],
        keywords=["pytorch", "helping functions"],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)
