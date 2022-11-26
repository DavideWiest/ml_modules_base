from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Helpful modules for pytorch'
LONG_DESCRIPTION = 'Machine learning help modules for pytorch'

# Setting up
setup(
        name="ml_modules", 
        version=VERSION,
        author="Jason Dsouza",
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
        keywords=['pytorch', 'helping functions'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)