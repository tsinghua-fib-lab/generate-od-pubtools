import setuptools

with open("README.md", "r") as f:
    long_description = f.read()


setuptools.setup(
    name="generate-od",
    version="0.1",
    author="Can Rong",
    author_email="276413973@qq.com",
    description="A tool to generate origin-destination matrix for any given area.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/",
    packages=setuptools.find_packages(),

    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",

        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',\
        
        'Topic :: Scientific/Engineering :: GIS',
        'Topic :: Scientific/Engineering :: Urban Science',
        'Topic :: Scientific/Engineering :: Computer Science',
        'Topic :: Scientific/Engineering :: Urban Planning',
        'Topic :: Scientific/Engineering :: Transportation',
        'Topic :: Scientific/Engineering :: Urban Evironment',
    ],

    install_requires=[
        'geopandas',
        'open-clip-torch>=2.23.0',
        'huggingface-hub',
        'numpy',
        'torch>=2.1.0',
        'Pillow',
        'scipy',
        'shapely',
        'scikit-learn',
        'contextily',
        'matplotlib',
        'rasterio'
    ],
    python_requires='>=3.8',
)