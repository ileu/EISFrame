import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="eisplottingtool",
    version="0.0.1",
    author="sauter ulrich",
    author_email="usauterv@outlook.com",
    description="A tool used to plot EIS data and other battery related data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ileu/EISFrame",
    project_urls={
        "Bug Tracker": "https://github.com/ileu/EISFrame/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=[
        "numpy",
        "pandas",
        "eclabfiles>=0.3.9",
        "pint",
        "scipy",
        "matplotlib",
    ],
    python_requires=">=3.9",
)