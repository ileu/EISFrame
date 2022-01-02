import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
        name="eisplottingtool",
        version="0.0.1",
        author="Ulrich Sauter",
        author_email="usauterv@outlook.com",
        description="A tool used to plot EIS data and other battery related data",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/ileu/EISFrame",
        project_urls={
            "Bug Tracker": "https://github.com/ileu/EISFrame/issues",
            },
        classifiers=[
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: GNU General Public License v3 or later (""GPLv3+)",
            "Operating System :: OS Independent",
            "Topic :: Scientific/Engineering :: Chemistry",
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
            "schemdraw>=0.12"
            ],
        python_requires=">=3.9",
        )
