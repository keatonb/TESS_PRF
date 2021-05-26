import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = fh.read().splitlines()

setuptools.setup(
    name="TESS_PRF",
    version="0.1.0",
    author="Keaton Bell",
    author_email="keatonbell@utexas.edu",
    description="Tools to display the TESS pixel response function (PRF) at any location on the detector",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/keatonb/TESS_PRF",
    project_urls={
        "Bug Tracker": "https://github.com/keatonb/TESS_PRF/issues",
    },
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
