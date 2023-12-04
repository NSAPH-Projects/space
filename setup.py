from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


def read_requirements(file):
    with open(file) as f:
        return f.read().splitlines()


setup(
    name="spacebench",
    version="0.1.4",
    author=(
        "Mauricio Tec, Ana Trisovic, Audirac, Michelle, Jie Hu,"
        "Sophie Mirabai Woodward, Naeem Khoshnevis, Francesca Dominici"
    ),
    author_email=(
        "mauriciogtec@hsph.harvard.edu,"
        "anatrisovic@g.harvard.edu,"
        "maudirac@hsph.harvard.edu,"
        "khu@hsph.harvard.edu,"
        "swoodward@fas.harvard.edu,"
        "nkhoshnevis@g.harvard.edu,"
        "fdominic@hsph.harvard.edu"
    ),
    maintainer="Naeem Khoshnevis",
    maintainer_email="nkhoshnevis@g.harvard.edu",
    description=(
        "Spatial confounding poses a significant challenge in scientific studies where unobserved spatial variables influence both treatment and outcome, leading to spurious associations. SpaCE provides realistic benchmark datasets and tools for systematically valuating causal inference methods for spatial confounding. Each dataset includes training data with spatial confounding, true counterfactuals, a spatial graph with coordinates, and realistic semi-synthetic outcomes."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NSAPH-Projects/space", # cli api needs update
    license="MIT",
    packages=find_packages(exclude=["tests*", "scripts*", "notebooks*", "examples*"]),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    python_requires=">=3.10",
    package_data={"spacebench": ["datasets/*.csv"]},
    include_package_data=True,
    install_requires=read_requirements('requirements.txt'),  # Normal dependencies
    extras_require={
        'all': read_requirements('optional-requirements.txt')  # Optional dependencies
    }
)
