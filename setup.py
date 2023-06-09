from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


def read_requirements(file):
    with open(file) as f:
        return f.read().splitlines()


setup(
    name="spacebench",
    version="0.0.1",
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
        "Spatial confounding poses a significant challenge in scientific ",
        "studies involving spatial data, where unobserved spatial ",
        "variables can influence both treatment and outcome, possibly ",
        "leading to spurious associations. To address this problem, ",
        "SpaCE provides realistic benchmark datasets and tools ",
        "for systematically evaluating causal inference methods designed to ",
        "alleviate spatial confounding. Each dataset includes training data, ",
        "true counterfactuals, a spatial graph with coordinates, and a ",
        "smoothness and confounding scores characterizing the effect of a ",
        "missing spatial confounder. The datasets cover real treatment and ",
        "covariates from diverse domains, including climate, health and social ",
        "sciences. Realistic semi-synthetic outcomes and counterfactuals are ",
        "generated using state-of-the-art machine learning ensembles, following ",
        "best practices for causal inference benchmarks. SpaCE facilitates ",
        "an automated end-to-end machine learning pipeline, simplifying data ",
        "loading, experimental setup, and model evaluation."
    ),
    long_description_content_type="text/markdown",
    # entry_points={"console_scripts": ["spacebench=spacebench.api.cli:main"]},
    # url="https://github.com/NSAPH-Projects/space", # cli api needs update
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
