from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="spacebench", 
    version="0.0.1",
    author=("Mauricio Tec, Ana Trisovic, Audirac, Michelle, Jie Hu," 
            "Sophie Mirabai Woodward, Naeem Khoshnevis, Francesca Dominici"),
    author_email=("mauriciogtec@hsph.harvard.edu,"
                  "anatrisovic@g.harvard.edu,"
                  "maudirac@hsph.harvard.edu,"
                  "khu@hsph.harvard.edu,"
                  "swoodward@fas.harvard.edu,"
                  "nkhoshnevis@g.harvard.edu,"
                  "fdominic@hsph.harvard.edu"),
    maintainer="Naeem Khoshnevis",
    maintainer_email = "nkhoshnevis@g.harvard.edu",
    description= ("spacebench offers benchmark datasets for causal inference, " 
                  "incorporating real confounder and exposure data from "
                  "environmental health studies. Synthetic outcomes are "
                  "generated using machine learning, with spatial confounding "
                  "introduced by masking key confounders."),
    long_description_content_type="text/markdown",
    entry_points={
        "console_scripts": ["spacebench=spacebench.api.cli:main"]
    },
    url="https://github.com/NSAPH-Projects/space",
    license="MIT",
    packages=find_packages(exclude=['tests*', 'scripts*', 'notebooks*'
                                    'examples*']),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    python_requires='>=3.7',
    package_data={"spacebench": ["datasets/*.csv"]},
    include_package_data=True,
)