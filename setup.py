from setuptools import setup, find_packages

setup(
    name="sequence",
    version="0.1",
    author="Ritchie Vink",
    author_email="ritchie46@gmail.com",
    url="https://github.com/ritchie46/sequence",
    license="mit",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.17.3",
        "torch>=1.3.0",
        "tensorboardX",
        "dumpster",
        "patool~=1.12",
        "pyunpack~=0.1.2",
        "nltk>=3.4.4",
        "dask[complete]",
    ],
    python_requires=">=3.6, <3.8",
    dependency_links=["https://github.com/ritchie46/dumpster#egg=dumpster"],
)
