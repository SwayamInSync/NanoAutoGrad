import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="NanoAutoGrad",
    version="0.1.2",
    author="Swayam Singh",
    author_email="singhswayam008@gmail.com",
    description="A miniature implementation of PyTorch's autograd",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/practice404/NanoAutoGrad",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
