"""Setup script for the package."""

import setuptools

VERSION = "0.1.0"

with open("README.md", "r", encoding="utf-8") as file_object:
  LONG_DESCRIPTION = file_object.read()

with open("requirements.txt", "r", encoding="utf-8") as file_object:
  INSTALL_REQUIRES = [
      line.strip()
      for line in file_object.read().splitlines()
      if line.strip() and not line.startswith("#")
  ]

setuptools.setup(
    name="wav2x",
    version=VERSION,
    author="Quan Wang",
    author_email="quanw@google.com",
    description=(
        "Audio representations and TFLite model inference for"
        " speaker-id and lang-id"
    ),
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/wq2012/wav2x",
    packages=setuptools.find_packages(exclude=["tests*", "demos*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=INSTALL_REQUIRES,
)
