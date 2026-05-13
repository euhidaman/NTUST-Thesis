"""
EmberVLM Setup Script

Note: This project primarily uses requirements.txt for dependency management.
      Run: pip install -r requirements.txt

      This setup.py is provided for optional editable installs if needed.
"""

import os
from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Read README.md if it exists
long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="embervlm",
    version="1.0.0",
    author="EmberVLM Team",
    description="Tiny Multimodal VLM for Robot Fleet Selection with Incident Reasoning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "edge": [
            "onnxruntime>=1.16.0",
            "psutil>=5.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "embervlm-train=scripts.train_all:main",
            "embervlm-eval=scripts.evaluate:main",
            "embervlm-deploy=scripts.deploy:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="vision-language-model, edge-ai, robot-selection, multimodal",
)

