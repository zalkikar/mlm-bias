#!/usr/bin/python
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(
    name='mlm-bias',
    version='0.1.4',
    author='Rahul Zalkikar',
    author_email='rayzck9@gmail.com',
    description='Bias Evaluation Methods for Masked Language Models implemented in PyTorch',
    license='MIT',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/zalkikar/mlm-bias',
    project_urls={
        'Bug Tracker': 'https://github.com/zalkikar/mlm-bias/issues',
    },
    python_requires=">=3.8, <3.11",
    packages=find_packages(),
    install_requires=[
        "transformers>=4.35.0",
        "numpy>=1.23.5,<2",
        "pandas>=2.0.3",
        "torch>=2.1.0",
        "regex>=2023.3.23"
    ],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        'License :: OSI Approved :: MIT License',
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)