#!/usr/bin/env python3
import os
import sys
import site
import setuptools
from distutils.core import setup


site.ENABLE_USER_SITE = '--user' in sys.argv[1:]

with open('README.md') as f:
    long_description = f.read()

with open(os.path.join('facenet', 'version.txt')) as f:
    version = f.read().strip()

setup(
    name='facenet',
    version=version,
    description='Real and fake face detector in PyTorch.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Mohammadamin Aliari',
    author_email='maminaliari@gmail.com',
    packages=setuptools.find_packages(),
    package_data={'facenet': ['version.txt']},
    install_requires=[
        'pytest',
        'numpy',
        'Pillow',
        'torch',
        'torchvision',
        'scikit-learn',
        'pandas',
        'seaborn',
        'matplotlib',
    ],
    python_requires='>=3.7',
    url='https://aminaliari.github.io/',
)
