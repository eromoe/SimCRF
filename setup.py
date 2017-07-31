# -*- coding: utf-8 -*-
# @Author: mithril

from __future__ import unicode_literals, print_function, absolute_import

import sys
from os import path
from codecs import open
from setuptools import setup, find_packages


here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    install_requires = f.read().split('\n')


setup(
    name='simcrf',
    version='0.1.1',

    description='simple and quick crf wrapper for crfsuite',
    long_description=long_description,


    author='Mithril | eromoe',
    author_email='eromoe@users.noreply.github.com',
    url = 'https://github.com/eromoe/SimCRF',

    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',

        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Environment :: Web Environment',
        'License :: OSI Approved :: MIT License',

        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],

    keywords='crf crfsuite',

    packages=find_packages(exclude=['experiments', 'tests', 'data']),

    install_requires=install_requires,

)

