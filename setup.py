# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 15:23:12 2018

@author: Dany
"""

from setuptools import setup, find_packages

from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))


with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    
    name='Lstm variationnal Auto-encoder', 
    version='1.1.0',  
    url='https://github.com/pypa/sampleproject',  
    author='Danyleb',  
    classifiers=[  
     
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        
    ],

    packages=find_packages('.', include=['LstmVAE', 'LstmVAE.*']),  

    project_urls={  
        'Bug Reports': 'https://github.com/pypa/sampleproject/issues',
        'Funding': 'https://donate.pypi.org',
        'Say Thanks!': 'http://saythanks.io/to/example',
        'Source': 'https://github.com/pypa/sampleproject/',
    },
)
