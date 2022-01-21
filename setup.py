"""
Installs:
    - ocrd-detectron2-segment
"""

import codecs
import json
from setuptools import setup
from setuptools import find_packages

with codecs.open('README.md', encoding='utf-8') as f:
    README = f.read()

with open('./ocrd-tool.json', 'r') as f:
    version = json.load(f)['version']
    
setup(
    name='ocrd_detectron2',
    version=version,
    description='OCR-D wrapper for detectron2 based segmentation models',
    long_description=README,
    long_description_content_type='text/markdown',
    author='Robert Sachunsky, Julian Balling',
    author_email='sachunsky@informatik.uni-leipzig.de, balling@infai.org',
    url='https://github.com/bertsky/ocrd_detectron2',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    install_requires=open('requirements.txt').read().split('\n'),
    # dependency links not supported anymore (must use pip install -f ... now)
    dependency_links=[
        'https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html',
        'https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.10/index.html',
        'https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html',
        'https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html',
    ],
    package_data={
        '': ['*.json', '*.yml', '*.yaml', '*.csv.gz', '*.jar', '*.zip'],
    },
    entry_points={
        'console_scripts': [
            'ocrd-detectron2-segment=ocrd_detectron2.cli:ocrd_detectron2_segment',
        ]
    },
)
