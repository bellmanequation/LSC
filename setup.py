from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup

description = """set up for LSC in MAgent
"""

setup(
    name='LSC',
    version='0.1',
    description='LSC: Learning to structured communication',
    long_description=description,
    license='MIT License',
    packages=[
        'graph_nets',
    ],
    install_requires=[
        'absl-py==0.7.0',
        'tensorflow-gpu==2.9.3',
        'tensorflow-probability==0.5.0',
        'tensorflow-probability-gpu==0.4.0',
        'numpy>=1.10',
        'dm-sonnet==1.23',
        'sklearn',
        'networkx',
        'pandas'
    ],
)
