# -*- coding: utf-8 -*-
from setuptools import setup

setup(name='glemmazon',
      version='0.4',
      description='Simple Python lemmatizer for several languages',
      url='http://github.com/gustavoauma/glemmazon',
      author='Gustavo Mendonça',
      author_email='gustavoauma@gmail.com',
      license='MIT',
      packages=['glemmazon'],
      install_requires=[
            'absl-py==0.7.1',
            'numpy==1.16.4',
            'pandas==0.24.2',
            'pyconll==2.0.0',
            'scikit-learn==0.21.2',
            'sklearn==0.0',
            'tensorflow==2.0.0-beta1',
            'tqdm==4.32.2',
      ],
      zip_safe=False)