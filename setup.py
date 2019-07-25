from setuptools import setup

setup(name='glemmazon',
      version='0.2',
      description='Simple Python lemmatizer for several languages',
      url='http://github.com/gustavoauma/glemmazon',
      author='Gustavo Mendonça',
      author_email='gustavoauma@gmail.com',
      license='MIT',
      packages=['glemmazon'],
      install_requires=[
            'absl-py==0.7.1',
            'Keras==2.2.4',
            'pandas==0.24.2',
            'pyconll==2.0.0',
            'scikit-learn==0.21.2',
            'sklearn==0.0',
            # TODO(gustavoauma): Find out which tensorflow packages are
            # actually needed here.
            'tqdm==4.32.2',
      ],
      zip_safe=False)