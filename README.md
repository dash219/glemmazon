# glemmazon
Simple Python lemmatizer and morphological generator for several 
languages.

![Version](https://img.shields.io/badge/version-0.3-red)
![Release Status](https://img.shields.io/badge/release-unstable-red)
![Commit Activity](https://img.shields.io/github/commit-activity/m/gustavoauma/glemmazon)

# Installation
The latest version of glemmazon is available over pip.
```bash
$ pip install glemmazon 
```

Note: glemmazon depends on Tensorflow. Please refer to their 
[installation guide](https://www.tensorflow.org/install/). Other
dependencies are already included in the pip package.

# Usage
## Analyzer
The main class is [`Analyzer`](./glemmazon/pipeline.py). It provides a 
single interface for getting the morphological attributes, under 
`__call__`:
```python
>>> from glemmazon import Analyzer
>>> analyzer = Analyzer.load('models/analyzer/pt')
>>> analyzer(word='carros', pos='NOUN')
{'case': '_UNSP', 'definite': '_UNSP', 'degree': '_UNSP', 
'foreign': '_UNSP', 'gender': 'masc', 'mood': '_UNSP', 'number': 'plur',
 'numtype': '_UNSP', 'person': '_UNSP', 'polarity': '_UNSP',
 'pos': 'NOUN', 'prontype': '_UNSP', 'reflex': '_UNSP', 
 'tense': '_UNSP', 'verbform': '_UNSP', 'voice': '_UNSP'}
```

### Training a new model
Basic setup
```bash
$ python -m glemmazon.train_analyzer \
  --conllu data/en_ewt-ud-train.conllu \
  --model models/analyzer/en
```

## Lemmatizer
The main class is [`Lemmatizer`](./glemmazon/pipeline.py). It 
provides a single interface for getting the lemmas, under `__call__`:
```python
>>> from glemmazon import Lemmatizer
>>> lemmatizer = Lemmatizer.load('models/lemmatizer/en')
>>> lemmatizer(word='loved', pos='VERB')
'love'
>>> lemmatizer(word='cars', pos='NOUN')
'car'
```

### Training a new model
Basic setup
```bash
$ python -m glemmazon.train_lemmatizer \
  --conllu data/en_ewt-ud-train.conllu \
  --model models/lemmatizer/en
```

Include a dictionary with exceptions:
```bash
$ python -m glemmazon.train_lemmatizer \
  --conllu data/en_ewt-ud-train.conllu \
  --exceptions data/en_exceptions.csv \
  --model models/lemmatizer/en
```

For other options, please see the flags defined in 
[train_lemmatizer.py](./glammatizer/train_lemmatizer.py).

## Inflector
The main class is [`Inflector`](./glemmazon/pipeline.py). It 
provides a single interface for getting the inflected forms, under 
`__call__`:
```python
>>> from glemmazon import Inflector
>>> inflector = Inflector.from_path('models/inflector/pt_inflec_md.pkl')
>>> inflector(word='amar', aspect='IMP', mood='SUB', number='PLUR', person='3', tense='PAST')
'amassem'
```

### Training a new model
Basic setup
```bash
$ python -m glemmazon.train_inflector \
  --conllu data/pt_bosque-ud-train.conllu \
  --model models/inflector/pt
```

For other options, please see the flags defined in 
[train_inflector.py](./glammatizer/train_inflector.py).

# License
Please note that this project contains two different licenses:

- Pickled models trained over [UniversalDependencies](
  http://github.com/UniversalDependencies), i.e. files under 
  [models/](./models/), are licensed under the terms of the [GNU General 
  Public License version 3](./models/!LICENSE).
  
- Everything else (.py scripts, exception lists in .csv, etc.) is 
  licensed under the terms of [MIT license](./LICENSE).

# Development
## Run all unittests
```bash
glemmazon$ python -m unittest -v
```
## Run a single unittest module
```bash
glemmazon$ python -m unittest $MODULE
```

For example
```bash
glemmazon$ python -m unittest glemmazon.tests.test_encoder
```