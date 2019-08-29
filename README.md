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
## Lemmatizer
The main class is [`Lemmatizer`](./glemmazon/lemmatizer.py). It 
provides a single interface for getting the lemmas, under `__call__`:
```python
>>> from glemmazon import Lemmatizer
>>> lemmatizer = Lemmatizer.from_path('models/lemmatizer/en.pkl')
>>> lemmatizer('loved', 'VERB')
'love'
>>> lemmatizer('cars', 'NOUN')
'car'
```

### Training a new model
Basic setup
```bash
$ python -m glemmazon.train_lemmatizer \
  --conllu data/en_ewt-ud-train.conllu \
  --model models/en.pkl
```

Include a dictionary with exceptions:
```bash
$ python -m glemmazon.train_lemmatizer \
  --conllu data/en_ewt-ud-train.conllu \
  --model models/en.pkl \
  --exceptions data/en_exceptions.csv
```

For other options, please see the flags defined in 
[train_lemmatizer.py](./glammatizer/train_lemmatizer.py).

## Inflector
The main class is [`Inflector`](./glemmazon/inflector.py). It 
provides a single interface for getting the inflected forms, under 
`__call__`:
```python
>>> from glemmazon import Inflector
>>> inflector = Inflector.from_path('models/inflector/pt_inflec_md.pkl')
>>> inflector('amar', aspect='IMP', mood='SUB', number='PLUR', person='3', tense='PAST')
'amassem'
```

### Training a new model
Basic setup
```bash
$ python -m glemmazon.train_inflector \
  --unimorph data/por \
  --mapping data/tag_mapping.csv \
  --model models/pt_inflect.pkl
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
