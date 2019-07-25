# glemmazon
Simple Python lemmatizer for several languages

# Installation
## Requirements
Note: glemmazon depends on Tensorflow. Please refer to their 
[installation guide](https://www.tensorflow.org/install/).

The latest version of glemmazon is available over pip.
```bash
$ pip install glemmazon 
```

# Usage
## Basic
The main class is [`Lemmatizer`](./glemmazon/lemmatizer.py). It 
provides a single interface for getting the lemmas, under `__call__`:
```python
>>> from glemmazon import Lemmatizer
>>> l = Lemmatizer()
>>> l.load('models/en.pkl')
>>> l(['loved', 'works', 'cars', 'was'], ['VERB', 'VERB', 'NOUN', 'VERB'])
['love', 'work', 'car', 'be']
```

## Training a new model
### Basic usage
```bash
$ python -m glemmazon.train \
  --conllu data/en_ewt-ud-train.conllu \
  --model models/en.pkl
```

### Include a dictionary with exceptions
```bash
$ python -m glemmazon.train \
  --conllu data/en_ewt-ud-train.conllu \
  --model models/en.pkl \
  --exceptions data/en_exceptions.csv
```

### Other
For other options, please see the flags defined in [train.py](
./glammatizer/train.py).

# License
Please note that this project contains two different licenses:

- Pickled models trained over [UniversalDependencies](
  http://github.com/UniversalDependencies), i.e. files under 
  [models/](./models/), are licensed under the terms of the [GNU General 
  Public License version 3](./models/!LICENSE).
  
- Everything else (.py scripts, exception lists in .csv, etc.) is 
  licensed under the terms of [MIT license](./LICENSE).
