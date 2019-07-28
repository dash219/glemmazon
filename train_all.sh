#!/usr/bin/env bash
# Recipes for training the released models.
# TODO(gustavoauma): Switch to Python so that it is easier to check that
# the URLs are valid and also that training work as intended.

TMP_DIR=tmp_data
MODELS_DIR=models

mkdir -p $TMP_DIR

# en.pkl
(cd $TMP_DIR &&
 curl -LJO https://raw.githubusercontent.com/unimorph/eng/master/eng &&
 curl -LJO https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-train.conllu)
python -m glemmazon.train_lemmatizer \
  --conllu $TMP_DIR/en_ewt-ud-train.conllu \
  --unimorph $TMP_DIR/eng \
  --model $MODELS_DIR/en.pkl > $MODELS_DIR/en.log

# pt.pkl
(cd $TMP_DIR &&
 curl -LJO https://raw.githubusercontent.com/unimorph/por/master/por &&
 curl -LJO https://raw.githubusercontent.com/UniversalDependencies/UD_Portuguese-GSD/master/pt_gsd-ud-train.conllu)
python -m glemmazon.train_lemmatizer \
  --conllu $TMP_DIR/pt_gsd-ud-train.conllu \
  --unimorph $TMP_DIR/por \
  --model $MODELS_DIR/pt.pkl > $MODELS_DIR/pt.log
