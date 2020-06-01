#!/usr/bin/env bash
# Recipes for training the released models.
# TODO(gustavoauma): Switch to Python so that it is easier to check that
# the URLs are valid and also that training work as intended.


MODELS_DIR=models/lemmatizer
TMP_DIR=tmp_data

mkdir -p $TMP_DIR

# en
(cd $TMP_DIR &&
 curl -LJO https://raw.githubusercontent.com/unimorph/eng/master/eng &&
 curl -LJO https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-train.conllu)
mkdir -p $MODELS_DIR/en
python -m glemmazon.train_lemmatizer \
  --conllu $TMP_DIR/en_ewt-ud-train.conllu \
  --unimorph $TMP_DIR/eng \
  --mapping data/tag_mapping.csv \
  --model $MODELS_DIR/en/

# pt
(cd $TMP_DIR &&
 curl -LJO https://raw.githubusercontent.com/unimorph/por/master/por &&
 curl -LJO https://raw.githubusercontent.com/UniversalDependencies/UD_Portuguese-GSD/master/pt_gsd-ud-train.conllu)
mkdir -p $MODELS_DIR/pt
python -m glemmazon.train_lemmatizer \
  --conllu $TMP_DIR/pt_gsd-ud-train.conllu \
  --unimorph $TMP_DIR/por \
  --mapping data/tag_mapping.csv \
  --model $MODELS_DIR/pt

# nl
(cd $TMP_DIR &&
 curl -LJO https://raw.githubusercontent.com/unimorph/nld/master/nld &&
 curl -LJO https://raw.githubusercontent.com/UniversalDependencies/UD_Dutch-Alpino/master/nl_alpino-ud-train.conllu)
mkdir -p $MODELS_DIR/nl
python -m glemmazon.train_lemmatizer \
  --conllu $TMP_DIR/nl_alpino-ud-train.conllu \
  --unimorph $TMP_DIR/nld \
  --mapping data/tag_mapping.csv \
  --model $MODELS_DIR/nl

