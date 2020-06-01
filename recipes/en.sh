#!/usr/bin/env bash
# Recipe for training the released models for "en".
DATA_DIR=data
MODEL_DIR=models/lemmatizer/en
TMP_DIR=$MODEL_DIR/tmp
CONLLU_URL=https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-train.conllu

mkdir -p $MODEL_DIR
mkdir -p $TMP_DIR

(cd $TMP_DIR && curl -LJO $CONLLU_URL)
python -m glemmazon.train_lemmatizer \
  --conllu $TMP_DIR/$(basename $CONLLU_URL) \
  --mapping data/tag_mapping.csv \
  --exceptions $DATA_DIR/en_exceptions.csv \
  --model $MODEL_DIR
