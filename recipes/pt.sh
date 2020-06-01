#!/usr/bin/env bash
# Recipe for training the released models for "pt".
DATA_DIR=data
MODEL_DIR=models/lemmatizer/pt
TMP_DIR=$MODEL_DIR/tmp
CONLLU_URL=https://raw.githubusercontent.com/UniversalDependencies/UD_Portuguese-Bosque/master/pt_bosque-ud-train.conllu

mkdir -p $MODEL_DIR
mkdir -p $TMP_DIR

(cd $TMP_DIR && curl -LJO $CONLLU_URL)
python -m glemmazon.train_lemmatizer \
  --conllu $TMP_DIR/$(basename $CONLLU_URL) \
  --mapping data/tag_mapping.csv \
  --exceptions $DATA_DIR/pt_exceptions.csv \
  --model $MODEL_DIR
