# Models
## en
- Latest release: 2019-07-25
- Training data: 
  - CoNLL-U: [en_ewt-ud-train.conllu](
    http://github.com/UniversalDependencies/UD_English-EWT)
  - UniMorph: [eng](https://github.com/unimorph/eng)
- Exceptions dict: [en_exceptions.csv](../data/en_exceptions.csv)

```
MODELS CONFIG
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 27, 16)            4096      
_________________________________________________________________
bidirectional_1 (Bidirection (None, 32)                4224      
_________________________________________________________________
dropout_1 (Dropout)          (None, 32)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 89)                2937      
=================================================================
Total params: 11,257
Trainable params: 11,257
Non-trainable params: 0


SUFFIX IDENTIFICATION
---------------------
Train on 89661 samples, validate on 22416 samples
[...]
Epoch 10/10
89661/89661 [==============================] - 142s 2ms/step - 
loss: 0.1739 - acc: 0.9522 - val_loss: 0.1549 - val_acc: 0.9545

INDEX IDENTIFICATION
--------------------
Train on 89661 samples, validate on 22416 samples
[...]
Epoch 10/10
89661/89661 [==============================] - 142s 2ms/step - 
loss: 0.1739 - acc: 0.9522 - val_loss: 0.1549 - val_acc: 0.9545
```

## en_large
- Latest release: 2019-07-25
- Training data: 
  - CoNLL-U: [en_ewt-ud-train.conllu](
    http://github.com/UniversalDependencies/UD_English-EWT)
  - UniMorph: [eng](https://github.com/unimorph/eng)
- Exceptions dict: 5,596 (losses from the model)
- Model config: Same as `en`

## pt
- Latest release: 2019-07-25
- Training data: 
  - CoNLL-U: [pt_gsd-ud-train.conllu](
    http://github.com/UniversalDependencies/UD_Portuguese-GSD)
  - UniMorph: [por](https://github.com/unimorph/por)
- Exceptions dict: None

```
MODELS CONFIG
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 26, 16)            4096      
_________________________________________________________________
bidirectional_1 (Bidirection (None, 32)                4224      
_________________________________________________________________
dropout_1 (Dropout)          (None, 32)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 67)                2211      
=================================================================
Total params: 10,531
Trainable params: 10,531
Non-trainable params: 0


SUFFIX IDENTIFICATION
---------------------
Train on 189318 samples, validate on 47330 samples
[...]
Epoch 10/10
189318/189318 [==============================] - 288s 2ms/step - 
loss: 0.1934 - acc: 0.9474 - val_loss: 0.1693 - val_acc: 0.9521


INDEX IDENTIFICATION
--------------------
Train on 189318 samples, validate on 47330 samples
[...]
Epoch 10/10
189318/189318 [==============================] - 289s 2ms/step - 
loss: 0.1343 - acc: 0.9655 - val_loss: 0.1224 - val_acc: 0.9667
```

## pt_large
- Latest release: 2019-07-25
- Training data: 
  - CoNLL-U: [pt_gsd-ud-train.conllu](
    http://github.com/UniversalDependencies/UD_Portuguese-GSD)
  - UniMorph: [por](https://github.com/unimorph/por)
- Exceptions dict: 13,640 (losses from the model)
- Model config: Same as `pt`
