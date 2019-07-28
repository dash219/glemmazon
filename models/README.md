# Models

## Lemmatizer
### en
- Latest release: 2019-07-29
- Training data: 
  - CoNLL-U: [en_ewt-ud-train.conllu](
    http://github.com/UniversalDependencies/UD_English-EWT)
  - UniMorph: [eng](https://github.com/unimorph/eng)
- Exceptions dict: 3,658 entries (from --no_losses)

```
MODELS CONFIG
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
bidirectional_1 (Bidirection (None, 32)                15232     
_________________________________________________________________
dropout_1 (Dropout)          (None, 32)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 88)                2904      
=================================================================
Total params: 18,136
Trainable params: 18,136
Non-trainable params: 0


SUFFIX IDENTIFICATION (val: 0.2 = 22,318 examples)
---------------------
Epoch 25/25
 - 102s - loss: 0.0614 - acc: 0.9836 - val_loss: 0.0507 - val_acc: 0.9862

INDEX IDENTIFICATION (val: 0.2 = 22,318 examples)
--------------------
Epoch 25/25
 - 102s - loss: 0.0883 - acc: 0.9725 - val_loss: 0.0768 - val_acc: 0.9744
```

### pt
- Latest release: 2019-07-29
- Training data: 
  - CoNLL-U: [pt_gsd-ud-train.conllu](
    http://github.com/UniversalDependencies/UD_Portuguese-GSD)
  - UniMorph: [por](https://github.com/unimorph/por)
- Exceptions dict: 6,041 entries (from --no_losses)

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


SUFFIX IDENTIFICATION (val: 0.2 = 48,468 examples)
---------------------
Epoch 25/25
 - 216s - loss: 0.0657 - acc: 0.9816 - val_loss: 0.0539 - val_acc: 0.9843

INDEX IDENTIFICATION (val: 0.2 = 48,468 examples)
--------------------
Epoch 25/25
 - 209s - loss: 0.0693 - acc: 0.9792 - val_loss: 0.0494 - val_acc: 0.9844
```
