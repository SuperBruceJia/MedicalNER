# Medical Named Entity Recognition (MedicalNER)

## Abstract

With the development of Medical Artificial Intelligence (AI) System, Natural Language Processing (NLP) has played an essential role to process medical texts and build intelligent machines. Named Entity Recognition (NER), one of the most basic NLP tasks, is primarily studied since it is the cornerstone of the following NLP downstream tasks, e.g., Relation Extraction. In this work, a character-level Bidirectional Long-short Term Memory (BiLSTM)-based models were introduced to tackle the challenge of medical texts. The input character embedding vectors were randomly initialized and then updated during training. The character-level BiLSTM extracted features from medical order-matter sequential data. The followed Conditional Random Field (CRF) predicted the final entity tag. Results have shown that the presented method took advantage of the recurrent architecture and achieved competitive performances for medical texts. Promising results paved the road towards building robust and powerful medical AI engines. 

--------------------------------------------------------------------------------

## Topic and Study

**Task**: Named Entity Recognition (NER) implemented using PyTorch

**Background**: Medical & Clinical Healthcare 

**Level**: Character (and Word) Level

**Data Annotation**: BIOES tagging Scheme

**Method**:

1. CRF++

2. Character-level BiLSTM + CRF

3. Character-level BiLSTM + Word-level BiLSTM + CRF

4. Character-level BiLSTM + Word-level CNN + CRF

**Results**:

&emsp;Results of this work can be downloaded [here](https://github.com/SuperBruceJia/MedicalNER/raw/master/NER-Models-Results.xlsx).

**Prerequisities**:

&emsp;For Word-level models:

&emsp;The pre-trained word vectors can be downloaded [here](https://drive.google.com/file/d/1Rfe7QObJnaOUYK3cIQ6BXLyuJg-5516Y/view?usp=sharing).

```python
def load_word_vector(self):
    """
    Load word vectors
    """
    print("Start to load pre-trained word vectors!!")
    pre_trained = {}
    for i, line in enumerate(codecs.open(self.model_path + "word_vectors.vec", 'r', encoding='utf-8')):
        line = line.rstrip().split()
        if len(line) == self.word_dim + 1:
            pre_trained[line[0]] = np.array([float(x) for x in line[1:]]).astype(np.float32)
    return pre_trained
```

&emsp;For Character-level models:

&emsp;The Embeddings of characters are randomly initialized and updated by a PyTorch Function, i.e., (nn.Embedding).

```python
self.char_embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.char_dim)
```

--------------------------------------------------------------------------------

## Some Statistics Info

Number of entities: 34

| No.   | Entity    | Number | Recognize|
| :----:| :----:    | :----: |:----:    |
| 1     | E95f2a617 | 3221   | ✔︎       |
| 2     | E320ca3f6 | 6338   | ✔︎       |
| 3     | E340ca71c | 22209  | ✔︎       |
| 4     | E1ceb2bd7 | 3706   | ✔︎       |
| 5     | E1deb2d6a | 9744   | ✔︎       |
| 6     | E370cabd5 | 6196   | ✔︎       |
| 7     | E360caa42 | 5268   | ✔︎       |
| 8     | E310ca263 | 6948   | ✔︎       |
| 9     | E300ca0d0 | 9490   | ✔︎       |
| 10    | E18eb258b | 4526   | ✔︎       |
| 11    | E3c0cb3b4 | 6280   | ✔︎       |
| 12    | E1beb2a44 | 1663   | ✔︎       |
| 13    | E3d0cb547 | 1025   | ✔︎       |
| __14__    | __E14eb1f3f__ | __406__    | __⨉__       |
| 15    | E8ff29ca5 | 1676   | ✔︎       |
| 16    | E330ca589 | 1487   | ✔︎       |
| __17__    | __E89f29333__ | __1093__   | __⨉__       |
| __18__    | __E8ef29b12__ | __217__    | __⨉__       |
| 19    | E1eeb2efd | 1637   | ✔︎       |
| __20__    | __E1aeb28b1__ | __209__    | __⨉__       |
| 21    | E17eb23f8 | 670    | ✔︎       |
| 22    | E87f05176 | 407    | ✔︎       |
| 23    | E88f05309 | 355    | ✔︎       |
| __24__    | __E19eb271e__ | __152__    | __⨉__       |
| __25__    | __E8df2997f__ | __135__    | __⨉__       |
| 26    | E94f2a484 | 584    | ✔︎       |
| __27__    | __E13eb1dac__ | __58__     | __⨉__       |
| __28__    | __E85f04e50__ | __6__      | __⨉__       |
| __29__    | __E8bf057c2__ | __7__      | __⨉__       |
| __30__    | __E8cf297ec__ | __6__      | __⨉__       |
| __31__    | __E8ff05e0e__ | __6__      | __⨉︎__       |
| __32__    | __E87e38583__ | __18__     | __⨉︎__       |
| __33__    | __E86f04fe3__ | __6__      | __⨉︎__       |
| __34__    | __E8cf05955__ | __64__     | __⨉︎__       |

**train data**: 6494 &nbsp;
**vocab size**: 2258 &nbsp;
**unique tag**: 74 &nbsp;

**dev data**: 865 &nbsp;
**vocab size**: 2258 &nbsp;
**unique tag**: 74 &nbsp;

data: number of sentences
vocab: character vocabulary
unique tag: number of (prefix + entities)

--------------------------------------------------------------------------------

## Structure of the code

At the root of the project, you will see:

```python
├── data
|  └── train # Training set 
|  └── val # Validation set 
|  └── test # Testing set 
├── models
|  └── data.pkl # Containing all the used data, e.g., look-up table
|  └── params.pkl # Saved PyTorch model
├── preprocess-data.py # Preprocess the original dataset
├── data_manager.py # Load the train/val/test data
├── model.py # BiLSTM-CRF with Attention Model
├── main.py # Main codes for the training and prediction
├── utils.py # Some functions for prediction stage and evaluation criteria
├── config.yml # Contain the hyper-parameters settings
```

--------------------------------------------------------------------------------

## Basic  Model Architecture

```text
    Character Input
          |                         
     Lookup Layer  <----------------|    Update Character Embedding
          |                         |
     Bi-LSTM Model  <---------------|        Extract Features
          |                         |     Back-propagation Errors
     Linear Layer  <----------------|   Update Trainable Parameters
          |                         |
       CRF Model  <-----------------|    
          |                         |
Output corresponding tags  ---> [NLL Loss] <---  Target tags
```

--------------------------------------------------------------------------------

## Limitations

1. Currently only support CPU training
    
    GPU is much more slower than the CPU as a result of the viterbi decode's FOR LOOP. 

2. Cannot recognize entities with fewer examples (< 500 samples)

--------------------------------------------------------------------------------

## Final Results

**Overall F1 score on 18 entities:** 

Separate F1 score on each entity:

| No.   | Entity    | Number | F1 Score|
| :----:| :----:    | :----: |:----:   |
| 1     | E95f2a617 | 3221   | ✔︎       |
| 2     | E320ca3f6 | 6338   | ✔︎       |
| 3     | E340ca71c | 22209  | ✔︎       |
| 4     | E1ceb2bd7 | 3706   | ✔︎       |
| 5     | E1deb2d6a | 9744   | ✔︎       |
| 6     | E370cabd5 | 6196   | ✔︎       |
| 7     | E360caa42 | 5268   | ✔︎       |
| 8     | E310ca263 | 6948   | ✔︎       |
| 9     | E300ca0d0 | 9490   | ✔︎       |
| 10    | E18eb258b | 4526   | ✔︎       |
| 11    | E3c0cb3b4 | 6280   | ✔︎       |
| 12    | E1beb2a44 | 1663   | ✔︎       |
| 13    | E3d0cb547 | 1025   | ✔︎       |
| 14    | E8ff29ca5 | 1676   | ✔︎       |
| 15    | E330ca589 | 1487   | ✔︎       |
| 16    | E1eeb2efd | 1637   | ✔︎       |
| 17    | E17eb23f8 | __670__    | ✔︎       |
| 18    | E94f2a484 | __584__    | ✔︎       |

--------------------------------------------------------------------------------

## Hyperparameters settings

| Name           | Value  | 
| :----:         | :----: |
|embedding_size  | 30 / 40 / 50 / 100    |
|hidden_size     | 128 / 256   |
|batch_size      | 8/ 16 / 32 / 64     | 
|dropout rate    | 0.50 / 0.75  |
|learning rate   | 0.01 / 0.001   |
|epochs          | 100    |
|weight decay    | 0.0005 |
|max length      | 100 / 120   |

--------------------------------------------------------------------------------

## Model Deployment

The BiLSTM + CRF model has been deployed using **Docker** + **Flask** as a webapp.

***The codes and demos were open-sourced in this [repo](https://github.com/SuperBruceJia/pytorch-flask-deploy-webapp).***

### Screenshot of the model output: 

<p align="center">
  <a href="https://github.com/SuperBruceJia/pytorch-flask-deploy-webapp"> <img src="https://github.com/SuperBruceJia/pytorch-flask-deploy-webapp/raw/master/screenshot.png"></a> 
</p>
    
--------------------------------------------------------------------------------

## Reference

### Traditional Methods for NER: BiLSTM + CNN + CRF 

1. [Neural Architectures for Named Entity Recognition](https://github.com/SuperBruceJia/paper-reading/raw/master/NLP-field/Named-Entity-Recognition/Neural%20Architectures%20for%20Named%20Entity%20Recognition.pdf)

2. [Log-Linear Models, MEMMs, and CRFs](https://github.com/SuperBruceJia/paper-reading/raw/master/NLP-field/Named-Entity-Recognition/Log-Linear%20Models%2C%20MEMMs%2C%20and%20CRFs.pdf)

3. [Named Entity Recognition with Bidirectional LSTM-CNNs](https://github.com/SuperBruceJia/paper-reading/raw/master/NLP-field/Named-Entity-Recognition/Named%20Entity%20Recognition%20with%20Bidirectional%20LSTM-CNNs.pdf)

4. [A Survey on Deep Learning for Named Entity Recognition](https://github.com/SuperBruceJia/paper-reading/raw/master/NLP-field/Named-Entity-Recognition/A%20Survey%20on%20Deep%20Learning%20for%20Named%20Entity%20Recognition.pdf)

### SOTA Method for NER (I think)

1. [Lattice LSTM](https://github.com/SuperBruceJia/paper-reading/raw/master/NLP-field/Named-Entity-Recognition/%E8%AF%8D%E6%B1%87%E5%A2%9E%E5%BC%BANER/Lattice%20LSTM%20%20-%20Lattice%20LSTM%20for%20Chinese%20Sentence%20Representation.pdf)

2. [中文NER的正确打开方式: 词汇增强方法总结 (从Lattice LSTM到FLAT)](https://zhuanlan.zhihu.com/p/142615620)

3. [工业界如何解决NER问题](https://my.oschina.net/u/4308645/blog/4333232)

--------------------------------------------------------------------------------

## Licence

MIT Licence
