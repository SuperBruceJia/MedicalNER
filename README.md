# Medical Named Entity Recognition (MedicalNER)

## Topic and Study

**Task**: Named Entity Recognition (NER)

**Background**: Medical & Clinical Healthcare 

**Level**: Character Level

**Method**:

1. CRF++

2. Character-level BiLSTM + CRF

3. Character-level BiLSTM + Word-level BiLSTM + CRF

4. Character-level BiLSTM + Word-level CNN + CRF

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

## Model Architecture

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
2. Cannot recognize entities with fewer examples (< 500 samples)

--------------------------------------------------------------------------------

## Questions

1. Some F1 Scores were always 0
2. Some entities' F1 scores are unstable, i.e., always fluctuate.
2. The loss of the model always ups and downs even with (very) small learning rate
3. Cannot recognize entities whose **number of samples are less than 500** --> data imbalance?

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
