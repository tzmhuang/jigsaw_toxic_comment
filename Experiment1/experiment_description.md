- [Experiments](#experiments)
  - [Purpose:](#purpose)
  - [Test1: Base Model](#test1-base-model)
  - [Test 2: Data Cleaning (1)](#test-2-data-cleaning-1)
  - [Test 3: Data Cleaning (2)](#test-3-data-cleaning-2)
  - [Test 4: Model Structure (1)](#test-4-model-structure-1)
  - [Test 5: Model Structure (2)](#test-5-model-structure-2)
  - [Test 6: Label Smoothing](#test-6-label-smoothing)
  - [Test 7: Clean(3) + Model(2) + LS](#test-7-clean3--model2--ls)
  - [Result](#result)
    - [*Bench vs Clean(1) vs Clean(2)*](#bench-vs-clean1-vs-clean2)
    - [*Bench vs Model(1) vs Model(2)*](#bench-vs-model1-vs-model2)
    - [*Bench vs Label Smoothing*](#bench-vs-label-smoothing)

# Experiments

  Experiments designed for testing performance of models fitted for Jigsaw Multilingual Toxic Comment Competition on Kaggle.

  Experiments uses distiled RoBERTa model mode toxic comment classification. Models were pretrained before further fine-tuned locally. Models are trained using TF-2.0, CUDA 10.2, on NVIDIA 1660 Super.
  

## Purpose: 
- Run tests by scripts
- Record test results and compare model performance in predicting toxic comments
- Compare between :
  - base model performance
  - cleaning data / not cleaning data
  - methods of cleaning data
  - model structure
  - label smoothing
  - pesudo labelling
  - translated data fitting
## Test1: Base Model 
|                     |                                     |
| ------------------- | ----------------------------------- |
| __Date__            | 28-05-2020                          |
| __Model__           | distilbert-base-multilingual-cased  |
| __Model structure__ | xlmroberta_dense                    |
| __Data cleaning__   | No                                  |
| __Encoding__        | len=128 \ saved                     |
| __Batch size__      | 64                                  |
| __Epochs trained__  | 8                                   |
| __Optimizer__       | Adam (lr=5e-6)                      |
| __Loss__            | BinaryCrossEntropy                  |
| __Training data__   | wiki data: 60000 toxic + non-toxic  |
| __Validation data__ | original validation & validation_en |
|                     |                                     |
## Test 2: Data Cleaning (1) 
|                     |                                     |
| ------------------- | ----------------------------------- |
| __Date__            | 28-05-2020                          |
| __Model__           | distilbert-base-multilingual-cased  |
| __Model structure__ | xlmroberta_dense                    |
| __Data cleaning*__  | Yes \ preserve sentiment features   |
| __Encoding__        | len=128 \ saved                     |
| __Batch size__      | 64                                  |
| __Epochs trained__  | 8                                   |
| __Optimizer__       | Adam (lr=5e-6)                      |
| __Loss__            | BinaryCrossEntropy                  |
| __Training data__   | wiki data: 60000 toxic + non-toxic  |
| __Validation data__ | original validation & validation_en |
|                     |                                     |
## Test 3: Data Cleaning (2)
|                     |                                     |
| ------------------- | ----------------------------------- |
| __Date__            | 28-05-2020                          |
| __Model__           | distilbert-base-multilingual-cased  |
| __Model structure__ | xlmroberta_dense                    |
| __Data cleaning*__  | Yes \ exclude sentiment features    |
| __Encoding__        | len=128 \ saved                     |
| __Batch size__      | 64                                  |
| __Epochs trained__  | 8                                   |
| __Optimizer__       | Adam (lr=5e-6)                      |
| __Loss__            | BinaryCrossEntropy                  |
| __Training data__   | wiki data: 60000 toxic + non-toxic  |
| __Validation data__ | original validation & validation_en |
|                     |                                     |
## Test 4: Model Structure (1)
|                      |                                     |
| -------------------- | ----------------------------------- |
| __Date__             | 28-05-2020                          |
| __Model__            | distilbert-base-multilingual-cased  |
| __Model structure*__ | xlmroberta_mean_max_pooling         |
| __Data cleaning__    | No                                  |
| __Encoding__         | len=128 \ loaded                    |
| __Batch size__       | 64                                  |
| __Epochs trained__   | 8                                   |
| __Optimizer__        | Adam (lr=5e-6)                      |
| __Loss__             | BinaryCrossEntropy                  |
| __Training data__    | wiki data: 60000 toxic + non-toxic  |
| __Validation data__  | original validation & validation_en |
|                      |                                     |
## Test 5: Model Structure (2)
|                      |                                     |
| -------------------- | ----------------------------------- |
| __Date__             | 28-05-2020                          |
| __Model__            | distilbert-base-multilingual-cased  |
| __Model structure*__ | xlmroberta_mean_max_min_pooling     |
| __Data cleaning__    | No                                  |
| __Encoding__         | len=128 \ loaded                    |
| __Batch size__       | 64                                  |
| __Epochs trained__   | 8                                   |
| __Optimizer__        | Adam (lr=5e-6)                      |
| __Loss__             | BinaryCrossEntropy                  |
| __Training data__    | wiki data: 60000 toxic + non-toxic  |
| __Validation data__  | original validation & validation_en |
|                      |                                     |

## Test 6: Label Smoothing
|                     |                                     |
| ------------------- | ----------------------------------- |
| __Date__            | 28-05-2020                          |
| __Model__           | distilbert-base-multilingual-cased  |
| __Model structure__ | xlmroberta_dense                    |
| __Data cleaning__   | No                                  |
| __Encoding__        | len=128 \ loaded                    |
| __Batch size__      | 64                                  |
| __Epochs trained__  | 8                                   |
| __Optimizer__       | Adam (lr=5e-6)                      |
| __Loss*__           | BinaryCrossEntropy (LS=0.1)         |
| __Training data__   | wiki data: 60000 toxic + non-toxic  |
| __Validation data__ | original validation & validation_en |
|                     |                                     |

## Test 7: Clean(3) + Model(2) + LS
|                      |                                     |
| -------------------- | ----------------------------------- |
| __Date__             | 28-05-2020                          |
| __Model__            | distilbert-base-multilingual-cased  |
| __Model structure*__ | xlmroberta_mean_max_min_pooling     |
| __Data cleaning*__   | Yes \ updated clean_text3 func      |
| __Encoding__         | len=128 \ loaded                    |
| __Batch size__       | 64                                  |
| __Epochs trained__   | 8                                   |
| __Optimizer__        | Adam (lr=5e-6)                      |
| __Loss*__            | BinaryCrossEntropy (LS=0.1)         |
| __Training data__    | wiki data: 60000 toxic + non-toxic  |
| __Validation data__  | original validation & validation_en |
|                      |                                     |


## Result

(For details, see Tensorboard with logdir='./logs/fit')

We focus on comparing AUC and loss, as AUC is a better metric than binary accuracy for measuring prediction accuracy, and loss for understanding how well the model fit data as well as shed light on training process.

### *Bench vs Clean(1) vs Clean(2)*

In terms of training process, during the first epoch (2k steps). Clean(1) has the lowest training loss and highest AUC. Bench performed worse than Clean(1) with slightly higher loss and lower AUC. While Clean(2) is the worst among the three. During subsequent epochs, Clean(1) and Bench have similar performence, while Clean(2) is noticeably worse than the other two.

In terms of validation, Clean(2) performs noticeably worse on validation dataset than Clean(1) and Clean(2) with higher loss and lower AUC at all epochs. The difference is less noticeable on validation_en dataset. On epoch 2&3, Clean(2) has comparable performance with the other two, however it performes worse in other epochs. Clean(1) and Bench have similar validation performace on both validation datasets in general but Clean(1) performs slightly better with lower loss increase rate.

Conclusion: cleaning is better than not cleaning the text data. However, we should be aware of not over cleaning the data. Going forward, we would base cleaning method on Clean(1) which tends to perserves sentiment information.

### *Bench vs Model(1) vs Model(2)*

All three models showed similar training loss and training AUC. Bench has a slightly lower training loss and a slightly better AUC but the difference is almost neglectable. Model(1) and Model(2) showed very similar performance.

During validation, Bench generally has higher loss and AUC on both datasets validation and validation_en, signalling a poorer performance. Model(1) and Model(2) have similar validation performance interms of loss and AUC. Model(2) has a more consistent performance and is usually better than Model(1). However, Model(1) had a spike in performance during epoch 6.

Conclusion: Model(2) performs better than Bench and Model(2). Which is expected due to richer information. We would continue based on Model(2).

### *Bench vs Label Smoothing*

Label Smoothing(LS) has lower AUC and significantly higher loss compared to Bench during training. This observation is consistent with previous tests. 

During validation, LS has similar AUC performance if not slightly poor compared to Bench on validation dataset. LS demonstrated a higher loss but a significantly slower loss increase rate on the same dataset. On dataset validation_en, LS showed a noticeable higher AUC than Bench. However, LS also showed a higher loss than Bench. The rate of increase of loss of LS is low during epoch 0 to 4, however the rate increased afterwards.

Conclusion: The test shows promise on LS's ability to regualarize and generalize the model, but the result is not conclusive. But, according to tests donw by the kaggle community, LS should deliver a better prediciton result.