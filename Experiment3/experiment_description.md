- [Experiment 2](#experiment-2)
  - [Purpose](#purpose)
  - [Test 1: Bench [Clean(3) + Model(2) + LS]](#test-1-bench-clean3--model2--ls)
  - [Test 2: Translated training data (RU, FR, PT)](#test-2-translated-training-data-ru-fr-pt)
  - [Test 3: Translated training data (EN, RU, FR, PT)](#test-3-translated-training-data-en-ru-fr-pt)
  - [Test 4: Translated training data (TR,IT,ES)](#test-4-translated-training-data-trites)
  - [Test 5: Translated training data (EM,TR,IT,ES)](#test-5-translated-training-data-emtrites)
  - [Test 6: Translated training data (TR,IT,ES,RU,FR,PT)](#test-6-translated-training-data-tritesrufrpt)
  - [Test 7: Translated training data (EN,TR,IT,ES,RU,FR,PT)](#test-7-translated-training-data-entritesrufrpt)
  - [Test 8: Translated training data non-repeat (RU, FR, PT)](#test-8-translated-training-data-non-repeat-ru-fr-pt)
  - [Test 9: Translated training data non-repeat (TR,IT,ES)](#test-9-translated-training-data-non-repeat-trites)
  - [Test 10: Translated training data non-repeat (TR,IT,ES,RU,FR,PT)](#test-10-translated-training-data-non-repeat-tritesrufrpt)
  - [Result](#result)

# Experiment 3

  Experiments designed for testing performance of models fitted for Jigsaw Multilingual Toxic Comment Competition on Kaggle.

  Experiments uses distiled RoBERTa model mode toxic comment classification. Models were pretrained before further fine-tuned locally. Models are trained using TF-2.0, CUDA 10.2, on NVIDIA 1660 Super.
  

## Purpose
- Run tests by scripts
- Record test results and compare model performance in predicting toxic comments
- Investigate: 
  - The effect of label smoothing alpha on training results


## Test 1: Bench (LS = 0.0)
|                      |                                                               |
| -------------------- | ------------------------------------------------------------- |
| __Date__             | 28-05-2020                                                    |
| __Model__            | distilbert-base-multilingual-cased                            |
| __Model structure*__ | xlmroberta_mean_max_min_pooling                               |
| __Data cleaning*__   | Yes \ updated clean_text3 func                                |
| __Encoding__         | len=128 \ loaded                                              |
| __Batch size__       | 64                                                            |
| __Epochs trained__   | 8                                                             |
| __Optimizer__        | Adam (lr=5e-6)                                                |
| __Loss*__            | BinaryCrossEntropy (LS=0.0)                                   |
| __Training data__    | translated wiki data(tr,it,es,ru,fr,pt): 25% toxic non-repeat |
| __Validation data__  | original validation & validation_en                           |

## Test 2: Bench (LS = 0.03)
|                      |                                                               |
| -------------------- | ------------------------------------------------------------- |
| __Date__             | 28-05-2020                                                    |
| __Model__            | distilbert-base-multilingual-cased                            |
| __Model structure*__ | xlmroberta_mean_max_min_pooling                               |
| __Data cleaning*__   | Yes \ updated clean_text3 func                                |
| __Encoding__         | len=128 \ loaded                                              |
| __Batch size__       | 64                                                            |
| __Epochs trained__   | 8                                                             |
| __Optimizer__        | Adam (lr=5e-6)                                                |
| __Loss*__            | BinaryCrossEntropy (LS=0.03)                                   |
| __Training data__    | translated wiki data(tr,it,es,ru,fr,pt): 25% toxic non-repeat |
| __Validation data__  | original validation & validation_en                           |

## Test 3: Bench (LS = 0.06)
|                      |                                                               |
| -------------------- | ------------------------------------------------------------- |
| __Date__             | 28-05-2020                                                    |
| __Model__            | distilbert-base-multilingual-cased                            |
| __Model structure*__ | xlmroberta_mean_max_min_pooling                               |
| __Data cleaning*__   | Yes \ updated clean_text3 func                                |
| __Encoding__         | len=128 \ loaded                                              |
| __Batch size__       | 64                                                            |
| __Epochs trained__   | 8                                                             |
| __Optimizer__        | Adam (lr=5e-6)                                                |
| __Loss*__            | BinaryCrossEntropy (LS=0.06)                                   |
| __Training data__    | translated wiki data(tr,it,es,ru,fr,pt): 25% toxic non-repeat |
| __Validation data__  | original validation & validation_en                           |

## Test 4: Bench (LS = 0.1)
|                      |                                                               |
| -------------------- | ------------------------------------------------------------- |
| __Date__             | 28-05-2020                                                    |
| __Model__            | distilbert-base-multilingual-cased                            |
| __Model structure*__ | xlmroberta_mean_max_min_pooling                               |
| __Data cleaning*__   | Yes \ updated clean_text3 func                                |
| __Encoding__         | len=128 \ loaded                                              |
| __Batch size__       | 64                                                            |
| __Epochs trained__   | 8                                                             |
| __Optimizer__        | Adam (lr=5e-6)                                                |
| __Loss*__            | BinaryCrossEntropy (LS=0.1)                                   |
| __Training data__    | translated wiki data(tr,it,es,ru,fr,pt): 25% toxic non-repeat |
| __Validation data__  | original validation & validation_en                           |

## Test 5: Bench (LS = 0.13)
|                      |                                                               |
| -------------------- | ------------------------------------------------------------- |
| __Date__             | 28-05-2020                                                    |
| __Model__            | distilbert-base-multilingual-cased                            |
| __Model structure*__ | xlmroberta_mean_max_min_pooling                               |
| __Data cleaning*__   | Yes \ updated clean_text3 func                                |
| __Encoding__         | len=128 \ loaded                                              |
| __Batch size__       | 64                                                            |
| __Epochs trained__   | 8                                                             |
| __Optimizer__        | Adam (lr=5e-6)                                                |
| __Loss*__            | BinaryCrossEntropy (LS=0.13)                                   |
| __Training data__    | translated wiki data(tr,it,es,ru,fr,pt): 25% toxic non-repeat |
| __Validation data__  | original validation & validation_en                           |

## Test 6: Bench (LS = 0.16)
|                      |                                                               |
| -------------------- | ------------------------------------------------------------- |
| __Date__             | 28-05-2020                                                    |
| __Model__            | distilbert-base-multilingual-cased                            |
| __Model structure*__ | xlmroberta_mean_max_min_pooling                               |
| __Data cleaning*__   | Yes \ updated clean_text3 func                                |
| __Encoding__         | len=128 \ loaded                                              |
| __Batch size__       | 64                                                            |
| __Epochs trained__   | 8                                                             |
| __Optimizer__        | Adam (lr=5e-6)                                                |
| __Loss*__            | BinaryCrossEntropy (LS=0.16)                                   |
| __Training data__    | translated wiki data(tr,it,es,ru,fr,pt): 25% toxic non-repeat |
| __Validation data__  | original validation & validation_en                           |

## Test 7: Bench (LS = 0.2)
|                      |                                                               |
| -------------------- | ------------------------------------------------------------- |
| __Date__             | 28-05-2020                                                    |
| __Model__            | distilbert-base-multilingual-cased                            |
| __Model structure*__ | xlmroberta_mean_max_min_pooling                               |
| __Data cleaning*__   | Yes \ updated clean_text3 func                                |
| __Encoding__         | len=128 \ loaded                                              |
| __Batch size__       | 64                                                            |
| __Epochs trained__   | 8                                                             |
| __Optimizer__        | Adam (lr=5e-6)                                                |
| __Loss*__            | BinaryCrossEntropy (LS=0.2)                                   |
| __Training data__    | translated wiki data(tr,it,es,ru,fr,pt): 25% toxic non-repeat |
| __Validation data__  | original validation & validation_en                           |
## Result

(For details, see Tensorboard with logdir='./logs/fit')

We focus on comparing AUC and loss, as AUC is a better metric than binary accuracy for measuring prediction accuracy, and loss for understanding how well the model fit data as well as shed light on training process.
