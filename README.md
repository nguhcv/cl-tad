# sf_tad
## How to train a model with SF_TAD:
1. Run **main/main_train.py**
2. Since we supported codes to train model, run one of them by seleting the dataset setting in main. Following image shows a example how to train ecg-dataset. (ts_num indicates ordinal number of the sub-dataset)
<p align="center">
  <img src="image/train_example.PNG" width = "650" title="training dataset ECG-A">
</p>

## How to test a model with SF_TAD:
1. Run **main/main_train.py**
2. Since we supported codes to test trained models, run one of them by seleting the dataset setting in main. Following image shows a example how to test ecg-dataset. (ts_num indicates ordinal number of the sub-dataset, saved_path indicates the directory that store the trained model). Our trained models is stored in **saved_modules** 
<p align="center">
  <img src="image/test_example.PNG" width = "650" title="training dataset ECG-A">
</p>
