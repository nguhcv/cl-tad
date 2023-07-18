# CL-TAD (Contrastive Learning based Model for Time series Anomaly Detection)

Pytorch implementation of CL-TAD Model for Time series anomaly detection

## Abstract

Anomaly detection has gained significant attention in recent years, but detecting anomalies in the time-series domain remains challenging due to temporal dynamics, label scarcity, and data diversity in real-world applications. To address these challenges, we introduce a novel method for anomaly detection in time-series data, called CL-TAD (Contrastive Learning-based model for Times series Anomaly Detection), which employs contrastive learning technique. Inspired by the successes of reconstruction learning approaches and contrastive learning approaches, the proposed method seeks to leverage these approaches for time series anomaly detection. The CL-TAD method is comprised of two main components: positive sample generation and contrastive learning. The former component generates positive samples by trying to reconstruct the original data from masked samples. These positive samples, in conjunction with the original data, serve as input for the contrastive learning component, which works to detect anomalies. Experimental results on benchmark datasets have demonstrated that the CL-TAD method achieves state-of-the-art performance on five datasets out of nine benchmark datasets. In addition, the method performs well even when training data is limited. By applying an online testing strategy, the method is effective for real-time applications requiring swift decision-making. By leveraging reconstruction learning and contrastive learning techniques, the CL-TAD method  offers a promising solution for effectively detecting anomalies in time series data, by handling the issues raised by label scarcity and the data diversity, delivering high performance.

## SF_TAD framework architecture

<img src="/image/overall_framework.png" width ="600" align="center" >  

## Requirements
The recommended requirements for SF_TAD are specified as follows:
- torch ==1.9.0
- torchvision==0.10+cu111
- torchaudio==0.9.0
- numpy==1.19.2
- pandas==1.0.1
- matplotlib==3.3.1

The dependencies can be installed by:

<div style="background-color: rgb(50, 50, 50);">

`` pip install -r requirements.txt
``

</div>

## Dataset
Our framework is evaluated on 5 datasets:
- Univariate datasets:
  - Power demand: This data set records the power demand in a year at a Dutch research facility 
  - UCR : An anomaly detection benchmark is used in KDD-21 competition   
- Multivariate datasets:
  - ECG: This data set consists of 6 time series from electrocardiograms readings
  - 2D-Gestures: This dataset records X Y cordinate of hand gesture in a video
  - PSM: This data set collect internally from multiple application server nodes at eBay

## Usage

The following command illustrates how to train and test the ECG dataset with our framework:

Train ECG-A sub-dataset:
<div style="background-color: rgb(50, 50, 50);">

`` main.py -mode train -dataset ecg -ts_num 0 -dataset_dim 2 ....
``  
</div>

Test ECG-A sub-dataset:
<div style="background-color: rgb(50, 50, 50);">

`` main.py -mode test -dataset ecg -ts_num 0 -dataset_dim 2 ....
``  
</div>

Detailed descriptions of arguments can be get by 

<div style="background-color: rgb(50, 50, 50);">

`` main.py -h
``

</div>

## Guideline how to use our package on Window

1. Use git bash to clone our project

<img src="/image/git_clone.PNG" width ="400" >

2. Using command line create the python environment:  

<img src="/image/py_venv.PNG" width ="400" >
  
3. Upgrade pip, install requirements and active py venv

<img src="/image/pip_update.PNG" width ="400" >
<img src="/image/install_requirements.PNG" width ="400" >
<img src="/image/active_venv.PNG" width ="400" >

4. Comeback to project directory, run main.py for training or testing available trained models
  
  - Training ECG-0
<img src="/image/train_example.PNG" width ="400" >

  - Testing ECG-0 (our trained model is stored in saved_modules directories)

<img src="/image/test_example.PNG" width ="400" > 

