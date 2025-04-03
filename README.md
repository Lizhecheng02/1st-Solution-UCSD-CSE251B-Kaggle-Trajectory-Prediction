## This GitHub Repo is the Solution for [UCSD-CSE251B-Kaggle-Trajectory-Prediction](https://www.kaggle.com/competitions/cse-251-b-2025) (Spring 2025)

#### Author: [Zhecheng Li](https://github.com/Lizhecheng02) && Professor: [Rose Yu](https://roseyu.com/)

### Python Environment

#### 1. Install Packages

```b
pip install -r requirements.txt
```

### Prepare Data

#### 1. Set Kaggle Api

```bash
export KAGGLE_USERNAME="your_kaggle_username"
export KAGGLE_KEY="your_api_key"
```

#### 2. Install unzip

```bash
sudo apt install unzip
```

#### 3. Download Datasets
```bash
cd data
kaggle competitions download -c cse-251-b-2025
unzip cse-251-b-2025.zip
```