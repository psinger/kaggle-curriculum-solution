## Kaggle - Learning Equality - Curriculum Recommendations: 4th place solution code

### Data preparation

Run the following commands to create a folder `data` and to download and extract the competition data:
```
mkdir data
cd data
kaggle competitions download -c learning-equality-curriculum-recommendations
unzip learning-equality-curriculum-recommendations.zip
```

To prepare the training dataframes, run the jupyter notebook `prep_data_final_v1.ipynb`, `prep_data_final_v2.ipynb` and `prep_data_final_v3.ipynb`.

### Training

Running training is very straight forward, just run the following command:

`python train.py -C yaml_accuracy/cfg_name.yaml`

To train all models sequentially, run:

`for cfg in yaml_accuracy/*; do python train.py -C $cfg; done`

### Inference & Evaluation

For inference and validation, please refer to the [inference kernel on Kaggle](https://www.kaggle.com/code/ilu000/curriculum-4th-place-solution).

Simply replace the corresponding checkpoints. 
All configs in this training code match the ones of the inference kernel.
You can just run the inference kernel as-is without re-training, all datasets are shared.

### Efficiency solution

To train all models sequentially, run:

`for cfg in yaml_efficiency/*; do python train.py -C $cfg; done`

[inference kernel on Kaggle](https://www.kaggle.com/code/philippsinger/efficiency-blend-v1)
