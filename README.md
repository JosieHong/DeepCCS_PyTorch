# DeepCCS (PyTorch)

This is the PyTorch implementation of DeepCCS. The official implementation with TensorFlow is [[here]](https://github.com/plpla/DeepCCS). 

Updates compared with official implementation:

- Fix the SMILES string splitting methods. The numbers larger than 9 can be split correctly. 

- Generate the SMILES encoding dictionary automatically according to the training and test data. 



## Set up

```bash
conda create -n deepccs python=3.7
conda activate deepccs

# set the CUDA version
export PATH=/usr/local/cuda-11.7/bin:$PATH
# export LD_LIB_PATH=/usr/local/cuda-11.7/lib64

# install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

pip install tqdm pandas
```



## Train

Please convert the .h5 dataset into .csv dataset and generate the SMILES encoding dictionary first:

```bash
# DATASETS.h5 is copied from the official implementation of DeepCCS. We chose 'MetCCS_test_neg' 
# and 'MetCCS_test_pos' as the test datasets, the others are training datasets. 
python convet_h5.py --h5_path ./DATASETS.h5 --out_dir ./data/

python gen_dict.py --data_dir ./data/ --output encode_smiles.json 
```

Then the model can be trained by: 

```bash
python train.py --train_data ./data/deepccs_train.csv \
                --test_data ./data/deepccs_test.csv \
                --checkpoint_path ./check_point/deepccs_ours.pt \
                --result_path ./result/deepccs_origin.csv 

# custom dataset
python train.py --train_data ./data/ccs_train.csv \
                --test_data ./data/ccs_test.csv \
                --checkpoint_path ./check_point/deepccs_ours.pt \
                --result_path ./result/deepccs_ours.csv 
```



## Results

|                                    | R2                 | Mean absolute error |
|------------------------------------|--------------------|---------------------|
| Original datasets from DeepCCS     | 0.9759022106902219 | 3.2407103581215018  |
| Custom datasets (AllCCS & BushCCS) | 0.8515206514437069 | 9.204849729155162   |



## Reference

This implementation refers to the paper of DeepCCS and their TensorFlow implementation: 

- Plante, Pier-Luc, et al. "Predicting ion mobility collision cross-sections using a deep neural network: DeepCCS." Analytical chemistry 91.8 (2019): 5191-5199.

- https://github.com/plpla/DeepCCS
