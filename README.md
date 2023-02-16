# DeepCCS (PyTorch)

This is the PyTorch implementation of DeepCCS. The official implementation with TensorFlow is [here](https://github.com/plpla/DeepCCS). 

Updates compared with official implementation:

- Fix the SMILES string splitting methods. The numbers larger than 9 can be split correctly. 

- Generate the SMILES encoding dictionary automatically according to the data. 



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

Please generate the SMILES encoding dictionary first:

```bash
python gen_dict.py --data_dir ./data/ --output encode_smiles.json 
```

```bash
python train.py --train_data ./data/ccs_train.csv \
                --test_data ./data/ccs_test.csv \
                --checkpoint_path ./check_point/deepccs_ours.pt \
                --result_path ./result/deepccs_ours.csv 
```

## Results

Update later...

## Reference

This implementation refers to the paper of DeepCCS: 

```
Plante, Pier-Luc, et al. "Predicting ion mobility collision cross-sections using a deep neural network: DeepCCS." Analytical chemistry 91.8 (2019): 5191-5199.
```