# DeepCCS (PyTorch)

This is the PyTorch implementation of DeepCCS. The official implementation with TensorFlow is [here](https://github.com/plpla/DeepCCS). 

Updates compared with official implementation:

- Fix the SMILES string splitting methods. The numbers larger than 9 can be split correctly. 

- Enlarge the SMILES encoding dictionary into:
    ```bash
    {'(': 0, 'Br': 1, '31': 2, '54': 3, '56': 4, ')': 5, '52': 6, '2': 7, '32': 8, '15': 9, 'N': 10, 
    'I': 11, '6': 12, '8': 13, '67': 14, '7': 15, '25': 16, '34': 17, 'S': 18, '4': 19, '45': 20, 
    '3': 21, 'O': 22, '43': 23, '65': 24, '46': 25, '9': 26, '42': 27, '5': 28, '64': 29, '=': 30, 
    '53': 31, 'P': 32, 'Cl': 33, 'C': 34, 'H': 35, '21': 36, '13': 37, '24': 38, '12': 39, '14': 40, 
    '1': 41, 'F': 42, '23': 43, 'Pad': 44}
    ```



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