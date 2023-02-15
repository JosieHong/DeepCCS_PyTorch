import torch
from torch.utils.data import Dataset

import re
import numpy as np
import pandas as pd



class SmilesCCSDataset(Dataset): 
	def __init__(self, data_path, len_smiles, num_add): 
		# self.ENCODE_SMILES = {"Na": 0, "Li": 1, " ": 2, "#": 3, ")": 4, "(": 5, "+": 6, "-": 7, "/": 8, ".": 9, "1": 10, 
		# 						"3": 11, "2": 12, "5": 13, "4": 14, "7": 20, "6": 16, "=": 17, "@": 18, "C": 19, "Br": 15, 
		# 						"F": 21, "I": 22, "H": 23, "K": 24, "O": 25, "N": 26, "P": 27, "S": 28, "[": 29, "]": 30, 
		# 						"\\": 31, "Se": 32, "c": 33, "Cl": 34, "Ca": 35, "n": 36, 
		# 						'Pad': 37} 
		self.ENCODE_SMILES = {'(': 0, 'Br': 1, '31': 2, '54': 3, '56': 4, ')': 5, '52': 6, '2': 7, '32': 8, '15': 9, 'N': 10, 
								'I': 11, '6': 12, '8': 13, '67': 14, '7': 15, '25': 16, '34': 17, 'S': 18, '4': 19, '45': 20, 
								'3': 21, 'O': 22, '43': 23, '65': 24, '46': 25, '9': 26, '42': 27, '5': 28, '64': 29, '=': 30, 
								'53': 31, 'P': 32, 'Cl': 33, 'C': 34, 'H': 35, '21': 36, '13': 37, '24': 38, '12': 39, '14': 40, 
								'1': 41, 'F': 42, '23': 43, 'Pad': 44}
		self.ENCODE_ADD = {'M+H': 0, 'M-H': 1, 'M+Na': 2, 'M+H-H2O': 3, 'M+2H': 4}

		df = pd.read_csv(data_path)
		self.id_list = []
		self.smiles_array = []
		self.adduct_array = []
		self.ccs_array = []
		for i, row in df.iterrows():
			smiles = self.splitting(row['SMILES'])
			if len(smiles) > len_smiles:
				continue
			not_env_flag = False
			for s in smiles: # check if can be encoded
				if s not in self.ENCODE_SMILES.keys():
					not_env_flag = True
					break
			if not_env_flag:
				continue
			smiles += ['Pad'] * (len_smiles - len(smiles)) # pad with 'Pad'
			smiles = np.array([self.ENCODE_SMILES[s] for s in smiles])

			add = self.clean_add(row['Adduct'])
			if add not in self.ENCODE_ADD.keys():
				continue
			add = np.array(self.ENCODE_ADD[add])

			ccs = np.array(row['CCS'])
			ccs_id = row['ID']

			self.smiles_array.append(smiles)
			self.adduct_array.append(add)
			self.ccs_array.append(ccs)
			self.id_list.append(ccs_id)
			

	def splitting(self, smiles):
		p = re.compile(r'[A-Z][a-z]+|[A-Z]|\d+|\s+|\=|\(|\)')
		splitted_smiles = re.findall(p, smiles)
		# print(smiles, splitted_smiles)
		return splitted_smiles

	def clean_add(self, add):
		p = re.compile(r'[\[](.*?)[\]]', re.S)
		adds = re.findall(p, add)
		if len(adds) > 0: 
			return adds[0]
		else:
			return add

	def num_sym(self,):
		return len(self.ENCODE_SMILES)

	def __getitem__(self, idx): 
		return self.id_list[idx], self.smiles_array[idx], self.adduct_array[idx], self.ccs_array[idx]

	def __len__(self): 
		return len(self.id_list)