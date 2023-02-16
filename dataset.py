import torch
from torch.utils.data import Dataset

import re
import numpy as np
import pandas as pd
import json



class SmilesCCSDataset(Dataset): 
	def __init__(self, data_path, len_smiles, num_add, smiles_dict_path='./encode_smiles.json'): 
		# original dictionary for SMILES encoding 
		# self.ENCODE_SMILES = {"Na": 0, "Li": 1, " ": 2, "#": 3, ")": 4, "(": 5, "+": 6, "-": 7, "/": 8, ".": 9, "1": 10, 
		# 						"3": 11, "2": 12, "5": 13, "4": 14, "7": 20, "6": 16, "=": 17, "@": 18, "C": 19, "Br": 15, 
		# 						"F": 21, "I": 22, "H": 23, "K": 24, "O": 25, "N": 26, "P": 27, "S": 28, "[": 29, "]": 30, 
		# 						"\\": 31, "Se": 32, "c": 33, "Cl": 34, "Ca": 35, "n": 36, 
		# 						'Pad': 37} 
		with open(smiles_dict_path, "r") as f: 
			self.ENCODE_SMILES = json.load(f)
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
		p = re.compile(r'He|Li|Be|Ne|Na|Mg|Al|Si|Cl|Ar|Ca|Sc|Ti|Cr|Mn|Fe|Co|Ni|Cu|Zn|Ga|Ge|As|Se|Br\
					|Kr|Rb|Sr|Zr|Nb|Mo|Tc|Ru|Rh|Pd|Ag|Cd|In|Sn|Sb|Te|Xe|Cs|Ba|La|Hf|Ta|Re|Os|Ir\
					|Pt|Au|Hg|Tl|Pb|Bi|Po|At|Rn|Fr|Ra|Ac|Rf|Db|Sg|Bh|Hs|Mt|Ds|Rg|Cn|Nh|Fl|Mc|Lv\
					|Ts|Og|[A-Za-z]|\d+|\+|\-|\=|\(|\)|\[|\]|\.|\@|\\|\/|\%|\#')
		splitted_smiles = re.findall(p, smiles)
		re_smiles = ''.join(splitted_smiles)
		assert re_smiles == smiles, "something is missing {} -> {}".format(smiles, re_smiles)
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