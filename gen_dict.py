import os
import argparse

import re
import pandas as pd
import json

'''PERIODIC TABLE OF ELEMENTS
H                                                  He
Li Be                               B  C  N  O  F  Ne
Na Mg                               Al Si P  S  Cl Ar
K  Ca Sc Ti V  Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr
Rb Sr Y  Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I  Xe
Cs Ba La Hf Ta W  Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn
Fr Ra Ac Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og
'''

def splitting(smiles):
	p = re.compile(r'He|Li|Be|Ne|Na|Mg|Al|Si|Cl|Ar|Ca|Sc|Ti|Cr|Mn|Fe|Co|Ni|Cu|Zn|Ga|Ge|As|Se|Br\
					|Kr|Rb|Sr|Zr|Nb|Mo|Tc|Ru|Rh|Pd|Ag|Cd|In|Sn|Sb|Te|Xe|Cs|Ba|La|Hf|Ta|Re|Os|Ir\
					|Pt|Au|Hg|Tl|Pb|Bi|Po|At|Rn|Fr|Ra|Ac|Rf|Db|Sg|Bh|Hs|Mt|Ds|Rg|Cn|Nh|Fl|Mc|Lv\
					|Ts|Og|[A-Za-z]|\d+|\+|\-|\=|\(|\)|\[|\]|\.|\@|\\|\/|\%|\#')
	splitted_smiles = re.findall(p, smiles)
	re_smiles = ''.join(splitted_smiles)
	assert re_smiles == smiles, "something is missing {} -> {}".format(smiles, re_smiles)
	# print(smiles, splitted_smiles)
	return splitted_smiles



if __name__ == "__main__":
	# Training settings
	parser = argparse.ArgumentParser(description='Mass Spectra to formula (train)')
	parser.add_argument('--data_dir', type=str, required=True,
						help='Folder of all the data')
	parser.add_argument('--output', type=str, default = '',
						help='Path to output the dictionary')
	args = parser.parse_args()

	file_list = os.listdir(args.data_dir)
	symbols = set()
	for file_name in file_list:
		print(file_name)
		df = pd.read_csv(os.path.join(args.data_dir, file_name))
		for i, row in df.iterrows():
			smiles = splitting(row['SMILES'])
			if len(smiles) > 250:
				continue
			symbols.update(smiles)

	symbols = list(symbols)
	symbol_dict = {k: i for i, k in enumerate(symbols)}
	symbol_dict['Pad'] = len(symbols)

	# Serializing json
	json_object = json.dumps(symbol_dict, indent=4)
	
	# Writing to sample.json
	with open(args.output, "w") as outfile:
		outfile.write(json_object)
	print('Done!')

