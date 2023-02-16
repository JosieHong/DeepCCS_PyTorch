import os
import argparse

import pandas as pd
import h5py as h5



if __name__ == "__main__":
	# Training settings
	parser = argparse.ArgumentParser(description='Mass Spectra to formula (train)')
	parser.add_argument('--h5_path', type=str, required=True,
						help='Path to h5 data')
	parser.add_argument('--out_dir', type=str, default = '',
						help='Dictionary to output')
	args = parser.parse_args()

	df_train_list = []
	df_test_list = []
	# Open reference file and retrieve data corresponding to the dataset name 
	with h5.File(args.h5_path, 'r') as f: 
		print(list(f.keys()))
		for dataset_name in ['Astarita_neg', 'Astarita_pos', 'Baker', 'CBM', 'McLean', 'MetCCS_test_neg', 'MetCCS_test_pos', 'MetCCS_train_neg', 'MetCCS_train_pos']: 
			df = pd.DataFrame(columns=["Compound", "CAS", "SMILES", "Mass", "Adduct", "CCS", "Metadata"])
			df["Compound"] = f[dataset_name + '/Compound']
			df["CAS"] = f[dataset_name + '/CAS']
			df["SMILES"] = f[dataset_name + '/SMILES']
			df["Mass"] = f[dataset_name + '/Mass']
			df["Adduct"] = f[dataset_name + '/Adducts']
			df["CCS"] = f[dataset_name + '/CCS']
			df["Metadata"] = f[dataset_name + '/Metadata']
			
			df = df.applymap(lambda x: x.decode() if isinstance(x, bytes) else x) # convert the bytes to string
			valid_smiles = [True if len(row['SMILES']) != 0 else False for i, row in df.iterrows()]
			df = df[valid_smiles & df['Adduct'].notnull() & df['CCS'].notnull()]
			
			if dataset_name in ['MetCCS_test_neg', 'MetCCS_test_pos']:
				df_test_list.append(df)
			else:
				df_train_list.append(df)

	df_test = pd.concat(df_test_list, ignore_index=True)
	df_test.reset_index(inplace=True)
	df_test = df_test.rename(columns = {'index':'ID'})

	df_train = pd.concat(df_train_list, ignore_index=True)
	df_train.reset_index(inplace=True)
	df_train = df_train.rename(columns = {'index':'ID'})

	print(df_test)
	print(df_train)

	df_test.to_csv(os.path.join(args.out_dir, 'deepccs_test.csv'), index=False)
	df_train.to_csv(os.path.join(args.out_dir, 'deepccs_train.csv'), index=False)
	print('Done!')