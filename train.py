import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import pandas as pd

from dataset import SmilesCCSDataset
from model import DeepCCS



def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return param_group['lr']

def train_step(model, loader, optimizer, num_sym, num_add, device):
	criterion = nn.MSELoss()

	with tqdm(total=len(loader)) as bar:
		for step, batch in enumerate(loader): 
			_, smiles, add, y = batch

			# encode adduct and instrument by one hot
			smiles = F.one_hot(smiles, num_classes=num_sym)
			add = F.one_hot(add, num_classes=num_add)

			smiles = smiles.type(torch.cuda.FloatTensor).to(device)
			add = add.type(torch.cuda.FloatTensor).to(device)
			y = y.type(torch.cuda.FloatTensor).to(device)

			optimizer.zero_grad()
			model.train()
			pred = model(smiles, add)
			loss = criterion(pred, y)
			loss.backward()

			bar.set_description('Train')
			bar.set_postfix(lr=get_lr(optimizer), loss=loss.item())
			bar.update(1)

			optimizer.step()
	return 

def eval_step(model, loader, num_sym, num_add, device): 
	model.eval()
	y_true = []
	y_pred = []
	spec_ids = []
	acc = []
	with tqdm(total=len(loader)) as bar:
		for _, batch in enumerate(loader):
			spec_id, smiles, add, y = batch

			# encode adduct and instrument by one hot
			smiles = F.one_hot(smiles, num_classes=num_sym)
			add = F.one_hot(add, num_classes=num_add)

			smiles = smiles.type(torch.cuda.FloatTensor).to(device)
			add = add.type(torch.cuda.FloatTensor).to(device)
			y = y.type(torch.cuda.FloatTensor).to(device)

			with torch.no_grad():
				pred = model(smiles, add)

			bar.set_description('Eval')
			bar.update(1)

			y_true.append(y.detach().cpu())
			y_pred.append(pred.detach().cpu())
			acc = acc + torch.mean(torch.abs(y - pred), dim=1).tolist()
			spec_ids = spec_ids + list(spec_id)

	y_true = torch.cat(y_true, dim = 0)
	y_pred = torch.cat(y_pred, dim = 0)
	return spec_ids, acc, y_true, y_pred



if __name__ == "__main__":
	# Training settings
	parser = argparse.ArgumentParser(description='Mass Spectra to formula (train)')
	parser.add_argument('--train_data', type=str, required=True,
						help='Path to training data')
	parser.add_argument('--test_data', type=str, required=True,
						help='Path to test data')
	parser.add_argument('--smiles_dict_path', type=str, default = './encode_smiles.json',
						help='Path to SMILES encoding dictionary')

	parser.add_argument('--checkpoint_path', type=str, default = '',
						help='Path to save checkpoint')
	parser.add_argument('--resume_path', type=str, default='',
						help='Path to pretrained model')
	parser.add_argument('--result_path', type=str, default = '',
						help='Path to save predicted results')
	parser.add_argument('--device', type=int, default=0,
						help='Which gpu to use if any (default: 0)')
	args = parser.parse_args()

	# 0. Settings
	len_smiles = 250
	# num_sym = 38
	num_add = 5

	lr = 0.001
	batch_size = 16
	epoch_num = 100
	early_stop_step = 20
	early_stop_patience = 0
	best_valid_acc = 9999

	# 1. Data
	train_set = SmilesCCSDataset(data_path=args.train_data, len_smiles=len_smiles, num_add=num_add, smiles_dict_path=args.smiles_dict_path)
	val_set = SmilesCCSDataset(data_path=args.test_data, len_smiles=len_smiles, num_add=num_add, smiles_dict_path=args.smiles_dict_path)
	num_sym = train_set.num_sym() # get the symbol number of SMILES representation
	assert num_sym == val_set.num_sym()

	train_loader = torch.utils.data.DataLoader(train_set,
												batch_size=batch_size,
												shuffle=True,
												num_workers=0,
												drop_last=True)
	val_loader = torch.utils.data.DataLoader(val_set,
												batch_size=batch_size,
												shuffle=False,
												num_workers=0,
												drop_last=True)

	# 2. Model
	device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
	print(f'Device: {device}')
	model = DeepCCS(num_sym, num_add)
	num_params = sum(p.numel() for p in model.parameters())
	print(f'{str(model)} #Params: {num_params}')
	# print(f'#Params: {num_params}')
	model.to(device)

	# 3. Train
	optimizer = optim.AdamW(model.parameters(), lr=lr)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
	# load the checkpoints
	if args.resume_path != '':
		print("Load the checkpoints...")
		epoch_start = torch.load(args.resume_path, map_location=device)['epoch']
		model.load_state_dict(torch.load(args.resume_path, map_location=device)['model_state_dict'])
		optimizer.load_state_dict(torch.load(args.resume_path, map_location=device)['optimizer_state_dict'])
		scheduler.load_state_dict(torch.load(args.resume_path, map_location=device)['scheduler_state_dict'])
		best_valid_acc = torch.load(args.resume_path)['best_val_acc']
	else:
		epoch_start = 1

	for epoch in range(epoch_start, epoch_num+1): 
		print('\nEpoch {}'.format(epoch))
		train_step(model, train_loader, optimizer, num_sym, num_add, device)

		spec_ids, acc, y_true, y_pred = eval_step(model, val_loader, num_sym, num_add, device)
		valid_acc = np.mean(acc)
		print("Validation error: {}".format(valid_acc))
		
		if valid_acc < best_valid_acc: 
			best_valid_acc = valid_acc

			if args.checkpoint_path != '':
				print('Saving checkpoint...')
				checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_val_acc': best_valid_acc, 'num_params': num_params}
				torch.save(checkpoint, args.checkpoint_path)

			early_stop_patience = 0
			print('Early stop patience reset')
		else:
			early_stop_patience += 1
			print('Early stop count: {}/{}'.format(early_stop_patience, early_stop_step))

		scheduler.step(valid_acc) # ReduceLROnPlateau
		print(f'Best absolute error so far: {best_valid_acc}')

		if early_stop_patience == early_stop_step: 
			print('Early stop!')
			break

	if args.result_path != '':
		print('Save the predicted results...')
		res_df = pd.DataFrame({'ID': spec_ids, 'CCS Exp': y_true.tolist(), 'CCS Pred': y_pred.tolist()})
		res_df.to_csv(args.result_path, sep='\t')

	print('Done!')