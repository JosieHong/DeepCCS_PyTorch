import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F



class DeepCCS(nn.Module):
	def __init__(self, num_sym, num_add): 
		super(DeepCCS, self).__init__() 
		self.smiles_encoder = nn.Sequential(nn.Conv1d(num_sym, 64, kernel_size=4, stride=1), 
											nn.ReLU(), 

											nn.Conv1d(64, 64, kernel_size=4, stride=1), 
											nn.ReLU(), 
											nn.MaxPool1d(kernel_size=2, stride=1), 

											nn.Conv1d(64, 64, kernel_size=4, stride=1), 
											nn.ReLU(), 
											nn.MaxPool1d(kernel_size=2, stride=1), 

											nn.Conv1d(64, 64, kernel_size=4, stride=1), 
											nn.ReLU(), 
											nn.MaxPool1d(kernel_size=2, stride=1), 

											nn.Conv1d(64, 64, kernel_size=4, stride=1), 
											nn.ReLU(), 
											nn.MaxPool1d(kernel_size=2, stride=1), 

											nn.Conv1d(64, 64, kernel_size=4, stride=1), 
											nn.ReLU(), 
											nn.MaxPool1d(kernel_size=2, stride=1), 
											
											nn.Conv1d(64, 64, kernel_size=4, stride=1), 
											nn.ReLU(), 
											nn.MaxPool1d(kernel_size=2, stride=2), 
											)

		self.decoder = nn.Sequential(nn.Linear(7168+num_add, 384), 
									nn.ReLU(), 

									nn.Linear(384, 384), 
									nn.ReLU(), 

									nn.Linear(384, 384), 
									nn.ReLU(), 

									nn.Linear(384, 1), 
									)

		for m in self.modules(): 
			if isinstance(m, (nn.Linear, nn.Conv1d)):
				nn.init.xavier_normal_(m.weight)
		
	def forward(self, smiles, add): 
		smiles = smiles.permute(0, 2, 1)
		x = self.smiles_encoder(smiles)
		x = x.view(x.size(0), -1)
		x = torch.cat((x, add), 1)
		
		x = self.decoder(x)
		return x