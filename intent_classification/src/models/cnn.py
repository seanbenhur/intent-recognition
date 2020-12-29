import torch
import torch.nn as nn
import torch.nn.functional as F 

class CNN(nn.Module):

	def __init__(self,
		input_dim,
		embedding_dim,
		n_filters,
		filter_sizes,
		output_dim,
		dropout):

		super.__init__()

		self.embedding = nn.Embedding(
			input_dim,embedding_dim)

		self.convs = nn.ModuleList([
			nn.Conv2d(in_channels=1, out_channels=n_filters,
				kernel_size=(fs,embedding_dim)) for fs in filter_sizes])
    
   		 self.fc = nn.Linear(len(filter_sizes)*n_filters,output_dim)

    def forward(self,text):

		text = text.permute(1,0)
		#text passed through embedding layer to get embeddings
		embedded = self.embedding(text)
		'''
        A conv layer wants the second dim of the input to be a channel dim
        text does not have a channel dim, so the tensor is unsqueezed to create one
        '''
		embedded = embedded.unsqueeze(1)

		conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
		
		pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2)
				 for conv in conved]

		cat = self.dropout(torch.cat(pooled,dim=1))
		#passed through linear layer to make predictions
		return self.fc(cat)


	 