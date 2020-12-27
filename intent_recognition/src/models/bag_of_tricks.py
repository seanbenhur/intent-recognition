import torch
import torch.nn as nn
import torch.nn.functional as F 


class FastText(nn.Module):
  def __init__(self,input_dim,embedding_dim,output_dim):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size,embedding_dim)
    self.fc = nn.Linear(embedding_dim,output_dim)
  
  def forward(self,text):
    #text = [sent_len,batch_size]
    embedded = self.embedding(text)
    #embedded = [sent_len,batch_size,emb_dim]
    embedded = embedded.permute(1,0,2)
    #embedded =  [batch_size,sent_len,emb_dim]
    pooled = F.avg_pool2d(embedded,(embedded.shape[1],1)).squeeze(1)
    #pooled = [batch_size,sent_len]
    return self.fc(pooled)




