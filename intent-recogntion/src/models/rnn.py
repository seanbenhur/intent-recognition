import torch
import torch.nn as nn
import torch.nn.functional as F 

class RNN(nn.Module):
  def __init__(self,input_dim,embedding_dim,hidden_dim,output_dim):
    super().__init__()
    #embedding layer
    self.embedding = nn.Embedding(input_dim,embedding_dim)
    #rnn layer
    self.rnn = nn.RNN(embedding_dim,hidden_dim)
    #last fc layer
    self.fc = nn.Linear(hidden_dim,output_dim)
 
  def forward(self,text):
    #x = [sent_len,batch_size]
    embedded =  self.embedding(text)
    #embedded = [sent_len,batch_size,emb_dim]
    #rnn outputs both hidden dim and output
    output,hidden = self.rnn(embedded)
    #output = [sent_len,batch_size,hid_dim]
    #hidden = [1,batch_size,hid_dim]
    assert torch.equal(output[-1,:,:],hidden.squeeze(0))
    return self.fc(hidden.squeeze(0))


