import re
import pickle
import spacy
import torchtext
import torch
from config import *
from models.lstm import LSTM  

pretrained_model_path = "/content/drive/MyDrive/Models/INTENT/lstm-model.pt"
pretrained_vocab_path = "/content/drive/MyDrive/data/dict.pkl"

#load spacy's nlp model for tokenization
nlp = spacy.load("en")
#load the model
model = LSTM(
    INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT
)

#load the pretrained model path
model.load_state_dict(torch.load(pretrained_model_path))
#load the pretrained vocab file
with open(pretrained_vocab_path,"rb") as f:
	TEXT = pickle.load(f)


def predict_class(intent,model=model):
  model.eval()
  tokenized = [tok.text for tok in nlp.tokenizer(intent)]
  indexed = [TEXT.vocab.stoi[t] for t in tokenized]
  tensor = torch.LongTensor(indexed).to(device)
  tensor = tensor.unsqueeze(1)
  preds = model(tensor)
  max_pred = preds.argmax(dim=1)
  return max_pred.item()