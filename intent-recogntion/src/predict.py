import re
import pickle
import spacy
import torchtext
import torch
from config import *
from models.cnn import CNN

pretrained_model_path = "/content/drive/MyDrive/Models/INTENT/cnn-model.pt"
pretrained_vocab_path = "/content/drive/MyDrive/Models/INTENT/cnndict.pkl"

# load spacy's nlp model for tokenization
nlp = spacy.load("en")
# load the model
model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)

# load the pretrained model path
model.load_state_dict(torch.load(pretrained_model_path))
# load the pretrained vocab file
with open(pretrained_vocab_path, "rb") as f:
    TEXT = pickle.load(f)


def predict_class(intent, model=model):
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(intent)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    preds = model(tensor)
    max_pred = preds.argmax(dim=1)
    return max_pred.item()
