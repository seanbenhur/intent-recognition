import re
import pickle
import spacy
import torchtext
import torch
from models.lstm import LSTM  

pretrained_model_path = "some path"
pretrained_vocab_path = "some path"

#load spacy's nlp model for tokenization
nlp = spacy.load("en")
#load the model
model = LSTM(
    INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT
)

#load the pretrained model path
model.load_state_dict(torch.load(pretrained_model_path),map_location="cpu")
#load the pretrained vocab file
with open(pretrained_vocab_path,"rb") as f:
	TEXT = pickle.load(f)

def predict_intent(intent,model=model):
	model.eval()
	#convert text to lower case
	intent = intent.lower()
	#remove punctuation
	intent = re.sub(r'[^\w\s]','',intent)
	tokenized = [tok.text for tok in nlp.tokenizer(intent)]
    indexed = [TEXT.stoi(t) for t in tokenized]
    tensor = torch.LongTensor(indexed)
    preds = model(tensor)
    max_pred = preds.argmax(dim=1)
    return max_pred