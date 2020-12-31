import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from torchtext import data
import random
from config import *
from models.lstm import LSTM


torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_save_path = "/content/drive/MyDrive/Models/INTENT/lstm-model.pt"


def generate_bigrams(x):

    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(" ".join(n_gram))
    return x


TEXT = data.Field(tokenize="spacy", include_lengths=True)
LABEL = data.LabelField()
fields = [(None, None), ("text", TEXT), ("label", LABEL), (None, None)]

train_data, test_data = data.TabularDataset.splits(
    path="/content/drive/MyDrive/data/benchmarking_data",
    train="train.csv",
    test="valid.csv",
    format="csv",
    fields=fields,
    skip_header=True,
)

# create a  new validation set from test data
train_data, valid_data = train_data.split(random_state=random.seed(SEED))

# build vocab
TEXT.build_vocab(
    train_data,
    max_size=MAX_VOCAB_SIZE,
    vectors="glove.6B.100d",
    unk_init=torch.Tensor.normal_,
)
LABEL.build_vocab(train_data)

train_dl, valid_dl, test_dl = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device,
    sort_within_batch=True,
    sort_key=lambda x: len(x.text),
)


model = LSTM(
    INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT
)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

model = model.to(device)
criterion = criterion.to(device)


def accuracy(y_pred, y):
    """
    Returns a accuracy score
    """
    max_preds = y_pred.argmax(dim=1, keepdim=True)
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum() / torch.FloatTensor([y.shape[0]])


# train the model
def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:

        optimizer.zero_grad()

        text, text_lengths = batch.text

        predictions = model(text, text_lengths)

        loss = criterion(predictions, batch.label)

        acc = accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:

            text, text_lengths = batch.text

            predictions = model(text, text_lengths)

            loss = criterion(predictions, batch.label)

            acc = accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


best_valid_loss = float("inf")

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss, train_acc = train(model, train_dl, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_dl, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(
            model.state_dict(), "/content/drive/MyDrive/Models/INTENT/lstm-model.pt"
        )

    print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
    print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
    print(f"\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%")
