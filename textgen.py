from collections import Counter

from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import numpy as np
from modules.utils import read_tokens


class ModelDataset(Dataset):
    def __init__(self, tokens: list):
        """__init__ for ModelDataset"""
        self.tokens = tokens
        self.n_snapshot = 3

        self.uniq_tokens = self.get_uniq_words(self.tokens)
        self.indexes = [self.get_id(i) for i in self.tokens]

    def __getitem__(self, item):
        offset = item + self.n_snapshot
        return (
            torch.tensor(self.indexes[item:offset], dtype=torch.float32),
            torch.tensor(self.indexes[item + 1:offset + 1], dtype=torch.float32),
        )

    def __len__(self):
        return len(self.tokens) - self.n_snapshot

    def get_id(self, token: str):
        return self.uniq_tokens.index(token)

    @classmethod
    def get_uniq_words(cls, tokens):
        token_counts = Counter(tokens)
        return sorted(token_counts, key=token_counts.get, reverse=True)


class Model(nn.Module):
    def __init__(self, dataset):
        super(Model, self).__init__()
        self.lstm_size = 128
        self.embedding_dim = 128
        self.num_layers = 3

        n_vocab = len(dataset.uniq_tokens)
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
            dtype=torch.float32,
        )
        self.fc = nn.Linear(self.lstm_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.embedding(x.long())
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size, dtype=torch.float32),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size, dtype=torch.float32))


def main():
    path = "data\\recensies.txt"
    tokens = read_tokens(path)

    dataset = ModelDataset(tokens)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = Model(dataset)

    n_epochs = 100
    every_iter = n_epochs / 10
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    current_loss = 0
    loss_history = []

    #training
    for epoch in range(1, n_epochs + 1):
        state_h, state_c = model.init_state(3)
        for batch, (src, tgt) in enumerate(loader):
            optimizer.zero_grad()

            output, (state_h, state_c) = model(src, (state_h, state_c))
            loss = criterion(output.transpose(1, 2), tgt.long())

            state_h = state_h.detach()
            state_c = state_c.detach()
            loss.backward()
            optimizer.step()

            current_loss += loss.item() / every_iter

        if epoch % every_iter == 0:
            print(f"Epoch: {epoch}/{n_epochs}, Loss: {current_loss}")
            loss_history.append(current_loss)
            current_loss = 0

    # predict
    model.eval()

    words = ['Том']
    state_h, state_c = model.init_state(len(words))

    for i in range(100):
        x = torch.tensor([[dataset.get_id(i) for i in words[i:]]])
        output, (state_h, state_c) = model(x, (state_h, state_c))

        last_word_logits = output[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        next_word = dataset.uniq_tokens[word_index]
        words.append(next_word)

    text_generated = ' '.join(words)
    print(f"""Text generated:
    {text_generated}""")


if __name__ == '__main__':
    main()
