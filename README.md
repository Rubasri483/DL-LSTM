# DL- Developing a Deep Learning Model for NER using LSTM

## AIM
To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset
Problem Statement and Dataset: Build a Named Entity Recognition (NER) model that can automatically identify and classify entities like names of people, locations, organizations, and other important terms from text. The goal is to tag each word in a sentence with its corresponding entity label.

## DESIGN STEPS
### STEP 1:
Load data, create word/tag mappings, and group sentences.

### STEP 2:
Convert sentences to index sequences, pad to fixed length, and split into training/testing sets.

### STEP 3:
Define dataset and DataLoader for batching.

### STEP 4:
Build a bidirectional LSTM model for sequence tagging.

### STEP 5:
Train the model over multiple epochs, tracking loss.





## PROGRAM

### Name: Rubasri R

### Register Number:212224240139

```python
class BiLSTM_NER(nn.Module):
    def __init__(self, vocab_size, num_tags, embedding_dim=128, hidden_dim=128, dropout_rate=0.3):
        super(BiLSTM_NER, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        self.dropout = nn.Dropout(dropout_rate)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # bidirectional=True doubles hidden dimension (hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, num_tags)

    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.dropout(self.embedding(x))   # (batch_size, seq_len, embed_dim)
        lstm_out, _ = self.lstm(embedded)           # (batch_size, seq_len, hidden_dim * 2)
        logits = self.fc(self.dropout(lstm_out))     # (batch_size, seq_len, num_tags)
        return logits
        
model = BiLSTM_NER(
    vocab_size=len(word2idx),
    num_tags=len(tag2idx),
    embedding_dim=128,
    hidden_dim=128
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=tag2idx["<PAD>"])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training and Evaluation Functions
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):
   train_losses, val_losses = [], []

    for epoch in range(epochs):

        model.train()

        total_loss = 0

        for batch in train_loader:

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids)

            loss = loss_fn(
                outputs.view(-1, len(tag2idx)),
                labels.view(-1)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_losses.append(total_loss)

        model.eval()

        val_loss = 0

        with torch.no_grad():

            for batch in test_loader:

                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids)

                loss = loss_fn(
                    outputs.view(-1, len(tag2idx)),
                    labels.view(-1)
                )

                val_loss += loss.item()

        val_losses.append(val_loss)

        print(
            f"Epoch {epoch+1}: "
            f"Train Loss = {total_loss:.4f}, "
            f"Val Loss = {val_loss:.4f}"
        )

    return train_losses, val_losses


```

### OUTPUT

## Loss Vs Epoch Plot

<img width="496" height="71" alt="image" src="https://github.com/user-attachments/assets/681995e9-2750-4884-b2ac-d085ba99c170" />

<img width="745" height="614" alt="image" src="https://github.com/user-attachments/assets/d4aba2ba-1b62-45c5-9a0f-46b172e9e18c" />



### Sample Text Prediction
Include your sample text prediction here
<img width="402" height="518" alt="image" src="https://github.com/user-attachments/assets/cf5423ca-15d3-4d4c-9f64-9a4e9b5a5096" />


## RESULT
Thus, an LSTM-based model for recognizing the named entities in the text has been developed successfully.


