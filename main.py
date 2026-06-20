import os

import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


DATA_PATH = "train_and_test2.csv"
CHECKPOINT_PATH = "checkpoints/titanic_weights.pt"

FEATURES = [
    "Pclass",
    "Sex",
    "Age",
    "sibsp",
    "Parch",
    "Fare",
    "Embarked"
]


class TitanicNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


model = TitanicNetwork()

if os.path.exists(CHECKPOINT_PATH):
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.eval()
    print("Weights loaded.")
else:
    print("No weights found.")


loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

df = pd.read_csv(DATA_PATH)

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())
df["Embarked"] = df["Embarked"].fillna(1)

X_all = torch.tensor(df[FEATURES].values, dtype=torch.float32)
y_all = torch.tensor(df["2urvived"].values, dtype=torch.float32).unsqueeze(1)
train_size = int(len(X_all) * 0.8)

X_train, y_train = X_all[:train_size], y_all[:train_size]
X_test, y_test = X_all[train_size:], y_all[train_size:]

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


def train(epochs=10):
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")

    os.makedirs(
        os.path.dirname(CHECKPOINT_PATH),
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        CHECKPOINT_PATH
    )


def test():
    model.eval()
    total_loss = 0.0

    with torch.inference_mode():
        for batch_x, batch_y in test_loader:
            pred = model(batch_x)

            loss = loss_fn(pred, batch_y)

            total_loss += (
                loss.item() * batch_x.size(0)
            )

    print(f"Loss: {total_loss / len(test_dataset):.4f}")


def accuracy():
    model.eval()

    correct = 0
    total = len(test_dataset)

    with torch.inference_mode():
        for batch_x, batch_y in test_loader:
            pred = model(batch_x)

            predicted = (pred >= 0.5).float()

            correct += (
                predicted == batch_y
            ).sum().item()

    print(
        f"Accuracy: "
        f"{correct / total * 100:.2f}%"
    )

if __name__ == '__main__':
    while True:
        print("1 | Train\n2 | Test\n3 | Accuracy")
        try:
            user_ans = int(input())
            if user_ans == 1: train()
            elif user_ans == 2: test()
            elif user_ans == 3: accuracy()
        except ValueError:
            print("Incorrect input")