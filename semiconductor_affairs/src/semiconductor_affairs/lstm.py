"""LSTM regressor with an explicit PyTorch training loop.

Same interface as the baselines (fit(X, y) / predict(X) on arrays of shape
(n, window, features)), so it plugs straight into evaluate.walk_forward.
"""

import numpy as np
import torch
from torch import nn


class LSTMNet(nn.Module):
    """Window of feature vectors -> one number (next-hour return).

    The LSTM reads the window one hour at a time, updating its hidden
    state h. Only the final h (after the most recent hour) goes to the
    linear head that makes the prediction.
    """

    def __init__(self, n_features: int, hidden: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):  # x: (batch, window, features)
        # outputs at every hour, and the final (h, c); only final h is used
        _outputs, (h_n, _c_n) = self.lstm(x)  # h_n: (1, batch, hidden)
        return self.head(self.dropout(h_n[-1])).squeeze(-1)


class LSTMRegressor:
    """Scaling + training loop + early stopping around LSTMNet."""

    def __init__(
        self,
        hidden: int = 16,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 50,
        patience: int = 5,
        val_fraction: float = 0.2,
        seed: int = 0,
    ):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.seed = seed
        self.history: list[dict] = []

    # --- scaling -------------------------------------------------------
    # Inputs: per-feature mean/std from the training part only.
    # Target: divided by its training std, so the loss is ~1 at the start
    # and the network's output layer works with numbers of normal size.

    def _scale_x(self, X):
        return torch.tensor((X - self.x_mean) / self.x_std, dtype=torch.float32)

    def fit(self, X: np.ndarray, y: np.ndarray):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        # Chronological split: the last part of the training data is used
        # to decide when to stop (early stopping), never shuffled in.
        n_val = int(len(X) * self.val_fraction)
        X_tr, y_tr = X[:-n_val], y[:-n_val]
        X_val, y_val = X[-n_val:], y[-n_val:]

        self.x_mean = X_tr.mean(axis=(0, 1))
        self.x_std = X_tr.std(axis=(0, 1))
        self.y_std = y_tr.std()

        xt, xv = self._scale_x(X_tr), self._scale_x(X_val)
        yt = torch.tensor(y_tr / self.y_std, dtype=torch.float32)
        yv = torch.tensor(y_val / self.y_std, dtype=torch.float32)

        self.net = LSTMNet(X.shape[2], self.hidden, self.dropout)
        opt = torch.optim.Adam(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = nn.MSELoss()

        best_val, best_state, bad_epochs = np.inf, None, 0
        self.history = []
        for epoch in range(self.max_epochs):
            # one pass over the training windows, in random mini-batches
            self.net.train()
            order = rng.permutation(len(xt))
            train_loss = 0.0
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.net(xt[idx]), yt[idx])
                loss.backward()  # backprop through time over the window
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                train_loss += loss.item() * len(idx)
            train_loss /= len(xt)

            # how well does it do on data it didn't train on?
            self.net.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.net(xv), yv).item()
            # loss of predicting 0 on the same validation data, for scale
            val_zero = (yv**2).mean().item()
            self.history.append(
                {"epoch": epoch, "train": train_loss, "val": val_loss,
                 "val_zero": val_zero}
            )

            if val_loss < best_val:
                best_val, bad_epochs = val_loss, 0
                best_state = {
                    k: v.clone() for k, v in self.net.state_dict().items()
                }
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        self.net.load_state_dict(best_state)  # weights of the best epoch
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            out = self.net(self._scale_x(X)).numpy()
        return out * self.y_std
