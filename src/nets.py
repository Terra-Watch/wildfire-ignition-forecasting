"""The neural models. One input per model type (see harness.Cube.gather):
  mlp       [B, F]                  one cell-day
  lstm      [B, 10, F]              ten days of one cell
  cnn       [B, F+1, 5, 5]          the cell and its neighbours on day t (the extra channel is 1 where the neighbour exists)
  convlstm  [B, 10, F+1, 5, 5]      the neighbours over ten days
  c1        same input as convlstm  ConvLSTM + attention over the ten days and over the 25 cells
  c2        same input as convlstm  graph model: one GRU per cell, two rounds of message passing over the neighbours that exist
All end in one logit. Sizes are kept close to each other (about 0.1 million weights for the three recurrent image-like models).
"""
import torch, torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, cin, hid):
        super().__init__(); self.hid = hid; self.conv = nn.Conv2d(cin + hid, 4 * hid, 3, padding=1)

    def forward(self, x, h, c):
        i, f, o, g = self.conv(torch.cat([x, h], 1)).chunk(4, 1); c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        return torch.sigmoid(o) * torch.tanh(c), c


class ConvLSTM(nn.Module):
    def __init__(self, cin, hid=32, drop=0.2, cell=0):
        super().__init__(); self.hid = hid; self.cell = ConvLSTMCell(cin, hid); self.own = nn.LSTM(cin - 1, 64, batch_first=True) if cell else None; self.fc1 = nn.Linear(hid * 25 + (64 if cell else 0), 64); self.drop = nn.Dropout(drop); self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        B, T = x.shape[:2]; h = x.new_zeros(B, self.hid, 5, 5); c = torch.zeros_like(h)
        for s in range(T): h, c = self.cell(x[:, s], h, c)
        z = h.flatten(1) if self.own is None else torch.cat([h.flatten(1), self.own(x[:, :, :-1, 2, 2])[0][:, -1]], 1)           # cell path: an LSTM over the middle cell's own ten days
        return self.fc2(self.drop(torch.relu(self.fc1(z)))).squeeze(-1)


class ConvLSTMAttention(nn.Module):
    """C1: the same ConvLSTM, but the end state is read through attention. Over the ten days the hidden state of the middle cell is weighted by a learned score;
    over the 25 cells of the last map a query from the middle cell weights every cell that exists (absent cells are masked). The two summaries replace the flattened map."""
    def __init__(self, cin, hid=32, att=32, drop=0.2):
        super().__init__(); self.hid, self.att = hid, att; self.cell = ConvLSTMCell(cin, hid)
        self.tq = nn.Linear(hid, att); self.tv = nn.Linear(att, 1, bias=False); self.q = nn.Linear(hid, att); self.k = nn.Linear(hid, att)
        self.fc1 = nn.Linear(2 * hid, 64); self.drop = nn.Dropout(drop); self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        B, T = x.shape[:2]; h = x.new_zeros(B, self.hid, 5, 5); c = torch.zeros_like(h); mids = []
        valid = x[:, -1, -1].flatten(1) > 0.5                                                         # [B, 25]: the neighbour exists
        for s in range(T): h, c = self.cell(x[:, s], h, c); mids.append(h[:, :, 2, 2])
        mids = torch.stack(mids, 1)                                                                   # [B, T, hid]
        wt = torch.softmax(self.tv(torch.tanh(self.tq(mids))).squeeze(-1), 1); tctx = (wt[..., None] * mids).sum(1)
        last = h.flatten(2).transpose(1, 2)                                                           # [B, 25, hid]
        sc = (self.k(last) * self.q(last[:, 12:13])).sum(-1) / self.att ** 0.5; ws = torch.softmax(sc.masked_fill(~valid, -1e4), 1); sctx = (ws[..., None] * last).sum(1)
        return self.fc2(self.drop(torch.relu(self.fc1(torch.cat([tctx, sctx], 1))))).squeeze(-1)


def adjacency():
    A = torch.zeros(25, 25)
    for a in range(25):
        for b in range(25):
            if a != b and abs(a // 5 - b // 5) <= 1 and abs(a % 5 - b % 5) <= 1: A[a, b] = 1
    return A


class GraphModel(nn.Module):
    """C2: every one of the 25 positions is a node with its own ten-day GRU; two rounds of message passing over the 8-neighbour graph restricted to nodes that exist;
    the end state of the middle node and the mean over existing nodes go to the head. Only neighbours that exist take part (nothing is filled with zeros)."""
    def __init__(self, cin, hid=96, drop=0.2, rounds=2, cell=0):
        super().__init__(); self.gru = nn.GRU(cin - 1, hid, batch_first=True); self.ls = nn.ModuleList(nn.Linear(hid, hid) for _ in range(rounds)); self.ln = nn.ModuleList(nn.Linear(hid, hid) for _ in range(rounds))
        self.own = nn.LSTM(cin - 1, 64, batch_first=True) if cell else None; self.fc1 = nn.Linear(2 * hid + (64 if cell else 0), 64); self.drop = nn.Dropout(drop); self.fc2 = nn.Linear(64, 1); self.register_buffer("A", adjacency())

    def forward(self, x):
        B, T, C = x.shape[:3]; nodes = x.flatten(3).permute(0, 3, 1, 2)                                  # [B, 25, T, C]
        valid = (nodes[:, :, -1, -1] > 0.5).float()[..., None]                                        # [B, 25, 1]
        _, h = self.gru(nodes[..., :-1].reshape(B * 25, T, C - 1)); h = h[0].reshape(B, 25, -1) * valid
        A = self.A[None] * valid.transpose(1, 2) * valid; deg = A.sum(-1, keepdim=True).clamp(min=1)
        for ls, ln in zip(self.ls, self.ln): h = torch.relu(ls(h) + ln((A @ h) / deg)) * valid
        z = torch.cat([h[:, 12], h.sum(1) / valid.sum(1).clamp(min=1)] + ([] if self.own is None else [self.own(x[:, :, :-1, 2, 2])[0][:, -1]]), 1)
        return self.fc2(self.drop(torch.relu(self.fc1(z)))).squeeze(-1)


class LSTM(nn.Module):
    def __init__(self, F):
        super().__init__(); self.l = nn.LSTM(F, 64, 1, batch_first=True); self.drop = nn.Dropout(0.2); self.fc = nn.Linear(64, 1)

    def forward(self, x): return self.fc(self.drop(self.l(x)[0][:, -1])).squeeze(-1)


class Squeeze(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return self.m(x).squeeze(-1)


def build(name, F, **kw):   # kw: hid, drop, rounds (only for the three patch models; the default settings are the main runs)
    if name == "mlp": return Squeeze(nn.Sequential(nn.Linear(F, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1)))
    if name == "lstm": return LSTM(F)
    if name == "cnn": return Squeeze(nn.Sequential(nn.Conv2d(F + 1, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Flatten(), nn.Linear(64 * 25, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1)))
    if name == "convlstm": return ConvLSTM(F + 1, **{k: v for k, v in kw.items() if k in ("hid", "drop", "cell")})
    if name == "c1": return ConvLSTMAttention(F + 1, **{k: v for k, v in kw.items() if k in ("hid", "drop")})
    if name == "c2": return GraphModel(F + 1, **kw)
    raise ValueError(name)


# one fixed training setting per neural model; the learning rate is the only grid (two values) so that every model gets the same search effort
NEURAL = {"mlp": dict(view="tabular", batch=4096, epochs=20), "lstm": dict(view="sequence", batch=2048, epochs=20), "cnn": dict(view="patch", batch=2048, epochs=20),
          "convlstm": dict(view="patch_sequence", batch=1024, epochs=15), "c1": dict(view="patch_sequence", batch=1024, epochs=15), "c2": dict(view="patch_sequence", batch=1024, epochs=15)}
