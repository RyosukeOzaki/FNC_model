import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WACE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, vocab_size, emb_size, embedding, dropout_rate, device):
        super(WACE, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.emb = nn.Embedding(vocab_size, emb_size)
        self.emb.weight = nn.Parameter(torch.Tensor(embedding), requires_grad=False)
        self.dropout_rate = dropout_rate
        self.r = []
        self.lstm_p = nn.LSTMCell(input_size, hidden_size).to(self.device)
        self.lstm_h = nn.LSTMCell(input_size, hidden_size).to(self.device)
        self.wy = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.wh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.wr = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.w = nn.Parameter(torch.randn(hidden_size, 1))
        wt, _ = np.linalg.qr(np.random.randn(hidden_size, hidden_size))
        self.wt = nn.Parameter(torch.Tensor(wt))
        self.wp = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.wx = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.fc = nn.Linear(hidden_size, num_classes).to(self.device)

    def init(self, x):
        # Set initial states
        h0 = torch.zeros(x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(x.size(0), self.hidden_size).to(self.device)
        self.r = []
        self.r.append(torch.zeros(len(x), self.hidden_size).to(self.device))
        return h0, c0

    def mask(self, r_t, r, mask_t):
        return (mask_t.expand(*r_t.size()) * r_t) + ((1. - mask_t.expand(*r_t.size())) * (r))

    def mask_LSTM_output(self, out, cell, h_p, c_p, mask):
        mask_out = torch.zeros(out.size()).type(torch.FloatTensor).to(self.device)
        mask_cell = torch.zeros(cell.size()).type(torch.FloatTensor).to(self.device)
        for t, mask_t in enumerate(mask):
            h = out[:, t, :]
            c = cell[:, t, :]
            h_p = self.mask(h, h_p, mask_t.unsqueeze(1))
            c_p = self.mask(c, c_p, mask_t.unsqueeze(1))
            mask_out[:, t, :] = h_p
            mask_cell[:, t, :] = c_p
        return mask_out, mask_cell

    def attention_forward(self, Y, h, r, mask_p, mask_t):
        mask_p = mask_p.transpose(1, 0)
        first = torch.matmul(Y, self.wy.unsqueeze(0).expand(-1, self.hidden_size, self.hidden_size)).to(self.device)
        second = (torch.matmul(h, self.wh)+torch.matmul(r, self.wr)).unsqueeze(1).expand(-1, Y.size(1), self.hidden_size).to(self.device)
        M = torch.tanh(first + second).to(self.device)
        alpha = torch.matmul(M, self.w.unsqueeze(0).expand(-1, self.hidden_size, 1)).squeeze(-1)
        alpha = alpha + (-1000.0 * (1. - mask_p))
        alpha = F.softmax(alpha, dim=1)
        r_t = torch.matmul(alpha.unsqueeze(1), Y).squeeze(1) + torch.tanh(torch.matmul(r, self.wt))
        r_t = self.mask(r_t, r, mask_t)
        return r_t, alpha

    def forward(self, premise, hypothesis):
        # Forward propagate LSTM
        embedded_p = self.emb(premise)
        embedded_h = self.emb(hypothesis)
        mask_p = torch.ne(premise, 0).type(torch.FloatTensor).to(self.device)
        mask_h = torch.ne(hypothesis, 0).type(torch.FloatTensor).to(self.device)
        mask_p = mask_p.transpose(1, 0)
        mask_h = mask_h.transpose(1, 0)
        hx, cx = self.init(embedded_p)
        h0, c0 = hx, cx
        out_p = []
        cell_p = []
        out_h = []
        cell_h = []
        for t_p, _ in enumerate(mask_p):
            hx, cx = self.lstm_p(embedded_p[:, t_p, :], (hx, cx))
            out_p.append(hx)
            cell_p.append(cx)
        out_p = torch.stack(out_p, dim=1)
        cell_p = torch.stack(cell_p, dim=1)
        out_p, cell_p = self.mask_LSTM_output(out_p, cell_p, h0, c0, mask_p)
        hx, cx = h0, cell_p[:, -1, :]
        for t_h, _ in enumerate(mask_h):
            hx, cx = self.lstm_h(embedded_h[:, t_h, :], (hx, cx))
            out_h.append(hx)
            cell_h.append(cx)
        out_h = torch.stack(out_h, dim=1)
        cell_h = torch.stack(cell_h, dim=1)
        out_h, cell_h = self.mask_LSTM_output(out_h, cell_h, h0, cell_p[:, -1, :], mask_h)
        for t, mask_t in enumerate(mask_h):
            r_t, alpha = self.attention_forward(out_p, out_h[:, t, :], self.r[-1], mask_p, mask_t.unsqueeze(1))
            self.r.append(r_t)
        h_star = torch.tanh(torch.matmul(self.r[-1], self.wp) + torch.matmul(out_h[:, -1, :], self.wx)).to(self.device)
        outputs = self.fc(h_star)
        outputs = F.dropout(outputs, self.dropout_rate, training=self.training)
        if self.training:
            outputs = F.log_softmax(outputs, dim=1)
        else:
            outputs = F.softmax(outputs, dim=1)
        return outputs
