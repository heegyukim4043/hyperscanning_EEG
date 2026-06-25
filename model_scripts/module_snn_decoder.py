import torch
import torch.nn as nn


class _SurrogateSpike(torch.autograd.Function):
    """
    Binary spike with smooth surrogate gradient.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, beta: float):
        ctx.save_for_backward(x)
        ctx.beta = beta
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        beta = ctx.beta
        # fast-sigmoid style surrogate derivative
        grad = 1.0 / (beta * x.abs() + 1.0).pow(2)
        return grad_output * grad, None


def surrogate_spike(x: torch.Tensor, beta: float = 10.0):
    return _SurrogateSpike.apply(x, float(beta))


class RecurrentSNNReconstruction(nn.Module):
    """
    Recurrent SNN decoder for sequence reconstruction.

    Interface matches TransformerReconstruction:
      forward(memory [B,W,enc_dim], x_target [B,W,out_dim]) -> recons [B,W,out_dim]
    """

    def __init__(
        self,
        window_size: int,
        enc_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
        threshold: float = 1.0,
        surrogate_beta: float = 10.0,
        learnable_decay: bool = True,
    ):
        super().__init__()
        self.window_size = int(window_size)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = max(int(num_layers), 1)
        self.surrogate_beta = float(surrogate_beta)

        self.enc_in = nn.Linear(enc_dim, hidden_dim)
        self.token_in = nn.Linear(out_dim, hidden_dim)

        self.recurrent_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.recurrent_layers.append(
                nn.ModuleDict(
                    {
                        "rec": nn.Linear(hidden_dim, hidden_dim, bias=False),
                        "ff": nn.Linear(hidden_dim, hidden_dim, bias=True),
                        "norm": nn.LayerNorm(hidden_dim),
                    }
                )
            )

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.start_token = nn.Parameter(torch.zeros(1, 1, out_dim))

        # decay constrained to (0,1) via sigmoid
        init_decay = torch.tensor(0.90).clamp(1e-4, 1 - 1e-4)
        init_logit = torch.log(init_decay / (1 - init_decay))
        decay = init_logit.repeat(self.num_layers)
        if learnable_decay:
            self.decay_logit = nn.Parameter(decay)
        else:
            self.register_buffer("decay_logit", decay)

        thr = torch.full((self.num_layers,), float(threshold))
        self.threshold = nn.Parameter(thr)

    def _decay(self):
        return torch.sigmoid(self.decay_logit)

    def forward(self, memory: torch.Tensor, x_target: torch.Tensor):
        # memory: [B,W,enc_dim], x_target: [B,W,out_dim]
        bsz, win, _ = x_target.shape
        if win != self.window_size:
            # keep compatibility for variable lookback runs
            self.window_size = win

        start = self.start_token.expand(bsz, 1, self.out_dim)
        shifted = torch.cat([start, x_target[:, :-1, :]], dim=1)  # teacher forcing

        mem_proj = self.enc_in(memory)            # [B,W,H]
        tok_proj = self.token_in(shifted)         # [B,W,H]
        inp = mem_proj + tok_proj

        decay = self._decay()                     # [L]
        threshold = self.threshold                # [L]

        # states per recurrent SNN layer
        v_states = [inp.new_zeros(bsz, self.hidden_dim) for _ in range(self.num_layers)]
        s_states = [inp.new_zeros(bsz, self.hidden_dim) for _ in range(self.num_layers)]

        outs = []
        for t in range(win):
            x_t = inp[:, t, :]  # [B,H]
            for li, layer in enumerate(self.recurrent_layers):
                pre = x_t + layer["ff"](x_t) + layer["rec"](s_states[li])
                v_states[li] = decay[li] * v_states[li] + pre
                s_t = surrogate_spike(v_states[li] - threshold[li], beta=self.surrogate_beta)
                # soft reset
                v_states[li] = v_states[li] - s_t * threshold[li]
                s_t = layer["norm"](s_t)
                if li < self.num_layers - 1:
                    s_t = self.dropout(s_t)
                s_states[li] = s_t
                x_t = s_t

            y_t = self.out_proj(x_t)  # [B,out_dim]
            outs.append(y_t.unsqueeze(1))

        recons = torch.cat(outs, dim=1)  # [B,W,out_dim]
        return recons
