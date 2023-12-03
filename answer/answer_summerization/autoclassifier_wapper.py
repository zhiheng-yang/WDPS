import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

# helper function


def exists(val):
    return val is not None


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


# top k filtering


def top_k(logits, thres=0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(1, ind, val)
    return probs


class AutoClassifierWrapper(nn.Module):
    def __init__(self, net, max_seq_len=2048, pad_value=0):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.pad_value = pad_value
        self.net = net

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        context,
        temperature=1.0,
        filter_thres=0.9,
        **kwargs
    ):
        b_q, t_q, device_q = *context.shape, context.device
        out = t_q

        logits = self.net(out, **kwargs)[:, -1, :]
        filtered_logits = top_k(logits, thres=filter_thres)
        probs = F.softmax(filtered_logits / temperature, dim=-1)
        sample = torch.multinomial(probs, 1)
        return sample


    def forward(self, x, **kwargs):
        logits = self.net(x)
        return logits