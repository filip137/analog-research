"""OPT decoder ported from HWA b0c1f06; no cross-worktree runtime imports."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

@dataclass(frozen=True)
class OptConfig:
    vocab_size: int
    hidden_size: int
    decoder_layers: int
    attention_heads: int
    ffn_dim: int
    max_positions: int

class Attention(nn.Module):
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.heads, self.head_dim = heads, hidden // heads
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)

    def forward(self, x):
        b, t, d = x.shape

        def heads(value):
            return value.view(b, t, self.heads, self.head_dim).transpose(1, 2)

        q = heads(self.q_proj(x) * self.head_dim**-0.5)
        k, v = heads(self.k_proj(x)), heads(self.v_proj(x))
        scores = q @ k.transpose(-1, -2)
        causal = torch.ones(t, t, dtype=torch.bool, device=x.device).triu(1)
        weights = scores.masked_fill(causal, float("-inf")).softmax(-1)
        return self.out_proj((weights @ v).transpose(1, 2).reshape(b, t, d))

class DecoderLayer(nn.Module):
    """Pre-LayerNorm OPT decoder layer with a ReLU MLP."""

    def __init__(self, config: OptConfig):
        super().__init__()
        self.self_attn = Attention(config.hidden_size, config.attention_heads)
        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size)
        self.fc1 = nn.Linear(config.hidden_size, config.ffn_dim)
        self.fc2 = nn.Linear(config.ffn_dim, config.hidden_size)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)

    def attention(self, x):
        return x + self.self_attn(self.self_attn_layer_norm(x))

    def forward(self, x):
        x = self.attention(x)
        return x + self.fc2(F.relu(self.fc1(self.final_layer_norm(x))))

class OptDecoder(nn.Module):
    """FP32 OPT decoder with learned positions (offset 2) and a tied LM head."""

    POSITION_OFFSET = 2

    def __init__(self, config: OptConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_positions = nn.Embedding(config.max_positions + self.POSITION_OFFSET, config.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(config) for _ in range(config.decoder_layers))
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)

    def embed(self, tokens):
        positions = torch.arange(tokens.shape[1], device=tokens.device) + self.POSITION_OFFSET
        return self.embed_tokens(tokens) + self.embed_positions(positions)

    def logits(self, hidden):
        return self.final_layer_norm(hidden) @ self.embed_tokens.weight.T

    def forward(self, tokens):
        hidden = self.embed(tokens)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.logits(hidden)

def from_hugging_face(state: dict, config: OptConfig) -> OptDecoder:
    """Build the decoder from an ``OPTForCausalLM`` state dict (any float dtype)."""

    decoder = {k.removeprefix("model.decoder."): v for k, v in state.items() if k.startswith("model.decoder.")}
    head = state.get("lm_head.weight")
    if head is not None and not torch.equal(head, decoder["embed_tokens.weight"]):
        raise ValueError("Expected the LM head to be tied to the token embedding.")
    model = OptDecoder(config)
    model.load_state_dict({k: v.float() for k, v in decoder.items()}, strict=True)
    return model
