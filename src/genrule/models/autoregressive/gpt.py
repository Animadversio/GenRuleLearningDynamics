"""GPT2 model and sampling used by the research parity CLI."""
import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel


def build_gpt(config):
    """Build the legacy architecture from a config dict; does not load weights."""
    values=dict(config)
    values.setdefault('loss_type','ForCausalLMLoss')
    return GPT2LMHeadModel(GPT2Config(**values))


def sample_sequences_parallel(model, num_samples=10, max_length=36, sos_token=2, temperature=1.0, device='cuda', batch_size=None):
    """Sample sequences from the trained GPT model in parallel with optional batching"""
    model.eval()

    # Use single batch if batch_size not specified or large enough
    if batch_size is None or batch_size >= num_samples:
        batch_size = num_samples

    all_sequences = []

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_size_i = min(batch_size, num_samples - i)

            # Generate batch
            input_ids = torch.full((batch_size_i, 1), sos_token, dtype=torch.long).to(device)

            for step in range(max_length):
                outputs = model(input_ids)
                next_token_logits = outputs.logits[:, -1, :] / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, 1)
                input_ids = torch.cat([input_ids, next_tokens], dim=1)

            all_sequences.append(input_ids[:, 1:].cpu())  # Remove SOS token

    return torch.cat(all_sequences, dim=0)
