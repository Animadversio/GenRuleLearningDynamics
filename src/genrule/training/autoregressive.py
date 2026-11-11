"""One original GPT AdamW update; loss shifting remains inside Hugging Face."""
def train_step(model, optimizer, batch_data):
    optimizer.zero_grad()
    input_ids = batch_data
    outputs = model(input_ids=input_ids, labels=input_ids)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    return loss.detach()
