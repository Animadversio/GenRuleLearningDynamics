"""Legacy score-model training loop, preserving callback/checkpoint conventions."""
import torch
from torch.optim import Adam, AdamW
from tqdm.auto import trange

def train_score_model_custom_loss(X_train_tsr, score_model_td, loss_fn,
                   lr=0.005,
                   nepochs=750,
                   batch_size=None,
                   device="cpu",
                   callback=None,
                   callback_freq=100,
                   callback_step_list=[],
                   save_ckpts=False,
                   save_ckpt_step_list=[],
                   ckpt_dir=None,
                   weight_decay=0.0):
    """
    Trains a score model using a custom loss function with an optional callback.

    Args:
        X_train_tsr (torch.Tensor): Training data tensor of shape [N, ndim].
        score_model_td (nn.Module): The score model to be trained.
        loss_fn (callable): The loss function to compute the loss.
        lr (float, optional): Learning rate for the optimizer. Defaults to 0.005.
        nepochs (int, optional): Number of training epochs. Defaults to 750.
        batch_size (int, optional): Batch size for training. If None, uses the entire dataset. Defaults to None.
        device (str, optional): Device to train on ("cpu" or "cuda"). Defaults to "cpu".
        callback (callable, optional): A function to call at specified intervals. Should accept arguments (epoch, loss, model).
        callback_freq (int, optional): Frequency (in epochs) to call the callback function. Defaults to 100. Set to None or 0 or negative to disable -> can use callback_step_list instead.
        callback_step_list (list, optional): List of steps to call the callback function. Defaults to [].
    Returns:
        tuple: Trained score model and a list of loss values per epoch.
    """
    def check_callback_step(ep):
        if ep == nepochs - 1 or ep == 0:
            # always call callback for the first and last epoch
            return True
        if callback_freq is not None and callback_freq > 0 and (ep + 1) % callback_freq == 0:
            # If callback_freq is set, call callback every callback_freq epochs
            return True
        if callback_step_list is not None and (ep + 1) in callback_step_list:
            # If callback_step_list is set, call callback for the list of steps in callback_step_list
            return True
        if save_ckpts and (ep in save_ckpt_step_list):
            return True
        return False

    ndim = X_train_tsr.shape[1]
    score_model_td.to(device)
    X_train_tsr = X_train_tsr.to(device)
    # Use AdamW when weight_decay > 0 (decoupled weight decay); Adam otherwise
    if weight_decay > 0:
        optim = AdamW(score_model_td.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optim = Adam(score_model_td.parameters(), lr=lr)
    pbar = trange(nepochs)
    score_model_td.train()
    loss_traj = []
    for ep in pbar:
        if batch_size is None:
            loss = loss_fn(score_model_td, X_train_tsr, )
        else:
            idx = torch.randint(0, X_train_tsr.shape[0], (batch_size,))
            loss = loss_fn(score_model_td, X_train_tsr[idx], )
        loss = loss.mean()
        optim.zero_grad()
        loss.backward()
        # Compute grad norm only at callback steps (avoid per-step overhead)
        grad_norm = None
        if callback is not None and check_callback_step(ep):
            grad_norm = torch.nn.utils.clip_grad_norm_(
                score_model_td.parameters(), max_norm=float('inf')).item()
        optim.step()
        pbar.set_description(f"step {ep} loss {loss.item():.3f}")
        if ep == 0:
            print(f"step {ep} loss {loss.item():.3f}")
        loss_traj.append(loss.item())

        # Invoke callback if specified and if epoch matches the frequency
        if callback is not None and check_callback_step(ep):
            callback(epoch=ep + 1, loss=loss.item(), model=score_model_td, grad_norm=grad_norm)

        if save_ckpts and (ep in save_ckpt_step_list or ep == nepochs - 1 or ep == 0):
            torch.save(score_model_td.state_dict(), f"{ckpt_dir}/model_epoch_{ep:06d}.pth")

    return score_model_td, loss_traj
