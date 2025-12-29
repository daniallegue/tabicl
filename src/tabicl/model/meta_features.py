import torch
from torch import nn, Tensor


def basic_meta_stats(X: Tensor, y: Tensor) -> Tensor:
    """
    Distribution-aware meta-features for MoIP.

    X: (T, H)  numeric features
    y: (T,)    integer labels

    Returns:
        z_raw: (7,) =
          [label_entropy,
           max_class_frac,
           mean_std, std_std,
           mean_skew, std_skew,
           mean_abs_corr]
    """
    X = X.float()
    y = y.long()

    T, H = X.shape
    device = X.device
    eps = 1e-8

    # ---------- 1) Label statistics ----------
    if y.numel() == 0:
        label_entropy = torch.tensor(0.0, device=device)
        max_class_frac = torch.tensor(0.0, device=device)
    else:
        num_classes = int(y.max().item() + 1)
        hist = torch.bincount(y, minlength=num_classes).float()
        p = hist / (hist.sum() + eps)
        label_entropy = -(p * (p + eps).log()).sum()
        max_class_frac = p.max()

    # ---------- 2) Column-wise distributional moments ----------
    col_mean = X.mean(dim=0)                      # (H,)
    col_std = X.std(dim=0, unbiased=False) + eps  # (H,)

    centered = X - col_mean
    skew = (centered.pow(3).mean(dim=0)) / (col_std.pow(3))  # (H,)

    def agg_stats(v: Tensor):
        return v.mean(), v.std(unbiased=False)

    mean_std, std_std = agg_stats(col_std)
    mean_skew, std_skew = agg_stats(skew)

    # ---------- 3) Correlation structure ----------
    if H > 1:
        X_std = (X - col_mean) / col_std
        corr = torch.corrcoef(X_std.T)  # (H, H)
        off_diag_mask = ~torch.eye(H, dtype=torch.bool, device=device)
        off_diag_corr = corr[off_diag_mask]
        mean_abs_corr = off_diag_corr.abs().mean()
    else:
        mean_abs_corr = torch.tensor(0.0, device=device)

    return torch.stack(
        [
            label_entropy,   # 1
            max_class_frac,  # 2
            mean_std,        # 3
            std_std,         # 4
            mean_skew,       # 5
            std_skew,        # 6
            mean_abs_corr,   # 7
        ],
        dim=0,
    )


class MetaFeatureEncoder(nn.Module):
    """
    Minimal MLP mapping distributional meta-features to z_meta in R^{d_model}.

    d_raw = 7 (see simple_meta_stats_dist)
    """

    def __init__(self, d_model: int, hidden: int = 32) -> None:
        super().__init__()
        d_raw = 7
        self.net = nn.Sequential(
            nn.Linear(d_raw, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, X: Tensor, y: Tensor) -> Tensor:
        """
        X: (T, H)
        y: (T,)
        returns: z_meta (d_model,)
        """
        z_raw = basic_meta_stats(X, y)      # (7,)
        z_meta = self.net(z_raw.unsqueeze(0))     # (1, d_model)
        return z_meta.squeeze(0)                  # (d_model,)
