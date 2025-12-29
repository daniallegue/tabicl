from __future__ import annotations

from collections import OrderedDict
import math
import torch
from torch import nn, Tensor
from typing import Callable, Optional, Tuple

from .layers import ClassNode, OneHotAndLinear
from .encoders import Encoder
from .inference import InferenceManager
from .inference_config import MgrConfig

class MoEBlock(nn.Module):
    """
    Mixture-of-Experts block with full expert observability.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int = 4,
        d_hidden: Optional[int] = None,
        num_priors: Optional[int] = None,
        use_moip: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.use_moip = use_moip and (num_priors is not None)

        if d_hidden is None:
            d_hidden = 2 * d_model

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_hidden, d_model),
            )
            for _ in range(num_experts)
        ])

        if self.use_moip:
            self.moip_emb = nn.Embedding(num_priors, d_model)
            gate_in_dim = d_model
        else:
            self.moip_emb = None
            gate_in_dim = d_model

        self.gate = nn.Linear(gate_in_dim, num_experts)

        # Observability
        self.last_gate_weights: Optional[Tensor] = None
        self.last_expert_outputs: Optional[Tensor] = None
        self.register_buffer("expert_grad_norms", torch.zeros(num_experts))

        self._register_expert_grad_hooks()

    def _register_expert_grad_hooks(self):
        for idx, expert in enumerate(self.experts):
            weight = expert[-1].weight

            def _make_hook(i):
                def hook(grad):
                    self.expert_grad_norms[i] += grad.norm().detach()
                return hook

            weight.register_hook(_make_hook(idx))

    @torch.no_grad()
    def init_from_ffn(self, ffn: nn.Module):
        """
        Initialize experts from a pretrained FFN.
        Expected structure:
            Linear -> Activation -> Linear
        """
        assert isinstance(ffn, nn.Sequential)
        src_lin1 = ffn[0]
        src_lin2 = ffn[2]

        for expert in self.experts:
            expert[0].weight.copy_(src_lin1.weight)
            expert[0].bias.copy_(src_lin1.bias)
            expert[3].weight.copy_(src_lin2.weight)
            expert[3].bias.copy_(src_lin2.bias)

        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(
        self,
        x: Tensor,
        moip_ids: Optional[Tensor] = None,
        return_expert_outputs: bool = False,
    ):
        B, T, D = x.shape
        assert D == self.d_model

        if self.use_moip and moip_ids is not None:
            context = self.moip_emb(moip_ids)
        else:
            context = x.mean(dim=1)

        temperature = 1.0 if self.training else 0.5
        gate_logits = self.gate(context) / temperature
        gate_weights = torch.softmax(gate_logits, dim=-1)

        if self.training:
            y = torch.zeros_like(x)

            expert_outputs = [] if return_expert_outputs else None

            for i, expert in enumerate(self.experts):
                w = gate_weights[:, i].view(B, 1, 1)  # (B, 1, 1)
                out_i = expert(x)  # (B, T, D)
                y = y + w * out_i

                if return_expert_outputs:
                    expert_outputs.append(out_i)

            self.last_gate_weights = gate_weights.detach()
            if return_expert_outputs:
                self.last_expert_outputs = torch.stack(expert_outputs, dim=1).detach()
                return y, self.last_expert_outputs, gate_weights

            return y

        # Inference: top-1 routing
        top_expert = gate_weights.argmax(dim=-1)  # (B,)
        y = torch.zeros_like(x)

        for i, expert in enumerate(self.experts):
            mask = (top_expert == i)
            if mask.any():
                y[mask] = expert(x[mask])

        if return_expert_outputs:
            # Not meaningful in sparse inference; return None for compatibility
            return y, None, gate_weights

        return y

    def reset_expert_grad_stats(self):
        self.expert_grad_norms.zero_()

class ICLearning(nn.Module):
    """
    Dataset-wise in-context learning with hierarchical classification.
    """

    def __init__(
        self,
        max_classes: int,
        d_model: int,
        num_blocks: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str | Callable = "gelu",
        norm_first: bool = True,
        *,
        use_moe_icl: bool = False,
        moe_num_experts: int = 4,
        moe_hidden_mult: float = 2.0,
        moe_num_priors: int | None = None,
        moe_use_moip: bool = True,
    ):
        super().__init__()
        self.max_classes = max_classes
        self.norm_first = norm_first
        self.use_moe_icl = use_moe_icl

        self.tf_icl = Encoder(
            num_blocks=num_blocks,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
        )

        if norm_first:
            self.ln = nn.LayerNorm(d_model)

        if use_moe_icl:
            self.moe_block = MoEBlock(
                d_model=d_model,
                num_experts=moe_num_experts,
                d_hidden=int(moe_hidden_mult * d_model),
                num_priors=moe_num_priors,
                use_moip=moe_use_moip,
                dropout=dropout,
            )
        else:
            self.moe_block = None

        self.y_encoder = OneHotAndLinear(max_classes, d_model)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, max_classes),
        )

        self.inference_mgr = InferenceManager(enc_name="tf_icl", out_dim=max_classes)

    # ------------------------------------------------------------------
    # Core ICL forward
    # ------------------------------------------------------------------
    def _icl_predictions(self, R: Tensor, y_train: Tensor, moip_ids: Tensor | None = None):
        train_size = y_train.shape[1]
        R[:, :train_size] += self.y_encoder(y_train.float())

        src = self.tf_icl(R, attn_mask=train_size)
        if self.norm_first:
            src = self.ln(src)

        if self.use_moe_icl and self.moe_block is not None:
            src = self.moe_block(src, moip_ids=moip_ids)

        return self.decoder(src)

    # ------------------------------------------------------------------
    # Hierarchical helpers (UNCHANGED from repo)
    # ------------------------------------------------------------------
    def _grouping(self, num_classes: int):
        if num_classes <= self.max_classes:
            return torch.zeros(num_classes, dtype=torch.int), 1

        num_groups = min(math.ceil(num_classes / self.max_classes), self.max_classes)
        group_assignments = torch.zeros(num_classes, dtype=torch.int)

        pos = 0
        for g in range(num_groups):
            size = math.ceil((num_classes - pos) / (num_groups - g))
            group_assignments[pos:pos + size] = g
            pos += size

        return group_assignments, num_groups

    def _fit_node(self, node, R: Tensor, y: Tensor, depth: int):
        unique = torch.unique(y).int()
        node.classes_ = unique

        if len(unique) <= self.max_classes:
            node.is_leaf = True
            node.R = R
            node.y = y
            return

        groups, n_groups = self._grouping(len(unique))
        node.class_mapping = {c.item(): g.item() for c, g in zip(unique, groups)}
        node.group_indices = torch.tensor([node.class_mapping[c.item()] for c in y])
        node.R = R
        node.y = y
        node.is_leaf = False

        for g in range(n_groups):
            mask = node.group_indices == g
            child = ClassNode(depth + 1)
            self._fit_node(child, R[mask], y[mask], depth + 1)
            node.child_nodes.append(child)

    def _fit_hierarchical(self, R_train: Tensor, y_train: Tensor):
        self.root = ClassNode(depth=0)
        self._fit_node(self.root, R_train, y_train, 0)

    def _label_encoding(self, y: Tensor):
        unique, _ = torch.unique(y, return_inverse=True)
        idx = unique.argsort()
        return idx[torch.searchsorted(unique, y)]

    def _predict_standard(self, R, y_train, return_logits, softmax_temperature, auto_batch=True, moip_ids=None):
        train_size = y_train.shape[1]
        num_classes = len(torch.unique(y_train[0]))

        inputs = OrderedDict(R=R, y_train=y_train)
        if moip_ids is not None:
            inputs["moip_ids"] = moip_ids

        out = self.inference_mgr(self._icl_predictions, inputs=inputs, auto_batch=auto_batch)
        out = out[:, train_size:, :num_classes]

        if not return_logits:
            out = torch.softmax(out / softmax_temperature, dim=-1)

        return out

    def _predict_hierarchical(self, R_test: Tensor, softmax_temperature: float):
        device = R_test.device
        test_size = R_test.shape[0]
        num_classes = len(self.root.classes_)

        def recurse(node, R_test):
            node_R = torch.cat([node.R.to(device), R_test], dim=0)

            if node.is_leaf:
                y_enc = self._label_encoding(node.y.to(device))
                probs = self._predict_standard(
                    node_R.unsqueeze(0),
                    y_enc.unsqueeze(0),
                    return_logits=False,
                    softmax_temperature=softmax_temperature,
                    auto_batch=False,
                ).squeeze(0)

                out = torch.zeros(test_size, num_classes, device=device)
                for i, c in enumerate(node.classes_):
                    out[:, c] = probs[:, i]
                return out

            probs = self._predict_standard(
                node_R.unsqueeze(0),
                node.group_indices.unsqueeze(0),
                return_logits=False,
                softmax_temperature=softmax_temperature,
                auto_batch=False,
            ).squeeze(0)

            out = torch.zeros(test_size, num_classes, device=device)
            for g, child in enumerate(node.child_nodes):
                out += recurse(child, R_test) * probs[:, g:g+1]

            return out

        return recurse(self.root, R_test)

    def _inference_forward(self, R, y_train, return_logits, softmax_temperature, mgr_config, moip_ids):
        if mgr_config is None:
            mgr_config = MgrConfig()

        self.inference_mgr.configure(**mgr_config)

        num_classes = len(torch.unique(y_train[0]))

        if num_classes <= self.max_classes:
            return self._predict_standard(R, y_train, return_logits, softmax_temperature, moip_ids=moip_ids)

        out = []
        train_size = y_train.shape[1]
        for ri, yi in zip(R, y_train):
            self._fit_hierarchical(ri[:train_size], yi)
            probs = self._predict_hierarchical(ri[train_size:], softmax_temperature)
            out.append(probs)

        out = torch.stack(out)
        if return_logits:
            out = softmax_temperature * torch.log(out + 1e-6)

        return out

    def get_moe_blocks(self):
        return [self.moe_block] if self.moe_block is not None else []

    def forward(
        self,
        R: Tensor,
        y_train: Tensor,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        mgr_config: MgrConfig | None = None,
        moip_ids: Tensor | None = None,
    ):
        if self.training:
            train_size = y_train.shape[1]
            return self._icl_predictions(R, y_train, moip_ids)[:, train_size:]

        return self._inference_forward(
            R,
            y_train,
            return_logits,
            softmax_temperature,
            mgr_config,
            moip_ids,
        )
