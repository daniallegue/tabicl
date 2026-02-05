from __future__ import annotations

from collections import OrderedDict
import math
import torch
from torch import nn, Tensor
from typing import Callable, Optional
import torch.nn.functional as F

from .layers import ClassNode, OneHotAndLinear
from .encoders import Encoder
from .inference import InferenceManager
from .inference_config import MgrConfig

class MoEBlock(nn.Module):
    """
    Mixture-of-Experts block with full expert observability.

    Supports two routing levels:
    - 'batch': All tokens in a batch use the same expert(s), routed by column context
    - 'token': Each token independently routed, conditioned on column context
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int = 4,
        d_hidden: Optional[int] = None,
        dropout: float = 0.0,
        gate_grad_scale: float = 1.0,
        routing_level: str = "batch",  # 'batch' or 'token'
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.gate_grad_scale = gate_grad_scale
        self.routing_level = routing_level

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

        # Auxiliary loss
        self.last_load_loss = None

        # Gating networks
        self.col_gate = nn.Linear(d_model, num_experts)  # For batch-level routing
        if routing_level == "token":
            self.token_gate = nn.Linear(d_model, num_experts)  # For token-level routing

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
    def init_from_ffn(self, lin1: nn.Linear, lin2: nn.Linear):
        """
        Initialize all experts from a pretrained FFN.

        Expected structure:
            lin1: Linear(d_model → d_hidden)
            lin2: Linear(d_hidden → d_model)
        """
        assert isinstance(lin1, nn.Linear)
        assert isinstance(lin2, nn.Linear)

        for expert in self.experts:
            expert[0].weight.copy_(lin1.weight)
            expert[0].bias.copy_(lin1.bias)
            expert[3].weight.copy_(lin2.weight)
            expert[3].bias.copy_(lin2.bias)

        # Neutral routing at init (uniform distribution over experts)
        nn.init.zeros_(self.col_gate.weight)
        nn.init.zeros_(self.col_gate.bias)
        if self.routing_level == "token":
            nn.init.zeros_(self.token_gate.weight)
            nn.init.zeros_(self.token_gate.bias)

    def forward(
        self,
        x: Tensor,
        column_context: Optional[Tensor] = None,
    ):
        B, T, D = x.shape
        assert D == self.d_model

        if self.routing_level == "token":
            return self._forward_token_level(x, column_context)
        else:
            return self._forward_batch_level(x, column_context)

    def _forward_batch_level(
        self,
        x: Tensor,
        column_context: Optional[Tensor] = None,
    ):
        """Batch-level routing: all tokens in a batch use same expert(s)."""
        B, _, _ = x.shape

        if column_context is not None:
            column_context = F.layer_norm(column_context, column_context.shape[-1:])
            gate_logits = self.col_gate(column_context)  # (B, E)

            if self.gate_grad_scale != 1.0:
                gate_logits = gate_logits * self.gate_grad_scale
        else:
            gate_logits = torch.zeros(B, self.num_experts, device=x.device, dtype=x.dtype)

        temperature = 0.7 if self.training else 0.5
        gate_weights = torch.softmax(gate_logits / temperature, dim=-1)

        if self.training:
            k = 2
            topk_vals, topk_idx = torch.topk(gate_weights, k=k, dim=-1)
            topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)

            y = torch.zeros_like(x)

            for expert_id in range(self.num_experts):
                expert_mask = (topk_idx == expert_id).any(dim=-1)  # (B,)

                if not expert_mask.any():
                    continue

                positions = (topk_idx == expert_id).long()  # (B, k)
                expert_weights = (positions * topk_vals).sum(dim=-1)  # (B,)

                expert_input = x[expert_mask]
                expert_output = self.experts[expert_id](expert_input)
                y[expert_mask] += expert_weights[expert_mask].view(-1, 1, 1) * expert_output

            self.last_gate_weights = gate_weights.detach()

            importance = gate_weights.mean(dim=0)
            self.last_load_loss = self.num_experts * (importance * importance).sum()

            return y

        # Inference: top-1 routing
        top_expert = gate_weights.argmax(dim=-1)  # (B,)
        y = torch.zeros_like(x)

        for expert_id in range(self.num_experts):
            mask = (top_expert == expert_id)
            if mask.any():
                y[mask] = self.experts[expert_id](x[mask])

        return y

    def _forward_token_level(
        self,
        x: Tensor,
        column_context: Optional[Tensor] = None,
    ):
        """Token-level routing: each token independently routed, conditioned on column context."""
        B, T, D = x.shape

        # Token-level gating conditioned on column context
        x_normed = F.layer_norm(x, x.shape[-1:])
        gate_logits = self.token_gate(x_normed)  # (B, T, E)

        # Add column context bias to routing (shared across tokens)
        if column_context is not None:
            col_normed = F.layer_norm(column_context, column_context.shape[-1:])
            col_bias = self.col_gate(col_normed)  # (B, E)
            gate_logits = gate_logits + col_bias.unsqueeze(1)  # (B, T, E)

        if self.gate_grad_scale != 1.0:
            gate_logits = gate_logits * self.gate_grad_scale

        temperature = 0.7 if self.training else 0.5
        gate_weights = torch.softmax(gate_logits / temperature, dim=-1)  # (B, T, E)

        if self.training:
            k = 2
            topk_vals, topk_idx = torch.topk(gate_weights, k=k, dim=-1)  # (B, T, k)
            topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)

            # Flatten for efficient processing
            x_flat = x.view(B * T, D)  # (B*T, D)
            topk_idx_flat = topk_idx.view(B * T, k)  # (B*T, k)
            topk_vals_flat = topk_vals.view(B * T, k)  # (B*T, k)

            y_flat = torch.zeros_like(x_flat)

            for expert_id in range(self.num_experts):
                expert_mask = (topk_idx_flat == expert_id).any(dim=-1)  # (B*T,)

                if not expert_mask.any():
                    continue

                positions = (topk_idx_flat == expert_id).long()  # (B*T, k)
                expert_weights = (positions * topk_vals_flat).sum(dim=-1)  # (B*T,)

                expert_input = x_flat[expert_mask]  # (num_tokens, D)
                expert_output = self.experts[expert_id](expert_input)
                y_flat[expert_mask] += expert_weights[expert_mask].unsqueeze(-1) * expert_output

            y = y_flat.view(B, T, D)

            self.last_gate_weights = gate_weights.mean(dim=1).detach()  # Average over tokens for logging

            # Load balancing loss over all tokens
            importance = gate_weights.mean(dim=(0, 1))  # (E,)
            self.last_load_loss = self.num_experts * (importance * importance).sum()

            return y

        # Inference: top-1 routing per token
        x_flat = x.view(B * T, D)
        gate_flat = gate_weights.view(B * T, self.num_experts)
        top_expert = gate_flat.argmax(dim=-1)  # (B*T,)

        y_flat = torch.zeros_like(x_flat)

        for expert_id in range(self.num_experts):
            mask = (top_expert == expert_id)
            if mask.any():
                y_flat[mask] = self.experts[expert_id](x_flat[mask])

        return y_flat.view(B, T, D)

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
        moe_gate_grad_scale: float = 1.0,
        moe_routing_level: str = "batch",
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
                dropout=dropout,
                gate_grad_scale=moe_gate_grad_scale,
                routing_level=moe_routing_level,
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

    def _icl_predictions(self, R: Tensor, y_train: Tensor, column_context: Tensor | None = None):
        train_size = y_train.shape[1]
        R[:, :train_size] += self.y_encoder(y_train.float())

        src = self.tf_icl(R, attn_mask=train_size)
        if self.norm_first:
            src = self.ln(src)

        if self.use_moe_icl and self.moe_block is not None:
            src = self.moe_block(src, column_context=column_context)

        return self.decoder(src)

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

    def _predict_standard(self, R, y_train, return_logits, softmax_temperature, auto_batch=True, column_context=None):
        train_size = y_train.shape[1]
        num_classes = len(torch.unique(y_train[0]))

        inputs = OrderedDict(R=R, y_train=y_train, column_context=column_context)

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

    def _inference_forward(self, R, y_train, return_logits, softmax_temperature, mgr_config, column_context):
        if mgr_config is None:
            mgr_config = MgrConfig()

        self.inference_mgr.configure(**mgr_config)

        num_classes = len(torch.unique(y_train[0]))

        if num_classes <= self.max_classes:
            return self._predict_standard(R, y_train, return_logits, softmax_temperature, column_context=column_context)

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
        column_context: Tensor | None = None,
    ):
        if self.training:
            train_size = y_train.shape[1]
            return self._icl_predictions(R, y_train, column_context)[:, train_size:]

        return self._inference_forward(
            R,
            y_train,
            return_logits,
            softmax_temperature,
            mgr_config,
            column_context
        )
