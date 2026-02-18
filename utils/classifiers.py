from einops import einsum
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
import numpy as np


# New centered classifier (Black magic)
class CenteredLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Domain Adaptive Logistic Regression.
    Centers the Training data to 0, and Centers the Test data to 0 *independently*.
    This removes the "Length Vector" shift between Long and Short datasets.
    """
    def __init__(self, C=1.0, max_iter=2000):
        self.C = C
        self.max_iter = max_iter
        self.model = LogisticRegression(C=self.C, max_iter=self.max_iter, solver='lbfgs')
        self.train_mean_ = None

    def fit(self, X, y):
        self.train_mean_ = np.mean(X, axis=0)
        X_centered = X - self.train_mean_
        
        self.model.fit(X_centered, y)
        return self

    def predict(self, X):
        test_mean = np.mean(X, axis=0)
        X_centered = X - test_mean
        return self.model.predict(X_centered)

    def predict_proba(self, X):
        # Same logic for probabilities
        test_mean = np.mean(X, axis=0)
        X_centered = X - test_mean
        return self.model.predict_proba(X_centered)
    
    def decision_function(self, X):
        test_mean = np.mean(X, axis=0)
        X_centered = X - test_mean
        return self.model.decision_function(X_centered)


class ContrastiveProbe(nn.Module):
    """
    Contrastive classifier using supervised contrastive loss + linear head.
    Learns to pull embeddings of same class together, push apart others.
    """

    def __init__(self, in_dim, out_dim=1, temperature=0.07, projection_dim=64, train_mode=None, d_prob=0.0):
        super().__init__()
        self.temperature = temperature
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        self.classifier = nn.Linear(projection_dim, out_dim)
        # store them if needed for compatibility
        self.train_mode = train_mode
        self.d_prob = d_prob

    def forward(self, x, return_projection=False, test_time_order=None):
        z = F.normalize(self.encoder(x), dim=-1)
        logits = self.classifier(z)
        if return_projection:
            return logits, z
        return logits

    def contrastive_loss(self, features, labels):
        """
        Supervised contrastive loss from Khosla et al. (2020).
        """
        device = features.device
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # cosine similarity
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # mask out self-comparisons
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()
        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0]).to(device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        loss = -mean_log_prob_pos.mean()
        return loss

class PolyProbe(nn.Module):
    def __init__(self, in_features, out_features, max_order=3, ranks=[64, 128, 256], d_prob=0.0, linear_init=None, train_linear=False, train_mode='', term_drop=True):
        super().__init__()
        print(f'PolyProbe: in_features={in_features}, out_features={out_features}, max_order={max_order}')
        self.layer_type = 'Poly_CP'
        self.train_mode = train_mode
        self.out_features = out_features
        self.d_prob = d_prob
        self.ranks = ranks
        self.max_order = max_order
        self.in_features = in_features
        self.term_drop = term_drop
        
        self.lam = nn.ParameterList([nn.Parameter(torch.randn(ranks[o])*0.02, requires_grad=True) for o in range(max_order-1)])

        # 0th and 1st order terms
        self.linear = nn.Linear(in_features, out_features, bias=True)

        self.W = nn.ParameterList([
            self.linear.bias,     # bias parameter
            self.linear.weight,   # weight parameter
        ])

        self.HO = [] # store the higher-order terms
        for order in range(max_order-1):
            self.HO.append(nn.Parameter(torch.nn.Linear(in_features, ranks[order]).weight, requires_grad=True))

        self.W = nn.ParameterList(self.W)
        self.HO = nn.ParameterList(self.HO)
        
        if linear_init is not None:
            self.W[0].data = torch.Tensor(linear_init[0], device=self.W[0].device)
            self.W[1].data = torch.Tensor(linear_init[1], device=self.W[1].device)
            print(f'Using {"(trainable)" if train_linear else "(frozen)"} linear probe initialization for PolyLayer!')

    def forward(self, x, test_time_order=None):
        # linear term
        y = einsum(self.W[1], x, 'o i, ... i -> ... o') + self.W[0]

        if self.term_drop and self.training:
            dout_mask = (torch.rand((x.shape[0]), device=x.device) > self.d_prob).float() if self.training else torch.ones((x.shape[0]), device=x.device)
            y = y * dout_mask.view(-1, 1)
        
        ys = [y]

        # loop over higher-orders
        for n in range(min(test_time_order, self.max_order)-1):
            order = n+2
            inner = einsum(x, self.HO[n], '... i, r i -> ... r') ** (order) # contract input with factor matrix, raise to power 'order'
            yn = einsum(inner, self.lam[n], '... r, r -> ...').unsqueeze(-1) # sum over the rank dimension

            # optional dropout for previous terms
            if self.term_drop and self.training and order < test_time_order:
                dout_mask = (torch.rand((x.shape[0]), device=x.device) > self.d_prob).float() if self.training else torch.ones((x.shape[0]), device=x.device)
                yn = yn * dout_mask.view(-1, 1)

            y = y + yn
            ys.append(y)

        return ys

################################################################
# enter: baselines etc below
################################################################

class LinearProbe(nn.Module):
    def __init__(self, input_dim, output_dim=1, **kwargs):
        super().__init__()
        self.net = nn.Linear(input_dim, output_dim)
        self.layer_type = 'Linear'

    # kwargs absorbs the order flags (e.g. test-time order)
    def forward(self, x: torch.Tensor, **kwargs):
        y = self.net(x)
        return [y]  # we're wrapping the [output] in a list with one element just to match the interface of the other probes


class BilinearProbe(nn.Module):
    """
    Bilinear probe with symmetric decomposition, and factorized forward pass

    Note here: this corrresponds to the 2nd order term from the polynomial expansion alone
    """
    def __init__(self, in_features, out_features, rank=64, **kwargs):
        super().__init__()
        #print(f'BilinearProbe: in_features={in_features}, out_features={out_features}, rank={rank}')
        self.layer_type = 'Bilinear'

        self.symmetric = True

        if not self.symmetric:
            # CP factors
            Wi1 = nn.Parameter(torch.nn.Linear(in_features, rank).weight)
            Wi2 = nn.Parameter(torch.nn.Linear(in_features, rank).weight)
            Wo = nn.Parameter(torch.nn.Linear(rank, out_features).weight)
            self.W = nn.ParameterList([Wi1, Wi2, Wo])
        else:
            W = nn.Parameter(torch.nn.Linear(in_features, rank).weight)
            self.lam = nn.Parameter(torch.randn(rank)*0.02, requires_grad=True)
            self.W = nn.ParameterList([W])

        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32), requires_grad=True)
        self.rank = rank
        self.in_features = in_features
        self.out_features = out_features
        
    # kwargs absorbs the order flags (e.g. test-time order)
    def forward(self, x, testing=False, **kwargs):
        if self.symmetric:
            y1 = einsum(x, self.W[0], '... i, r i -> ... r')**2
            y = einsum(y1, self.lam, '... r, r -> ...').unsqueeze(-1) + self.bias # contract the rank dimension with the output mode
        else:
            y1 = einsum(x, self.W[0], '... i, r i -> ... r')
            y2 = einsum(x, self.W[1], '... i, r i -> ... r')
            y = einsum(y1, y2, self.W[2], '... r, ... r, o r -> ... o') + self.bias

        return [y]  # we're wrapping the [output] in a list with one element just to match the interface of the other probes


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, d_prob=0.0, **kwargs):
        super().__init__()
        self.layer_type = 'MLP'
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=d_prob) if d_prob > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, output_dim, bias=True),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        y = self.net(x)
        return [y]  # we're wrapping the [output] in a list with one element just to match the interface of the other probes

#Revisit this implementation

class EEMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims=[256, 128],
        num_layers=2,
        output_dim=1,
        increasing_depth=False,
        d_prob=0.0,
        **kwargs
    ):
        super().__init__()
        self.layer_type = 'EEMLP'

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims] * num_layers

        if len(hidden_dims) != num_layers:
            print(f"[EEMLP] Warning: hidden_dims ({len(hidden_dims)}) != num_layers ({num_layers}). Adjusting automatically.")
            if len(hidden_dims) < num_layers:
                hidden_dims = hidden_dims + [hidden_dims[-1]] * (num_layers - len(hidden_dims))
            else:
                hidden_dims = hidden_dims[:num_layers]

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=d_prob) if d_prob > 0 else nn.Identity()
            )
            for (in_dim, hidden_dim) in zip([input_dim] + hidden_dims[:-1], hidden_dims)
        ])

        # Side output branches (one per layer + input)
        self.side_branches = nn.ModuleList([
            nn.Linear(input_dim, output_dim),  # branch 0
            *[nn.Linear(h, output_dim) for h in hidden_dims]
        ])

    def forward(self, x: torch.Tensor, test_time_order: int = None):
        # ✅ default to all available branches
        if test_time_order is None:
            test_time_order = len(self.side_branches)

        ys = [self.side_branches[0](x)]
        for i in range(test_time_order - 1):
            x = self.layers[i](x)
            ys.append(self.side_branches[i + 1](x))
        return ys