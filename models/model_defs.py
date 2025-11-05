import torch
import torchdiffeq
import torch.nn as nn
import dgl

"""This class is directly from torchgde models and is not
   our original work. All credit to original programmers. 
   Link to torchgde GitHub: https://github.com/Zymrael/gde/
   Date of Retrieval: September 2024
"""
class ODEBlock(nn.Module):
    def __init__(self, odefunc: nn.Module, method: str = 'dopri5', rtol: float = 1e-3, atol: float = 1e-4, adjoint: bool = True):
        """Standard ODEBlock class. Can handle all types of ODE functions."""
        super().__init__()
        self.odefunc = odefunc
        self.method = method
        self.adjoint_flag = adjoint
        self.atol, self.rtol = atol, rtol

    def forward(self, x: torch.Tensor, T: int = 1, time_points: int = 2, return_trajectory: bool = False):
        self.integration_time = torch.linspace(0, T, time_points).float()
        self.integration_time = self.integration_time.type_as(x)
        
        if self.adjoint_flag:
            out = torchdiffeq.odeint_adjoint(self.odefunc, x, self.integration_time,
                                            rtol=self.rtol, atol=self.atol, method=self.method)
        else:
            out = torchdiffeq.odeint(self.odefunc, x, self.integration_time,
                                    rtol=self.rtol, atol=self.atol, method=self.method)
        
        if return_trajectory:
            return out
        else:
            return out[-1]

    def forward_batched(self, x: torch.Tensor, nn: int, indices: list, timestamps: set):
        """Modified forward for ODE batches with different integration times."""
        timestamps = torch.Tensor(list(timestamps))
        if self.adjoint_flag:
            out = torchdiffeq.odeint_adjoint(self.odefunc, x, timestamps,
                                                rtol=self.rtol, atol=self.atol, method=self.method)
        else:
            out = torchdiffeq.odeint(self.odefunc, x, timestamps,
                                        rtol=self.rtol, atol=self.atol, method=self.method)

        out = self._build_batch(out, nn, indices).reshape(x.shape)
        return out

    def _build_batch(self, odeout, nn, indices):
        b_out = []
        for i in range(len(indices)):
            b_out.append(odeout[indices[i], i*nn:(i+1)*nn])
        return torch.cat(b_out).to(odeout.device)
                
    def trajectory(self, x: torch.Tensor, T: int, num_points: int):
        self.integration_time = torch.linspace(0, T, num_points)
        self.integration_time = self.integration_time.type_as(x)
        out = torchdiffeq.odeint(self.odefunc, x, self.integration_time,
                                    rtol=self.rtol, atol=self.atol, method=self.method)
        return out

""" Adapted from torchgde models, with minor original work for
    specific fulfillment of our needs.
    Link to torchgde GitHub: https://github.com/Zymrael/gde/
    Date of Retrieval: September 2024
"""
class GCNLayer(nn.Module):
    """GCNLayer carries out the computation of a single layer GCN.

    Args:
        g (DGLGraph): Underlying graph
        in_feats (int): Input features in graph
        out_feats (int): Output features in graph
        activation (nn.Module): Activation function, if any
        dropout (float): Dropout parameter, if any
        K (int): Value of K to define neighborhood. Default 1.
        bias (bool): Whether to bias outputs. Default True.
    """
    def __init__(self, g: dgl.DGLGraph, in_feats: int, out_feats: int, activation: nn.Module, dropout: float, K: int = 1, bias: bool = True, use_scaled_adj: bool = True):
        super().__init__()
        self.g = g
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats, K))
        self.K = K
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout) if dropout else nn.Identity()
        self.reset_parameters()
        self.use_scaled_adj = use_scaled_adj

    def reset_parameters(self):
        stdv = 1. / (self.weight.size(1) ** 0.5)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Forward computation of the GCN Layer.

        Args:
            h (tensor.Tensor): Input data on graph.
            t (torch.Tensor): Currently unused. Passed from solver.

        Returns:
            torch.Tensor: GCN single layer output
        """
        g = self.g
        with g.local_scope():
            if isinstance(self.dropout, nn.Dropout):
                h = self.dropout(h)
            n = h.shape[0]
            h_out = torch.zeros(n, self.out_feats, device=h.device)
            scale = 1.0 / n

            use_edge_weights = 'weight' in g.edata

            h_all = [h]
            for k in range(1, self.K):
                g.ndata['h'] = h_all[-1]
                if use_edge_weights:
                    g.update_all(
                        dgl.function.u_mul_e('h', 'weight', 'm'),
                        dgl.function.sum('m', 'h')
                    )
                else:
                    g.update_all(
                        dgl.function.copy_u('h', 'm'),
                        dgl.function.sum('m', 'h')
                    )
                h = scale * g.ndata.pop('h')
                h_all.append(h)

            h_all_stacked = torch.stack(h_all)
            weights_arranged = self.weight.permute(1, 0, 2)
            h_out = torch.einsum('kni,oik->no', h_all_stacked, weights_arranged)
            
            if self.bias is not None:
                h_out = h_out + self.bias
            if self.activation:
                h_out = self.activation(h_out)
            return h_out
        
class ThreeLayerGCN(nn.Module):
    """Combines three GCNLayers into a ThreeLayerGCN, used for convenience in experiments.

    Args:
        g (DGLGraph): Underlying graph
        input_dim (int): Input features in graph
        hidden_dim (int): Common dimension of all hidden layers
        output_dim (int): Output features in graph
        activation (nn.Module): Activation function, if any
        K (int): Value of K to define neighborhood
        dropout (float): Dropout value. Default 0.
    """
    def __init__(self, g: dgl.DGLGraph, input_dim: int, hidden_dim: int, output_dim: int, activation: nn.Module, K: int, dropout: float = 0):
        super(ThreeLayerGCN, self).__init__()
        self.fc1 = GCNLayer(g, input_dim, hidden_dim, activation, dropout=dropout, K=K, bias=True)
        self.mid1 = GCNLayer(g, hidden_dim, hidden_dim, activation, dropout=dropout, K=K, bias=True)
        self.mid2 = GCNLayer(g, hidden_dim, hidden_dim, activation, dropout=dropout, K=K, bias=True)
        self.fc2 = GCNLayer(g, hidden_dim, output_dim, activation=None, dropout=dropout, K=K, bias=True)

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Performs forward pass through all GCN layers.
        
        Args:
            t (torch.Tensor): Currently unused, as ODEs are autonomous
            h (torch.Tensor): Input data on graph

        Returns:
            torch.Tensor: Two layer model output
        """
        x = self.fc1(h)
        x = self.mid1(x)
        x = self.mid2(x)
        return self.fc2(x)
    
class TwoLayerGCN(nn.Module):
    """Combines two GCNLayers into a TwoLayerGCN, used in experiments. (Legacy code for compatibility with convergence_experiment.)

    Args:
        g (DGLGraph): Underlying graph
        input_dim (int): Input features in graph
        hidden_dim (int): Output dim in layer 1 = input dim in layer 2
        output_dim (int): Output features in graph
        activation (nn.Module): Activation function, if any
        K (int): Value of K to define neighborhood
        dropout (float): Dropout value. Default 0.
    """
    def __init__(self, g: dgl.DGLGraph, input_dim: int, hidden_dim: int, output_dim: int, activation: nn.Module, K: int, dropout: float = 0):
        super(TwoLayerGCN, self).__init__()
        self.fc1 = GCNLayer(g, input_dim, hidden_dim, activation, dropout=dropout, K=K, bias=True)
        self.fc2 = GCNLayer(g, hidden_dim, output_dim, activation=None, dropout=dropout, K=K, bias=True)

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Performs forward pass through both GCN layers.
        
        Args:
            t (torch.Tensor): Currently unused, as ODEs are autonomous
            h (torch.Tensor): Input data on graph

        Returns:
            torch.Tensor: Two layer model output
        """
        x = self.fc1(h)
        return self.fc2(x)