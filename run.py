import e3nn
import torch
import os

from e3nn import o3
from torch.fx.experimental.proxy_tensor import make_fx
e3nn.set_optimization_defaults(jit_script_fx=False)


device = "cuda" if torch.cuda.is_available() else "cpu"

import torch
import torch.nn as nn
from torch_scatter import scatter_sum
from e3nn import o3

class ScatterLinear(nn.Module):
    def __init__(self, input_dim=160, num_nodes=50):
        super(ScatterLinear, self).__init__()
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        
        self.linear = o3.Linear(irreps_in="1x0e", irreps_out="32x0e+8x1o+8x2e")

    def forward(self, node_features, receivers):
        # Ensure node_features has shape (num_nodes, input_dim)
        assert node_features.shape == (self.num_nodes, self.input_dim), f"Expected node_features shape ({self.num_nodes}, {self.input_dim}), got {node_features.shape}"
        
        # Ensure receivers has shape (num_nodes,) or (num_nodes, 1)
        if receivers.dim() == 1:
            receivers = receivers.unsqueeze(1)
        assert receivers.shape[0] == self.num_nodes, f"Expected receivers shape ({num_nodes}, 1) or ({num_nodes},), got {receivers.shape}"

        # Expand receivers to match the shape of node_features
        receivers_expanded = receivers.expand(-1, self.input_dim)

        # Apply scatter_mean
        shortcut_aggregated = scatter_sum(
            src=node_features,
            index=receivers_expanded,
            dim=0,
            dim_size=32
        )

        return shortcut_aggregated
        shortcut = self.linear(shortcut_aggregated)
        
        return shortcut
# Example usage
input_dim = 1
num_nodes = 50

# Create random input data
node_features = torch.randn(num_nodes, input_dim).to(device=device)
receivers = torch.randint(0, 32, (num_nodes,)).to(device=device)

# Instantiate the module
scatter_module = ScatterLinear(input_dim, num_nodes).to(device=device)

# Forward pass
output = scatter_module(node_features, receivers)

print(f"Input node_features: {node_features.shape} {node_features.dtype}")
print(f"Input receivers: {receivers.shape} {receivers.dtype}")
print(f"Output shape: {output.shape} {output.dtype}")

so_path = torch._export.aot_compile(
        scatter_module,
        args = (node_features, receivers),
        options={"aot_inductor.output_path": os.path.join(os.getcwd(), "export/scatter_works/model.so"),
    })

model_export = torch._export.aot_load(os.path.join(os.getcwd(), f"export/scatter_works/model.so"), device=device)

model_export(node_features, receivers)

