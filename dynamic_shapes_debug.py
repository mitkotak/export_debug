from torch_runstats.scatter import scatter
import torch
from typing import Optional

from torch.fx.experimental.proxy_tensor import make_fx 

def _broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src



def scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> torch.Tensor:
    assert reduce == "sum"  # for now, TODO
    index = _broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


class Scatter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, src, index, pos):
        # Make src require gradients
        src.requires_grad_(True)
        
        # Perform the scatter operation
        result = scatter(
            src=src,
            index=index,
            dim=0,
            dim_size=pos.shape[0]
        )

        # Compute gradients of result with respect to src
        grads = torch.autograd.grad(
            outputs=result.sum(),
            inputs=src,
        )[0]
        
        return result

index = torch.arange(64).repeat_interleave(28).to(torch.int64)
src = torch.randn(index.shape[0], 8, 4)
pos = torch.randn(64, 3)

neighbors = torch.export.Dim("neighbors", min=2, max=torch.inf)
nodes = torch.export.Dim("nodes", min=2, max=torch.inf)

scatter_model = Scatter()
scatter_model = make_fx(scatter_model)(src, index, pos)
so_path = torch._export.aot_compile(
    scatter_model,
    args=(src, index, pos),
    dynamic_shapes=(
                {0: neighbors, 1: None, 2: None}, # src
                {0: neighbors}, # index
                {0: nodes, 1: None}, # pos
            ),
    options={"aot_inductor.output_path": "./scatter.so"}
)

outputs_export = torch._export.aot_load(so_path, device="cpu")(src, index, pos)
assert tuple(outputs_export.shape) == (64, 8, 4)