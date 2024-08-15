class <lambda>(torch.nn.Module):
    def forward(self):
        arg0_1: "f32[50, 1]"; arg1_1: "i64[50]"; 
    
        arg0_1, arg1_1, = fx_pytree.tree_flatten_spec([], self._in_spec)
        # File: /home/mkotak/.local/lib/python3.10/site-packages/torch_scatter/scatter.py:20 in scatter_sum, code: out = torch.zeros(size, dtype=src.dtype, device=src.device)
        full_default: "f32[32, 1]" = torch.ops.aten.full.default([32, 1], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # File: /home/mkotak/atomic_architects/projects/export_debug/run.py:31 in forward, code: receivers = receivers.unsqueeze(1)
        unsqueeze: "i64[50, 1]" = torch.ops.aten.unsqueeze.default(arg1_1, 1);  arg1_1 = None
        
        # File: /home/mkotak/atomic_architects/projects/export_debug/run.py:35 in forward, code: receivers_expanded = receivers.expand(-1, self.input_dim)
        expand: "i64[50, 1]" = torch.ops.aten.expand.default(unsqueeze, [-1, 1]);  unsqueeze = None
        
        # File: /home/mkotak/.local/lib/python3.10/site-packages/torch_scatter/utils.py:12 in broadcast, code: src = src.expand(other.size())
        expand_1: "i64[50, 1]" = torch.ops.aten.expand.default(expand, [50, 1]);  expand = None
        
        # File: /home/mkotak/.local/lib/python3.10/site-packages/torch_scatter/scatter.py:21 in scatter_sum, code: return out.scatter_add_(dim, index, src)
        scatter_add: "f32[32, 1]" = torch.ops.aten.scatter_add.default(full_default, 0, expand_1, arg0_1);  full_default = expand_1 = arg0_1 = None
        return (scatter_add,)
        