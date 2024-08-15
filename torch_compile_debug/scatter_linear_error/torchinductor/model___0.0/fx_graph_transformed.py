class <lambda>(torch.nn.Module):
    def forward(self):
        arg1_1: "f32[50, 1]"; arg2_1: "i64[50]"; 
    
        arg1_1, arg2_1, = fx_pytree.tree_flatten_spec([], self._in_spec)
        # No stacktrace found for following nodes
        l__self___linear_weight: "f32[32]" = self.L__self___linear_weight
        
        # File: /home/mkotak/.local/lib/python3.10/site-packages/torch_scatter/scatter.py:20 in scatter_sum, code: out = torch.zeros(size, dtype=src.dtype, device=src.device)
        full_default: "f32[32, 1]" = torch.ops.aten.full.default([32, 1], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # File: /home/mkotak/atomic_architects/projects/export_debug/run.py:31 in forward, code: receivers = receivers.unsqueeze(1)
        unsqueeze: "i64[50, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, 1);  arg2_1 = None
        
        # File: /home/mkotak/atomic_architects/projects/export_debug/run.py:35 in forward, code: receivers_expanded = receivers.expand(-1, self.input_dim)
        expand: "i64[50, 1]" = torch.ops.aten.expand.default(unsqueeze, [-1, 1]);  unsqueeze = None
        
        # File: /home/mkotak/.local/lib/python3.10/site-packages/torch_scatter/utils.py:12 in broadcast, code: src = src.expand(other.size())
        expand_1: "i64[50, 1]" = torch.ops.aten.expand.default(expand, [50, 1]);  expand = None
        
        # File: /home/mkotak/.local/lib/python3.10/site-packages/torch_scatter/scatter.py:21 in scatter_sum, code: return out.scatter_add_(dim, index, src)
        scatter_add: "f32[32, 1]" = torch.ops.aten.scatter_add.default(full_default, 0, expand_1, arg1_1);  full_default = expand_1 = arg1_1 = None
        
        # File: <eval_with_key>.5:14 in forward, code: tensordot = torch.functional.tensordot(reshape_2, reshape_3, dims = ((1,), (0,)), out = None);  reshape_2 = reshape_3 = None
        view_4: "f32[32, 1]" = torch.ops.aten.reshape.default(scatter_add, [-1, 1]);  scatter_add = None
        view_5: "f32[32, 1, 1]" = torch.ops.aten.reshape.default(view_4, [32, 1, 1]);  view_4 = None
        sum_1: "f32[32, 1, 1]" = torch.ops.aten.sum.dim_IntList(view_5, [1], True, dtype = torch.float32);  view_5 = None
        permute: "f32[32, 1, 1]" = torch.ops.aten.permute.default(sum_1, [0, 2, 1]);  sum_1 = None
        view_6: "f32[32, 1]" = torch.ops.aten.reshape.default(permute, [32, 1]);  permute = None
        
        # File: <eval_with_key>.5:11 in forward, code: reshape_1 = w.reshape(-1, 32);  w = None
        view_1: "f32[1, 32]" = torch.ops.aten.reshape.default(l__self___linear_weight, [-1, 32]);  l__self___linear_weight = None
        
        # File: <eval_with_key>.5:14 in forward, code: tensordot = torch.functional.tensordot(reshape_2, reshape_3, dims = ((1,), (0,)), out = None);  reshape_2 = reshape_3 = None
        permute_1: "f32[1, 32]" = torch.ops.aten.permute.default(view_1, [0, 1]);  view_1 = None
        mm: "f32[32, 32]" = torch.ops.aten.mm.default(view_6, permute_1);  view_6 = permute_1 = None
        view_8: "f32[32, 1, 32]" = torch.ops.aten.reshape.default(mm, [32, 1, 32]);  mm = None
        
        # File: <eval_with_key>.5:15 in forward, code: permute = tensordot.permute(0, 2, 1);  tensordot = None
        permute_2: "f32[32, 32, 1]" = torch.ops.aten.permute.default(view_8, [0, 2, 1]);  view_8 = None
        
        # File: <eval_with_key>.5:16 in forward, code: reshape_4 = permute.reshape(getitem_1, 32);  permute = None
        view_9: "f32[32, 32]" = torch.ops.aten.reshape.default(permute_2, [32, 32]);  permute_2 = None
        
        # File: <eval_with_key>.5:17 in forward, code: new_zeros = reshape.new_zeros((getitem_1, 24))
        full_default_1: "f32[32, 24]" = torch.ops.aten.full.default([32, 24], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # File: <eval_with_key>.5:18 in forward, code: new_zeros_1 = reshape.new_zeros((getitem_1, 40));  reshape = getitem_1 = None
        full_default_2: "f32[32, 40]" = torch.ops.aten.full.default([32, 40], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # File: <eval_with_key>.5:19 in forward, code: cat = torch.cat([reshape_4, new_zeros, new_zeros_1], dim = -1);  reshape_4 = new_zeros = new_zeros_1 = None
        cat: "f32[32, 96]" = torch.ops.aten.cat.default([view_9, full_default_1, full_default_2], -1);  view_9 = full_default_1 = full_default_2 = None
        return (cat,)
        