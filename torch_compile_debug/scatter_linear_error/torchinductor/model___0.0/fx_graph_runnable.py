
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.cpp_wrapper = True
torch._inductor.config.triton.autotune_cublasLt = False
torch._inductor.config.triton.store_cubin = True
torch._inductor.config.aot_inductor.output_path = '/home/mkotak/atomic_architects/projects/export_debug/export/model.so'
torch._inductor.config.aot_inductor.serialized_in_spec = '[1, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.dict", "context": "[]", "children_spec": []}]}]'
torch._inductor.config.aot_inductor.serialized_out_spec = '[1, {"type": null, "context": null, "children_spec": []}]'




isolate_fails_code_str = None



# torch version: 2.4.0+cu121
# torch cuda version: 12.1
# torch git version: e4ee3be4063b7c430974252fdf7db42273388d86


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2024 NVIDIA Corporation 
# Built on Wed_Apr_17_19:19:55_PDT_2024 
# Cuda compilation tools, release 12.5, V12.5.40 
# Build cuda_12.5.r12.5/compiler.34177558_0 

# GPU Hardware Info: 
# NVIDIA RTX A5500 : 3 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.L__self___linear_weight = torch.nn.Parameter(torch.randn([32], dtype=torch.float32, device="cuda"))

    
    
    def forward(self):
        arg1_1, arg2_1, = fx_pytree.tree_flatten_spec([], self._in_spec)
        l__self___linear_weight = self.L__self___linear_weight
        unsqueeze = torch.ops.aten.unsqueeze.default(arg2_1, 1);  arg2_1 = None
        expand = torch.ops.aten.expand.default(unsqueeze, [-1, 1]);  unsqueeze = None
        expand_1 = torch.ops.aten.expand.default(expand, [50, 1]);  expand = None
        full_default = torch.ops.aten.full.default([32, 1], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scatter_add = torch.ops.aten.scatter_add.default(full_default, 0, expand_1, arg1_1);  full_default = expand_1 = arg1_1 = None
        view_1 = torch.ops.aten.view.default(l__self___linear_weight, [-1, 32]);  l__self___linear_weight = None
        view_4 = torch.ops.aten.view.default(scatter_add, [-1, 1]);  scatter_add = None
        view_5 = torch.ops.aten.view.default(view_4, [32, 1, 1]);  view_4 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(view_5, [1], True, dtype = torch.float32);  view_5 = None
        permute = torch.ops.aten.permute.default(sum_1, [0, 2, 1]);  sum_1 = None
        view_6 = torch.ops.aten.view.default(permute, [32, 1]);  permute = None
        permute_1 = torch.ops.aten.permute.default(view_1, [0, 1]);  view_1 = None
        mm = torch.ops.aten.mm.default(view_6, permute_1);  view_6 = permute_1 = None
        view_8 = torch.ops.aten.view.default(mm, [32, 1, 32]);  mm = None
        permute_2 = torch.ops.aten.permute.default(view_8, [0, 2, 1]);  view_8 = None
        view_9 = torch.ops.aten.view.default(permute_2, [32, 32]);  permute_2 = None
        full_default_1 = torch.ops.aten.full.default([32, 24], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2 = torch.ops.aten.full.default([32, 40], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat = torch.ops.aten.cat.default([view_9, full_default_1, full_default_2], -1);  view_9 = full_default_1 = full_default_2 = None
        return (cat,)
        
def load_args(reader):
    buf0 = reader.storage(None, 200, device=device(type='cuda', index=0))
    reader.tensor(buf0, (50, 1), is_leaf=True)  # arg1_1
    buf1 = reader.storage(None, 400, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf1, (50,), dtype=torch.int64, is_leaf=True)  # arg2_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)