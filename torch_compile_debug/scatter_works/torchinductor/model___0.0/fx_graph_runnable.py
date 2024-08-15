
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

    
    
    def forward(self):
        arg0_1, arg1_1, = fx_pytree.tree_flatten_spec([], self._in_spec)
        unsqueeze = torch.ops.aten.unsqueeze.default(arg1_1, 1);  arg1_1 = None
        expand = torch.ops.aten.expand.default(unsqueeze, [-1, 1]);  unsqueeze = None
        expand_1 = torch.ops.aten.expand.default(expand, [50, 1]);  expand = None
        full_default = torch.ops.aten.full.default([32, 1], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scatter_add = torch.ops.aten.scatter_add.default(full_default, 0, expand_1, arg0_1);  full_default = expand_1 = arg0_1 = None
        return (scatter_add,)
        
def load_args(reader):
    buf0 = reader.storage(None, 200, device=device(type='cuda', index=0))
    reader.tensor(buf0, (50, 1), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 400, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf1, (50,), dtype=torch.int64, is_leaf=True)  # arg1_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)