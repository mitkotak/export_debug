

from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_mkotak/hs/chspp3w5add3wtyabbixla2tsdcgbd5oj7e4r6lo5siuk5t5sx3b.py
# Source Nodes: [out, shortcut_aggregated], Original ATen: [aten.scatter_add, aten.zeros]
# out => full_default
# shortcut_aggregated => scatter_add
triton_poi_fused_scatter_add_zeros_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[32], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_scatter_add_zeros_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '521822B9F3468AF6E86806E85D57CDE414A140644FBC5A96ABF3FC1C38FC8077', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')

import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


# kernel path: /tmp/torchinductor_mkotak/tz/ctzdu4zjdw33zmagfisgt7566p45gvqzzzm4ecftsj3ma5zy6bxl.py
# Source Nodes: [out, shortcut_aggregated], Original ATen: [aten.scatter_add, aten.zeros]
# out => full_default
# shortcut_aggregated => scatter_add
triton_poi_fused_scatter_add_zeros_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[64], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_scatter_add_zeros_1', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '521822B9F3468AF6E86806E85D57CDE414A140644FBC5A96ABF3FC1C38FC8077', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = tl.full([XBLOCK], 32, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 32)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 32")
    tl.atomic_add(out_ptr0 + (tl.broadcast_to(tmp4, [XBLOCK])), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_mkotak/qs/cqs6xkmd5q4mmdbfblmsr7s74zkwccqgozi4xwfflbkfcq3zjbqo.py
# Source Nodes: [tensordot], Original ATen: [aten.sum]
# tensordot => sum_1
triton_poi_fused_sum_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[32], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sum_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521822B9F3468AF6E86806E85D57CDE414A140644FBC5A96ABF3FC1C38FC8077', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_mkotak/ng/cngno3cynplxomgh2jkdgzbk4ykqmlri5n7rqkgfc7jejona2a22.py
# Source Nodes: [cat], Original ATen: [aten.cat]
# cat => cat
triton_poi_fused_cat_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4096], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_3', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521822B9F3468AF6E86806E85D57CDE414A140644FBC5A96ABF3FC1C38FC8077', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 96
    x1 = (xindex // 96)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + (32*x1)), tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 56, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = 0.0
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1], 96, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.where(tmp15, tmp12, tmp13)
    tmp19 = tl.where(tmp11, tmp14, tmp18)
    tmp20 = tl.where(tmp4, tmp7, tmp19)
    tl.store(out_ptr0 + (x2), tmp20, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg1_1, (50, 1), (1, 1))
    assert_size_stride(arg2_1, (50, ), (1, ))

    for kernel in globals().values():
        if isinstance(kernel, torch._inductor.runtime.triton_heuristics.CachingAutotuner):
            kernel.cuda_kernel_saved = False
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 1), (1, 1), torch.float32)
        # Source Nodes: [out, shortcut_aggregated], Original ATen: [aten.scatter_add, aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_scatter_add_zeros_0.run(buf0, 32, grid=grid(32), stream=stream0)
        # Source Nodes: [out, shortcut_aggregated], Original ATen: [aten.scatter_add, aten.zeros]
        triton_poi_fused_scatter_add_zeros_1.run(arg2_1, arg1_1, buf0, 50, grid=grid(50), stream=stream0)
        del arg1_1
        del arg2_1
        buf2 = empty_strided_cuda((32, 1, 1), (1, 1, 1), torch.float32)
        # Source Nodes: [tensordot], Original ATen: [aten.sum]
        triton_poi_fused_sum_2.run(buf0, buf2, 32, grid=grid(32), stream=stream0)
        del buf0
        buf3 = empty_strided_cuda((32, 32), (32, 1), torch.float32)
        # Source Nodes: [tensordot], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf2, (32, 1), (1, 0), 0), reinterpret_tensor(L__self___linear_weight, (1, 32), (32, 1), 0), out=buf3)
        del buf2
        buf4 = empty_strided_cuda((32, 96), (96, 1), torch.float32)
        # Source Nodes: [cat], Original ATen: [aten.cat]
        triton_poi_fused_cat_3.run(buf3, buf4, 3072, grid=grid(3072), stream=stream0)
        del buf3

    for kernel in globals().values():
        if isinstance(kernel, torch._inductor.runtime.triton_heuristics.CachingAutotuner):
            if not kernel.cuda_kernel_saved:
                if len(kernel.launchers) == 0:
                    kernel.precompile()
                kernel.save_cuda_kernel(
                    grid=(0, 0, 0),   # use dummy grid
                    stream="stream",  # use dummy stream
                    launcher=kernel.launchers[0],
                )
    return (buf4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    global L__self___linear_weight
    L__self___linear_weight = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((50, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((50, ), (1, ), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
