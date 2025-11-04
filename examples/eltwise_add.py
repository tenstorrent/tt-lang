# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Any, Callable, Union, Tuple
import sys
sys.path.append('../python')

import torch
import math
# from pykernel.kernel_ast import *
# from utils import assert_pcc
from sim import TILE_SIZE, TensorAccessor, IndexType, CircularBuffer, dma

def pykernel_gen(grid: Union[str, Tuple[int, int]] = 'auto', granularity: int = 4):
    """
    Decorator that generates a kernel with specified grid and granularity.
    If grid='auto', defaults to (2, 2).
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Set grid to (2, 2) if 'auto'
            actual_grid = (2, 2) if grid == 'auto' else grid
            
            # Inject granularity into the function's local scope
            import inspect
            frame = inspect.currentframe()
            if frame and frame.f_back:
                frame.f_back.f_locals['granularity'] = granularity
            
            # Call the original function with the processed grid
            return func(*args, grid=actual_grid, **kwargs)
        
        # Store the decorator parameters for later access
        wrapper.__pykernel_config__ = {'grid': grid, 'granularity': granularity}
        wrapper.granularity = granularity  # Make granularity accessible
        return wrapper
    return decorator

def compute():
    """Decorator for compute functions"""
    def decorator(func: Callable) -> Callable:
        return func
    return decorator

def datamovement():
    """Decorator for data movement functions"""
    def decorator(func: Callable) -> Callable:
        return func
    return decorator

def core_index() -> int:
    """Returns the core index in a 2D grid (stays consistent during one program run)"""
    if not hasattr(core_index, 'current_core'):
        core_index.current_core = 0
    return core_index.current_core

def next_core():
    """Advance to the next core for the next program run"""
    if not hasattr(core_index, 'current_core'):
        core_index.current_core = 0
    core_index.current_core = (core_index.current_core + 1) % 4

def Program(*funcs):
    """Program class that combines compute and data movement functions"""
    class ProgramImpl:
        def __init__(self, *functions):
            self.functions = functions
            self.context = {}  # Will be populated when __call__ is invoked
        
        def __call__(self, *args, **kwargs):
            # Capture ALL local variables from the calling function (eltwise_add) automatically
            import inspect
            frame = inspect.currentframe().f_back
            if frame:
                # Get all local variables from the calling function
                self.context = dict(frame.f_locals)
                
                # Also add the function arguments for convenience
                if len(args) > 0:
                    self.context['_args'] = args
                if kwargs:
                    self.context['_kwargs'] = kwargs
            
            # Get grid info for multi-core execution
            grid = self.context.get('grid', (1, 1))
            total_cores = grid[0] * grid[1]
            
            # Execute the functions in proper order: dm0 -> compute_func -> dm1
            compute_func, dm0, dm1 = self.functions
            
            # Helper function to execute a function with context available
            def execute_with_context(func, func_name):
                # Make context available to the function by injecting into its globals
                if hasattr(func, '__globals__'):
                    # Inject context variables into the function's globals
                    # NO restoration - allow functions to modify shared state
                    for key, value in self.context.items():
                        func.__globals__[key] = value
                    
                    try:
                        # Execute the function (handling async functions)
                        result = func()
                        
                        # Check if it's a coroutine (async function)
                        import inspect
                        import asyncio
                        if inspect.iscoroutine(result):
                            # Run the async function using asyncio
                            asyncio.run(result)
                        return result
                    except Exception as e:
                        print(f"✗ {func_name} failed with error: {e}")
                        import traceback
                        traceback.print_exc()
                    # NO cleanup - preserve state changes between function calls
                else:
                    try:
                        result = func()
                        return result
                    except Exception as e:
                        print(f"✗ {func_name} failed with error: {e}")
            
            # Execute the program for each core
            for core in range(total_cores):
                # Execute the functions in the correct order with context available
                execute_with_context(dm0, "dm0 (data movement in)")
                execute_with_context(compute_func, "compute_func (computation)")
                execute_with_context(dm1, "dm1 (data movement out)")
                
                # Advance to next core (except for the last core)
                if core < total_cores - 1:
                    next_core()
            
            return None
    
    return ProgramImpl(*funcs)

def assert_pcc(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> None:
    # tensors should be equal
    assert tensor_a.shape == tensor_b.shape, "Tensors must have the same shape"
    assert tensor_a.dtype == tensor_b.dtype, "Tensors must have the same dtype"
    assert tensor_a.device == tensor_b.device, "Tensors must be on the same device"
    assert torch.allclose(tensor_a, tensor_b), "Tensors values are not close enough"

def is_tiled(tensor: torch.Tensor) -> bool:
    return tensor.shape[0] % TILE_SIZE == 0 and tensor.shape[1] % TILE_SIZE == 0

@pykernel_gen(
    grid='auto', # NOTE: allow compiler to choose grid
    granularity=4, # compute granularity. could be passed by user, or left for auto-tuning
)
def eltwise_add(a_in: torch.Tensor, b_in: torch.Tensor, out: torch.Tensor, grid: Optional[Any] = None):
    assert grid is not None
    
    # Get granularity from decorator (hardcoded for now since decorator system is simplified)
    granularity = 4
    
    # Assuming lightweight op input validation should be here
    assert a_in.shape == b_in.shape == out.shape
    assert all(is_tiled(tensor) for tensor in [a_in, b_in, out])
    assert a_in.shape[0] % granularity == 0

    row_tiles = a_in.shape[0] // TILE_SIZE
    col_tiles = a_in.shape[1] // TILE_SIZE
    
    # Parallelizing by columns here to get reuse on C
    cols_per_core = math.ceil(col_tiles / (grid[0] * grid[1]))

    a_accessor = TensorAccessor(a_in, index_type=IndexType.TILE)
    b_accessor = TensorAccessor(b_in, index_type=IndexType.TILE)
    out_accessor = TensorAccessor(out, index_type=IndexType.TILE)
    
    # NOTE: (Kostas) I don't understand why a CircularBuffer needs to be associated with a tensor accessor.
    #                Perhaps we need to know its specific type? Or to prevent mixups of tensors on the same cb?
    a_in_cb = CircularBuffer(a_accessor, shape=(granularity,1), buffer_factor=2)
    b_in_cb = CircularBuffer(b_accessor, shape=(granularity,1), buffer_factor=2)
    out_cb = CircularBuffer(out_accessor, shape=(granularity,1), buffer_factor=2)

    @compute()
    async def compute_func():
        core_num = core_index() # core number in 2d grid
        start_col_tile = core_num * cols_per_core
        end_col_tile = min(start_col_tile + cols_per_core, col_tiles)
        
        for _ in range(start_col_tile, end_col_tile):
            # Reuse C across rows of A, B
            # returns a RingView object as defined in <put up the link>.
            # TODO: Perhaps consider making RingView pointers that come from wait()/reserve() read/write only respectively?
            for _ in range(row_tiles // granularity):
                # again, these return RingView pointers:
                a_block = a_in_cb.wait() # blocking 
                b_block = b_in_cb.wait() # blocking
                # NOTE: Please consider making non-approx the default for eltwise unary, but leave the option for the user to specify approx=True
                out_block = out_cb.reserve() # blocking

                
                # Use fill() to properly populate the RingView with computed results
                out_block.fill([a_block[i] + b_block[i] for i in range(len(a_block))])
                
                # finalize push, this advances the cb pointers, the writing happened at the line above
                out_cb.push()
                # finalize pop, this advances the cb pointers, essentially freeing the memory
                # After poping, the corresponding RingView(a_block) points to stale data. Should probably make it an error to access it at that point
                a_in_cb.pop()
                # ditto
                b_in_cb.pop()

    @datamovement()
    async def dm0():
        core_num = core_index() # core number in 2d grid
        start_col_tile = core_num * cols_per_core
        end_col_tile = min(start_col_tile + cols_per_core, col_tiles)
        
        for ct in range(start_col_tile, end_col_tile):
            for rt_block in range(row_tiles // granularity):
                """
                Since the TensorAccessor indexes by tile, slicing is cleaner
                """
                row_slice = slice(rt_block*granularity, (rt_block+1)*granularity)
                col_slice = slice(ct, ct+1)
                # Write the cbs just as above
                a_block = a_in_cb.reserve()
                tx = dma(a_accessor[row_slice, col_slice], a_block)
                tx.wait()
                a_in_cb.push()
                b_block = b_in_cb.reserve()
                tx = dma(b_accessor[row_slice, col_slice], b_block)
                tx.wait()
                b_in_cb.push()

    @datamovement()
    async def dm1():
        core_num = core_index() # core number in 2d grid
        start_col_tile = core_num * cols_per_core
        end_col_tile = min(start_col_tile + cols_per_core, col_tiles)
        
        for ct in range(start_col_tile, end_col_tile):
            for rt_block in range(row_tiles // granularity):
                """
                Since the TensorAccessor indexes by tile, slicing is cleaner
                """
                row_slice = slice(rt_block*granularity, (rt_block+1)*granularity)
                col_slice = slice(ct, ct+1)
                
                out_block = out_cb.wait()
                
                tx = dma(out_block, out_accessor[row_slice, col_slice])
                tx.wait()
                out_cb.pop()
    # Execute the program across all cores
    return Program(compute_func, dm0, dm1)(a_in, b_in, out)

"""
out = a + b
"""

a_in = torch.randn(128, 128)
b_in = torch.randn(128, 128)
out = torch.zeros(128, 128)
eltwise_add(a_in, b_in, out)

golden = a_in + b_in
assert_pcc(golden, out)