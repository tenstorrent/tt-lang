# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import types
from typing import Optional, Any, Callable, Union, Tuple, Dict, Protocol, Coroutine
import copy
import torch
import math
from types import CellType, FunctionType
# from pykernel.kernel_ast import *
# from utils import assert_pcc
from sim import TILE_SIZE, TensorAccessor, IndexType, CircularBuffer, dma, CBAPI, torch_utils as tu

MAX_CORES = 4  # assuming a 2x2 core grid for simplicity

# Protocol for templates that have a bind method
class BindableTemplate(Protocol):
    __name__: str
    def bind(self, ctx: Dict[str, Any]) -> Callable[[], Coroutine[Any, Any, Any]]: ...

# TODO: Preamble work should either merge with tt-lang or sim
def pykernel_gen(grid: Union[str, Tuple[int, int]] = 'auto', granularity: int = 4):
    """
    Decorator that generates a kernel with specified grid and granularity.
    If grid='auto', defaults to (2, 2).
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
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
        setattr(wrapper, '__pykernel_config__', {'grid': grid, 'granularity': granularity})
        setattr(wrapper, 'granularity', granularity)  # Make granularity accessible
        return wrapper
    return decorator

def _make_cell(value: Any) -> CellType:
    # create a real closure cell holding `value`
    def inner() -> Any:
        return value
    assert inner.__closure__ is not None
    return inner.__closure__[0]

def _rebind_func_with_ctx(func: FunctionType, ctx: Dict[str, Any]) -> FunctionType:
    """
    Create a new function from `func` but with:
      - globals = func.__globals__ + ctx
      - closure cells rebuilt from ctx when possible
    so that names like `out_cb` that were captured will now point to the per-core objects.
    """
    freevars = func.__code__.co_freevars
    orig_closure = func.__closure__ or ()
    orig_cell_map: Dict[str, CellType] = {name: cell for name, cell in zip(freevars, orig_closure)}

    new_cells: list[CellType] = []
    for name in freevars:
        if name in ctx:
            new_cells.append(_make_cell(ctx[name]))
        else:
            # fall back to original cell if we don't have an override
            new_cells.append(orig_cell_map[name])

    # merge globals with ctx so globals-based lookups also see per-core state
    new_globals: Dict[str, Any] = dict(func.__globals__)
    new_globals.update(ctx)

    new_func = types.FunctionType(
        func.__code__,
        new_globals,
        func.__name__,
        func.__defaults__,
        tuple(new_cells)
    )
    return new_func

def compute() -> Callable[[FunctionType], BindableTemplate]:
    def decorator(func: FunctionType) -> BindableTemplate:
        class ComputeTemplate:
            __name__ = func.__name__

            def bind(self, ctx: Dict[str, Any]) -> Callable[[], Coroutine[Any, Any, Any]]:
                # rebuild function with per-core closure
                bound_func = _rebind_func_with_ctx(func, ctx)

                async def runner():
                    res = bound_func()
                    import inspect
                    if inspect.iscoroutine(res):
                        await res
                    else:
                        return res
                return runner

        return ComputeTemplate()
    return decorator

def datamovement() -> Callable[[FunctionType], BindableTemplate]:
    def decorator(func: FunctionType) -> BindableTemplate:
        class DMTemplate:
            __name__ = func.__name__

            def bind(self, ctx: Dict[str, Any]) -> Callable[[], Coroutine[Any, Any, Any]]:
                bound_func = _rebind_func_with_ctx(func, ctx)

                async def runner():
                    res = bound_func()
                    import inspect
                    if inspect.iscoroutine(res):
                        await res
                    else:
                        return res
                return runner

        return DMTemplate()
    return decorator

# Global state for core management
_core_state = {'current_core': 0}

def core_index() -> int:
    """Returns the core index in a 2D grid (stays consistent during one program run)"""
    return _core_state['current_core']

def next_core():
    """Advance to the next core for the next program run"""
    _core_state['current_core'] = (_core_state['current_core'] + 1) % MAX_CORES

def Program(*funcs: BindableTemplate) -> Any:
    """Program class that combines compute and data movement functions"""
    class ProgramImpl:
        def __init__(self, *functions: BindableTemplate):
            self.functions = functions
            self.context: Dict[str, Any] = {}
        
        def __call__(self, *args: Any, **kwargs: Any) -> None:
            import inspect
            frame = inspect.currentframe()
            if frame and frame.f_back:
                # capture locals from the caller (eltwise_add)
                self.context = dict(frame.f_back.f_locals)

            grid = self.context.get('grid', (1, 1))
            total_cores = grid[0] * grid[1]

            compute_func_tmpl, dm0_tmpl, dm1_tmpl = self.functions

            # collect user-facing errors here
            errors: list[str] = []

            for core in range(total_cores):
                # build per-core context
                memo: dict[int, Any] = {}
                core_context: dict[str, Any] = {}
                api = CBAPI[torch.Tensor]()  # new CBAPI per core

                for key, value in self.context.items():
                    if isinstance(value, (torch.Tensor, TensorAccessor)):
                        core_context[key] = value
                        memo[id(value)] = value
                    elif isinstance(value, CircularBuffer):
                        # create a fresh CB for this core
                        core_context[key] = CircularBuffer(
                            accessor=value.accessor,
                            shape=value.shape,
                            buffer_factor=value.buffer_factor,
                            api=api
                        )
                    else:
                        core_context[key] = copy.deepcopy(value, memo)

                # also make the core number visible
                core_context['core'] = core

                # bind per-core
                core_dm0 = dm0_tmpl.bind(core_context)
                core_compute = compute_func_tmpl.bind(core_context)
                core_dm1 = dm1_tmpl.bind(core_context)

                # run the three in parallel threads, because CB ops are blocking
                import threading, asyncio, traceback

                # we store (stage_name, exception, traceback_str)
                thread_results: list[Tuple[str, Exception, str]] = []
                
                def run_coro_in_thread(name: str, coro_factory: Callable[[], Coroutine[Any, Any, Any]]) -> None:
                    try:
                        coro = coro_factory()
                        try:
                            # normal path
                            asyncio.run(coro)
                        except RuntimeError as re:
                            # only fallback if it's the "event loop is running" case
                            msg = str(re)
                            if "event loop is running" in msg:
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                # create a *new* coroutine for this loop
                                coro2 = coro_factory()
                                loop.run_until_complete(coro2)
                                loop.close()
                            else:
                                # it's some other runtime error; re-raise to outer except
                                raise
                    except Exception as e:
                        tb_str = traceback.format_exc()
                        thread_results.append((name, e, tb_str))

                t_dm0 = threading.Thread(target=run_coro_in_thread, args=(f"core{core}-dm0", core_dm0))
                t_comp = threading.Thread(target=run_coro_in_thread, args=(f"core{core}-compute", core_compute))
                t_dm1 = threading.Thread(target=run_coro_in_thread, args=(f"core{core}-dm1", core_dm1))

                # start all three
                t_dm0.start()
                t_comp.start()
                t_dm1.start()

                # wait for all to finish
                t_dm0.join()
                t_comp.join()
                t_dm1.join()

                # check if any failed
                if thread_results:
                    for name, e, tb_str in thread_results:
                        # print a user-readable header
                        print(f"\n❌ {name} failed on core {core}")
                        print(f"   error type   : {type(e).__name__}")
                        print(f"   error message: {e}")
                        print("   traceback:")
                        print(tb_str)
                        print("-" * 50)

                        # add to final aggregation (short)
                        errors.append(f"{name} on core {core}: {type(e).__name__}: {e}")

                if core < total_cores - 1:
                    next_core()

            if errors:
                raise RuntimeError("One or more cores failed:\n" + "\n".join(errors))

            return None

    return ProgramImpl(*funcs)

def assert_pcc(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> None:
    # tensors should be equal
    assert tensor_a.shape == tensor_b.shape, "Tensors must have the same shape"
    assert tensor_a.dtype == tensor_b.dtype, "Tensors must have the same dtype"
    assert tensor_a.device == tensor_b.device, "Tensors must be on the same device"
    assert tu.allclose(tensor_a, tensor_b), "Tensors values are not close enough"

def is_tiled(tensor: torch.Tensor) -> bool:
    return tensor.shape[0] % TILE_SIZE == 0 and tensor.shape[1] % TILE_SIZE == 0

@pykernel_gen(
    grid='auto', # NOTE: allow compiler to choose grid
    granularity=2, # compute granularity. could be passed by user, or left for auto-tuning
)
def eltwise_add(a_in: torch.Tensor, b_in: torch.Tensor, out: torch.Tensor, grid: Optional[Any] = None) -> None:
    assert grid is not None
    
    # Get granularity from decorator (hardcoded for now since decorator system is simplified)
    granularity = 2
    
    # Assuming lightweight op input validation should be here
    assert a_in.shape == b_in.shape == out.shape
    assert all(is_tiled(tensor) for tensor in [a_in, b_in, out])
    assert a_in.shape[0] % granularity == 0

    row_tiles = a_in.shape[0] // TILE_SIZE
    col_tiles = a_in.shape[1] // TILE_SIZE
    
    # Parallelizing by columns here to get reuse on C
    cols_per_core = math.ceil(col_tiles / (grid[0] * grid[1]))
    buffer_factor = 2

    a_accessor = TensorAccessor(a_in, index_type=IndexType.TILE)
    b_accessor = TensorAccessor(b_in, index_type=IndexType.TILE)
    out_accessor = TensorAccessor(out, index_type=IndexType.TILE)
    
    # NOTE: (Kostas) I don't understand why a CircularBuffer needs to be associated with a tensor accessor.
    #                Perhaps we need to know its specific type? Or to prevent mixups of tensors on the same cb?
    a_in_cb = CircularBuffer(a_accessor, shape=(granularity,1), buffer_factor=buffer_factor)
    b_in_cb = CircularBuffer(b_accessor, shape=(granularity,1), buffer_factor=buffer_factor)
    out_cb = CircularBuffer(out_accessor, shape=(granularity,1), buffer_factor=buffer_factor)

    @compute()
    async def compute_func():
        core_num = core_index() # core number in 2d grid
        start_col_tile = core_num * cols_per_core
        end_col_tile = min(start_col_tile + cols_per_core, col_tiles)
        
        for ct in range(start_col_tile, end_col_tile):
            # Reuse C across rows of A, B
            # returns a RingView object as defined in <put up the link>.
            # TODO: Perhaps consider making RingView pointers that come from wait()/reserve() read/write only respectively?
            for rt_block in range(row_tiles // granularity):
                print("Compute: ", f"core={core_num}", f"column={ct}", f"block={rt_block}")
                # again, these return RingView pointers:
                a_block = a_in_cb.wait() # blocking 
                b_block = b_in_cb.wait() # blocking
                # NOTE: Please consider making non-approx the default for eltwise unary, but leave the option for the user to specify approx=True
                out_block = out_cb.reserve() # blocking
                
                # Use store() to properly populate the RingView with computed results
                out_block.store([a_block[i] + b_block[i] for i in range(len(a_block))])
                
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
                print("dm0: ", f"core={core_num}", f"column={ct}", f"block={rt_block}")
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
                print("dm1: ", f"core={core_num}", f"column={ct}", f"block={rt_block}")
                row_slice = slice(rt_block*granularity, (rt_block+1)*granularity)
                col_slice = slice(ct, ct+1)
                
                out_block = out_cb.wait()
                # out_block[100] # accessing out of bounds should fail
                
                tx = dma(out_block, out_accessor[row_slice, col_slice])
                tx.wait()
                out_cb.pop()
                # TODO: We might want better error messages, most of them come from the underlying CBAPI
                #       which might be confusing to the higher level CircularBuffer user.
                # TODO: What if another thread writes to the same positions this RingView points to?
                # out_block[0] # using pointer on stale data should fail
                # out_cb.pop() # double pop should fail

    # Execute the program across all cores
    return Program(compute_func, dm0, dm1)(a_in, b_in, out)

"""
out = a + b
"""
dim = 256
a_in = torch.randn(dim, dim) #type: ignore
b_in = torch.randn(dim, dim) #type: ignore
out = torch.zeros(dim, dim) #type: ignore
eltwise_add(a_in, b_in, out)

golden = a_in + b_in
print(golden)
print(out)
assert_pcc(golden, out)
