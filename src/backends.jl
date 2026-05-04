# GPU backend dispatch for TensorBinding.
#
# Zero-cost CPU path: to_device / from_device are no-ops on CPU.
# GPU path loads automatically when `using CUDA` precedes `using TensorBinding`
# (Julia 1.9+ package extension in ext/TensorBindingCUDAExt.jl).
#
# Usage in computational functions:
#   backend = _resolve_backend(run_on)   # run_on ∈ {:cpu, :gpu}
#   H_dev   = to_device(H, backend)     # move MPO/MPS to target device
#   ...                                  # run recurrence on device
#   result  = from_device(x, backend)   # move result back to CPU

abstract type AbstractTBBackend end
struct CPUBackend <: AbstractTBBackend end
struct GPUBackend <: AbstractTBBackend end

to_device(x, ::CPUBackend)   = x
from_device(x, ::CPUBackend) = x

function to_device(_, ::GPUBackend)
    error("TensorBinding GPU backend: CUDA.jl is not loaded.\n" *
          "Add `using CUDA` before `using TensorBinding`, or pass run_on=:cpu.")
end
from_device(x, ::GPUBackend) = x   # overridden by extension; fallback is identity

_resolve_backend(run_on::Symbol) =
    run_on === :cpu ? CPUBackend() :
    run_on === :gpu ? GPUBackend() :
    error("run_on must be :cpu or :gpu, got :$(run_on)")
