```@meta
CurrentModule = TensorBinding
```

# GPU

GPU-accelerated mirrors of the CPU solvers, powered by CUDA.jl. Most entry points accept a `type` (alias `dtype`) keyword (`ComplexF32`, `ComplexF64`, `Float32`, `Float64`). Real types are valid only for real Hamiltonians and real-space outputs; `get_bands_gpu` (quantics Fourier transform), the non-Hermitian and time-evolution entry points, and `get_C_gpu` require a complex type.

```@autodocs
Modules = [TensorBinding]
Pages   = ["gpu/GPU_tk.jl"]
```
