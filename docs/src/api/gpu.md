```@meta
CurrentModule = TensorBinding
```

# GPU

GPU-accelerated mirrors of the CPU solvers, powered by CUDA.jl. All entry points accept a `dtype` keyword (`Float32`, `Float64`, `ComplexF32`, `ComplexF64`); non-Hermitian systems require a complex type.

```@autodocs
Modules = [TensorBinding]
Pages   = ["gpu/GPU_tk.jl"]
```
