module TensorBindingCUDAExt

# Loaded automatically by Julia 1.9+ when the user does:
#   using CUDA, TensorBinding
# Overrides the error-throwing stubs in src/backends.jl with real GPU dispatch.

using TensorBinding
using CUDA
using ITensors
using ITensorMPS
using NDTensors

# Move tensors to GPU.  NDTensors.cu works on individual ITensors, MPS, and MPO.
TensorBinding.to_device(x::MPS, ::TensorBinding.GPUBackend) = NDTensors.cu(x)
TensorBinding.to_device(x::MPO, ::TensorBinding.GPUBackend) = NDTensors.cu(x)

# Move tensors back to CPU.
TensorBinding.from_device(x::MPS, ::TensorBinding.GPUBackend) = NDTensors.cpu(x)
TensorBinding.from_device(x::MPO, ::TensorBinding.GPUBackend) = NDTensors.cpu(x)

end
