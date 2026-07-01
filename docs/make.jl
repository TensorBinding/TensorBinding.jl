using TensorBinding
using Documenter

DocMeta.setdocmeta!(TensorBinding, :DocTestSetup, :(using TensorBinding); recursive=true)

makedocs(;
    modules=[TensorBinding],
    authors="TensorBinding",
    sitename="TensorBinding.jl",
    format=Documenter.HTML(;
        canonical="https://TensorBinding.github.io/TensorBinding/",
        edit_link="main",
        assets=String[],
        size_threshold=nothing,
        size_threshold_warn=nothing,
    ),
    pages=[
        "Home" => "index.md",
        "API Reference" => [
            "Core"    => "api/core.md",
            "Lattice" => "api/lattice.md",
            "Solvers" => "api/solvers.md",
            "Physics" => "api/physics.md",
            "GPU"     => "api/gpu.md",
        ],
    ],
)

deploydocs(;
    repo="github.com/TensorBinding/TensorBinding.git",
    devbranch="main",
)
