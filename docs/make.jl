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
        size_threshold=nothing,        # index.md holds @autodocs for the whole
        size_threshold_warn=nothing,   # package on one page; disable the
                                        # single-page size guard rail.
    ),
    pages=[
        "Home" => "index.md",
    ],
    build=joinpath(tempdir(), "TensorBinding_docs_build"),  # outside OneDrive sync,
                                                             # avoids rm() lock errors
)

deploydocs(;
    repo="github.com/TensorBinding/TensorBinding.git",
    devbranch="main",
)
