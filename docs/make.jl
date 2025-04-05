using MyPackageName
using Documenter

DocMeta.setdocmeta!(MyPackageName, :DocTestSetup, :(using MyPackageName); recursive=true)

makedocs(;
    modules=[MyPackageName],
    authors="YTS2223 <yitao.sun@aalto.fi> and contributors",
    sitename="MyPackageName.jl",
    format=Documenter.HTML(;
        canonical="https://YITAOSUN42.github.io/MyPackageName.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/YITAOSUN42/MyPackageName.jl",
    devbranch="master",
)
