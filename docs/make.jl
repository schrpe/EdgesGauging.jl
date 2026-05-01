# Build the docs from the repo root:
#   julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#   julia --project=docs docs/make.jl

using Documenter
using EdgesGauging

DocMeta.setdocmeta!(
    EdgesGauging,
    :DocTestSetup,
    :(using EdgesGauging, Random);
    recursive = true,
)

makedocs(
    modules  = [EdgesGauging],
    authors  = "schrpe",
    sitename = "EdgesGauging.jl",
    format   = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical  = "https://schrpe.github.io/EdgesGauging.jl",
    ),
    pages = [
        "Home"     => "index.md",
        "API"      => "api.md",
    ],
    checkdocs = :exports,
)
