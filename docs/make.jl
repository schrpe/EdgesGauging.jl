# Build the docs from the repo root:
#   julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#   julia --project=docs docs/make.jl

using Documenter
using Edges

DocMeta.setdocmeta!(
    Edges,
    :DocTestSetup,
    :(using Edges, Random);
    recursive = true,
)

makedocs(
    modules  = [Edges],
    authors  = "schrpe",
    sitename = "Edges.jl",
    format   = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical  = "https://schrpe.github.io/Edges.jl",
    ),
    pages = [
        "Home"     => "index.md",
        "API"      => "api.md",
    ],
    checkdocs = :exports,
)
