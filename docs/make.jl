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
    repo     = "https://github.com/schrpe/EdgesGauging.jl/blob/{commit}{path}#{line}",
    sitename = "EdgesGauging.jl",
    format   = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical  = "https://schrpe.github.io/EdgesGauging.jl",
        assets     = String[],
    ),
    pages = [
        "Home" => "index.md",
        "API"  => "api.md",
    ],
    checkdocs = :exports,
)

deploydocs(;
    repo = "github.com/schrpe/EdgesGauging.jl",
)
