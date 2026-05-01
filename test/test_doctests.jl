using Documenter, Edges, Random

DocMeta.setdocmeta!(
    Edges,
    :DocTestSetup,
    :(using Edges, Random);
    recursive = true,
)

@testset "Doctests" begin
    doctest(Edges; manual=false)
end
