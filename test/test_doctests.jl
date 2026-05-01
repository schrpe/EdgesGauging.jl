using Documenter, EdgesGauging, Random

DocMeta.setdocmeta!(
    EdgesGauging,
    :DocTestSetup,
    :(using EdgesGauging, Random);
    recursive = true,
)

@testset "Doctests" begin
    doctest(EdgesGauging; manual=false)
end
