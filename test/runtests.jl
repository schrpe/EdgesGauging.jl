using Test
using EdgesGauging
using Random
using Statistics

@testset "EdgesGauging" begin
    include("test_profile_edges.jl")
    include("test_fitting.jl")
    include("test_models.jl")
    include("test_ransac.jl")
    include("test_image_edges.jl")
    include("test_profiles.jl")
    include("test_edge_points.jl")
    include("test_gauging.jl")
    include("test_doctests.jl")
end
