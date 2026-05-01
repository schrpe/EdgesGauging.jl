@testset "ransac" begin

    rng_seed(s) = MersenneTwister(s)

    # ── helpers ───────────────────────────────────────────────────────────────

    # n inlier points on y = slope*x + intercept with Gaussian noise σ
    function line_inliers(n, slope, intercept; σ=0.05, rng=rng_seed(0))
        [(x, slope*x + intercept + σ*randn(rng)) for x in range(1.0, 20.0, length=n)]
    end

    # n inlier points on circle (cx,cy,r) with Gaussian noise σ
    function circle_inliers(n, cx, cy, r; σ=0.1, rng=rng_seed(0))
        [(cx + r*cos(θ) + σ*randn(rng), cy + r*sin(θ) + σ*randn(rng))
         for θ in range(0, 2π, length=n+1)[1:end-1]]
    end

    # random outlier points in a box
    random_outliers(n, lo, hi; rng=rng_seed(99)) =
        [(lo + (hi-lo)*rand(rng), lo + (hi-lo)*rand(rng)) for _ in 1:n]

    # ── classic ransac: line ──────────────────────────────────────────────────

    @testset "ransac line: 30% outliers" begin
        rng = rng_seed(1)
        pts = shuffle!(rng, [line_inliers(20, 2.0, 3.0; rng=rng_seed(2));
                              random_outliers(8, 0.0, 25.0; rng=rng_seed(3))])
        model, inl, outl = ransac(pts, LineModel, 0.5; rng=rng)
        @test !isnothing(model)
        @test length(inl) >= 18      # recover most inliers
        @test length(inl) + length(outl) == length(pts)
    end

    @testset "ransac line: 50% outliers" begin
        rng = rng_seed(10)
        inl_pts  = line_inliers(20, -1.0, 5.0; rng=rng_seed(11))
        outl_pts = random_outliers(20, 0.0, 30.0; rng=rng_seed(12))
        pts = shuffle!(rng, [inl_pts; outl_pts])
        model, inl, outl = ransac(pts, LineModel, 0.5; confidence=0.99, rng=rng)
        @test !isnothing(model)
        @test length(inl) >= 16
    end

    @testset "ransac line: 100% inliers (no outliers)" begin
        pts = line_inliers(15, 1.0, 0.0; σ=0.02, rng=rng_seed(20))
        model, inl, outl = ransac(pts, LineModel, 0.3; rng=rng_seed(21))
        @test !isnothing(model)
        @test length(outl) == 0
        @test length(inl)  == 15
    end

    # ── classic ransac: circle ────────────────────────────────────────────────

    @testset "ransac circle: 35% outliers" begin
        rng = rng_seed(30)
        inl_pts  = circle_inliers(24, 50.0, 50.0, 30.0; rng=rng_seed(31))
        outl_pts = random_outliers(13, 0.0, 100.0; rng=rng_seed(32))
        pts = shuffle!(rng, [inl_pts; outl_pts])
        model, inl, outl = ransac(pts, CircleModel, 1.0; rng=rng)
        @test !isnothing(model)
        @test model.cx ≈ 50.0 atol=1.0
        @test model.cy ≈ 50.0 atol=1.0
        @test model.r  ≈ 30.0 atol=1.0
        @test length(inl) >= 20
    end

    # ── too few points ────────────────────────────────────────────────────────

    @testset "ransac: fewer points than DOF returns nothing gracefully" begin
        for (M, n_pts) in [(LineModel, 1), (CircleModel, 2)]
            pts = [(Float64(i), Float64(i)) for i in 1:n_pts]
            model, inl, outl = ransac(pts, M, 0.5; rng=rng_seed(0))
            @test isnothing(model)
            @test isempty(inl)
            @test length(outl) == n_pts
        end
    end

    @testset "ransac: empty point set returns nothing gracefully" begin
        model, inl, outl = ransac(Tuple{Float64,Float64}[], LineModel, 0.5)
        @test isnothing(model)
        @test isempty(inl)
        @test isempty(outl)
    end

    # ── ransac2 with constraints ──────────────────────────────────────────────

    @testset "ransac2 line: angle constraint rejects oblique lines" begin
        rng = rng_seed(40)
        # Data is on a nearly-horizontal line (angle ≈ 0)
        pts = line_inliers(20, 0.1, 2.0; σ=0.05, rng=rng_seed(41))
        c_pass = LineConstraints{Float64}(min_angle=-Float64(π/4), max_angle=Float64(π/4))
        model, inl, _ = ransac2(pts, LineModel, 0.5, c_pass; rng=rng)
        @test !isnothing(model)
        @test length(inl) >= 15
    end

    @testset "ransac2 circle: radius constraint rejects wrong-sized circles" begin
        rng = rng_seed(50)
        # True circle has radius 30 — constraint: 25 ≤ r ≤ 35 → should pass
        pts = shuffle!(rng, [circle_inliers(24, 0.0, 0.0, 30.0; rng=rng_seed(51));
                              random_outliers(10, -60.0, 60.0; rng=rng_seed(52))])
        c = CircleConstraints{Float64}(min_radius=25.0, max_radius=35.0)
        model, inl, _ = ransac2(pts, CircleModel, 1.0, c; rng=rng)
        @test !isnothing(model)
        @test 25.0 <= model.r <= 35.0
        @test length(inl) >= 20
    end

    @testset "ransac2 circle: impossible radius constraint returns nothing" begin
        rng = rng_seed(60)
        pts = circle_inliers(20, 0.0, 0.0, 30.0; rng=rng_seed(61))
        # Require radius > 100, but true radius is 30 → no model should pass
        c = CircleConstraints{Float64}(min_radius=100.0, max_radius=200.0)
        model, inl, _ = ransac2(pts, CircleModel, 1.0, c;
                                 rng=rng, max_iter=200, min_inliers=3)
        @test isnothing(model)
    end

    # ── ransac2 with LineSegmentModel (length-constrained line) ──────────────

    @testset "ransac2 LineSegmentModel: length constraint accepts valid segment" begin
        # Inliers span x ∈ [1, 20] — segment length ≈ 19
        rng = rng_seed(90)
        inl_pts  = line_inliers(20, 0.0, 5.0; σ=0.05, rng=rng_seed(91))
        outl_pts = random_outliers(10, 0.0, 25.0; rng=rng_seed(92))
        pts = shuffle!(rng, [inl_pts; outl_pts])

        c_ok = LineSegmentConstraints{Float64}(min_length=10.0, max_length=50.0)
        model, inl, _ = ransac2(pts, LineSegmentModel, 0.5, c_ok; rng=rng)
        @test !isnothing(model)
        @test length(inl) >= 15
        @test 10.0 <= segment_length(model, pts[inl]) <= 50.0
    end

    @testset "ransac2 LineSegmentModel: max_length rejects over-long segments" begin
        rng = rng_seed(93)
        # Inliers span ~19 pixels
        pts = line_inliers(20, 0.0, 5.0; σ=0.05, rng=rng_seed(94))
        # Require a segment ≤ 5 pixels — the 19-pixel true line is too long and
        # is rejected by data_constraints_met for LineSegmentModel.
        c_short = LineSegmentConstraints{Float64}(min_length=0.0, max_length=5.0)
        model, _, _ = ransac2(pts, LineSegmentModel, 0.5, c_short;
                               rng=rng, max_iter=200)
        @test isnothing(model)
    end

    @testset "ransac2 LineSegmentModel: min_length rejects too-short segments" begin
        rng = rng_seed(95)
        pts = line_inliers(20, 0.0, 5.0; σ=0.05, rng=rng_seed(96))
        # Segment length ~19 pixels — require > 50 → should fail
        c_long = LineSegmentConstraints{Float64}(min_length=50.0, max_length=200.0)
        model, _, _ = ransac2(pts, LineSegmentModel, 0.5, c_long;
                               rng=rng, max_iter=200)
        @test isnothing(model)
    end

    # ── indices are disjoint and cover all points ─────────────────────────────

    # ── ransac2: data-aware constraint rejects incomplete-arc majority ───────

    @testset "ransac2 circle: rejects half-arc majority in favour of full small circle" begin
        # Majority points are a half-arc of a larger circle; a second, smaller
        # circle has full-arc coverage.  With `min_completeness=0.8`, RANSAC
        # must pick the second-best-by-inlier-count but arc-valid circle.
        big_half   = [(40.0*cos(θ), 40.0*sin(θ)) for θ in range(0.0, π, length=40)]
        small_full = [(200.0 + 12.0*cos(θ), 12.0*sin(θ))
                      for θ in range(0.0, 2π, length=33)[1:end-1]]
        rng = rng_seed(80)
        pts = shuffle!(rng, [big_half; small_full])

        c_strict = CircleConstraints{Float64}(min_radius=5.0, max_radius=100.0,
                                              min_completeness=0.8)
        # `initial_outlier_ratio=0.9` keeps the iteration budget large until a
        # minority-consensus valid model is found.
        model, inl, _ = ransac2(pts, CircleModel, 1.0, c_strict;
                                rng=MersenneTwister(81), max_iter=20_000,
                                initial_outlier_ratio=0.9)
        @test !isnothing(model)
        # Winner should be the small full circle, not the large half-arc one.
        @test model.r < 20.0
        @test abs(model.cx - 200.0) < 3.0
    end

    @testset "ransac: inlier ∪ outlier == all indices, no overlap" begin
        rng = rng_seed(70)
        pts = shuffle!(rng, [line_inliers(15, 1.0, 0.0; rng=rng_seed(71));
                              random_outliers(5, 0.0, 20.0; rng=rng_seed(72))])
        _, inl, outl = ransac(pts, LineModel, 0.5; rng=rng)
        @test isempty(intersect(inl, outl))
        @test sort([inl; outl]) == 1:length(pts)
    end

end
