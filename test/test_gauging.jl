@testset "gauging" begin

    # ── helpers ───────────────────────────────────────────────────────────────

    # Dark-left / bright-right image; vertical edge at `col`
    function vimg(col, nrows=100, ncols=120; amplitude=200.0)
        [c < col ? 0.0 : amplitude for _ in 1:nrows, c in 1:ncols]
    end

    # Bright disc of radius `r` centred at (rc, cc)
    function disc(rc, cc, r, nrows=200, ncols=200; amplitude=200.0)
        [sqrt((row-rc)^2 + (col-cc)^2) < r ? amplitude : 0.0
         for row in 1:nrows, col in 1:ncols]
    end

    # ── gauge_line ────────────────────────────────────────────────────────────

    @testset "gauge_line: vertical edge, default constraints" begin
        img = vimg(60)
        fit = gauge_line(img, (5, 5, 95, 115), LEFT_TO_RIGHT, 8.0, 3, 1.5, 20.0)
        @test fit isa LineFit{Float64}
        # Line should be nearly vertical: |A| >> |B|  (normal is nearly horizontal)
        @test abs(fit.A) > abs(fit.B)
        # Intercept: the line passes near x=60 at any y, so A*60 + C ≈ 0
        x_intercept = -fit.C / fit.A
        @test abs(x_intercept - 60.0) < 2.0
        @test !isempty(fit.inliers)
        @test fit.rms < 1.0
    end

    @testset "gauge_line: too few edge points → GaugingError(:too_few_points)" begin
        # Flat image → no edges detected → error
        img = fill(128.0, 50, 80)
        err = try
            gauge_line(img, (5, 5, 45, 75), LEFT_TO_RIGHT, 5.0, 3, 1.5, 50.0)
            nothing
        catch e
            e
        end
        @test err isa GaugingError
        @test err.reason === :too_few_points
    end

    @testset "gauge_line: inlier + outlier indices cover all strips" begin
        img = vimg(60)
        fit = gauge_line(img, (5, 5, 95, 115), LEFT_TO_RIGHT, 8.0, 3, 1.5, 20.0)
        all_idx = sort([fit.inliers; fit.outliers])
        @test all_idx == 1:length(all_idx)
        @test isempty(intersect(fit.inliers, fit.outliers))
    end

    # ── gauge_circle ──────────────────────────────────────────────────────────

    @testset "gauge_circle: bright disc, outside-in scan" begin
        true_r = 45.0
        img = disc(100, 100, true_r)
        c = CircleConstraints{Float64}(min_radius=30.0, max_radius=60.0, min_completeness=0.5)
        fit = gauge_circle(img, (100.0, 100.0),
                           0.0, 2π, deg2rad(4.0), 80,
                           1.5, 20.0;
                           polarity=POLARITY_NEGATIVE, selector=SELECT_FIRST,
                           constraints=c, confidence=0.99)
        @test fit isa CircleFit{Float64}
        @test fit.cx ≈ 100.0 atol=2.0
        @test fit.cy ≈ 100.0 atol=2.0
        @test fit.r  ≈ true_r  atol=2.0
        @test fit.rms < 2.0
    end

    @testset "gauge_circle: too few edge points → GaugingError(:too_few_points)" begin
        img = fill(0.0, 200, 200)   # no edges at all
        err = try
            gauge_circle(img, (100.0, 100.0), 0.0, 2π, deg2rad(5.0), 80, 1.5, 500.0)
            nothing
        catch e
            e
        end
        @test err isa GaugingError
        @test err.reason === :too_few_points
    end

    @testset "gauge_circle: partial arc fails completeness → :ransac_failed" begin
        # Scan only a narrow wedge; min_completeness=0.9 is impossible from
        # that wedge, so RANSAC (with the data-aware arc check now inside its
        # inner loop) rejects every candidate.
        img = disc(100, 100, 40.0)
        c = CircleConstraints{Float64}(min_radius=30.0, max_radius=55.0,
                                       min_completeness=0.9)
        err = try
            gauge_circle(img, (100.0, 100.0),
                         0.0, π/4, deg2rad(5.0), 70,
                         1.5, 20.0;
                         polarity=POLARITY_NEGATIVE, constraints=c,
                         max_iter=500)
            nothing
        catch e
            e
        end
        @test err isa GaugingError
        @test err.reason === :ransac_failed
    end

    @testset "gauge_circle: refine=true produces RMS ≤ refine=false" begin
        img = disc(100, 100, 40.0)
        c = CircleConstraints{Float64}(min_radius=30.0, max_radius=55.0,
                                       min_completeness=0.5)
        kwargs = (; polarity=POLARITY_NEGATIVE, constraints=c)

        # Re-seed before each call so both RANSAC runs use the same random
        # samples → identical inlier sets → meaningful RMS comparison.
        Random.seed!(12345)
        f0 = gauge_circle(img, (100.0, 100.0), 0.0, 2π, deg2rad(5.0), 70,
                          1.5, 20.0; kwargs..., refine=false)
        Random.seed!(12345)
        f1 = gauge_circle(img, (100.0, 100.0), 0.0, 2π, deg2rad(5.0), 70,
                          1.5, 20.0; kwargs..., refine=true)
        @test f0.inliers == f1.inliers
        # Geometric refinement on the same inlier set cannot increase the
        # orthogonal-distance RMS.
        @test f1.rms <= f0.rms + 1e-9
        @test f1.cx ≈ 100.0 atol=2.0
        @test f1.cy ≈ 100.0 atol=2.0
        @test f1.r  ≈  40.0 atol=2.0
    end

    @testset "gauge_circle: fit result indices disjoint and complete" begin
        img = disc(100, 100, 40.0)
        c = CircleConstraints{Float64}(min_radius=30.0, max_radius=55.0, min_completeness=0.5)
        fit = gauge_circle(img, (100.0, 100.0),
                           0.0, 2π, deg2rad(5.0), 70,
                           1.5, 20.0;
                           polarity=POLARITY_NEGATIVE, constraints=c)
        @test isempty(intersect(fit.inliers, fit.outliers))
        @test sort([fit.inliers; fit.outliers]) == 1:length(fit.inliers)+length(fit.outliers)
    end

    # ── fit quality: RMS is always non-negative and finite ────────────────────

    @testset "all fit results have finite non-negative RMS" begin
        img_line = vimg(60)
        fl = gauge_line(img_line, (5, 5, 95, 115), LEFT_TO_RIGHT, 8.0, 3, 1.5, 20.0)
        @test fl.rms >= 0.0
        @test isfinite(fl.rms)

        img_circ = disc(100, 100, 40.0)
        c = CircleConstraints{Float64}(min_radius=30.0, max_radius=55.0, min_completeness=0.4)
        fc = gauge_circle(img_circ, (100.0, 100.0),
                          0.0, 2π, deg2rad(5.0), 70,
                          1.5, 20.0; polarity=POLARITY_NEGATIVE, constraints=c)
        @test fc.rms >= 0.0
        @test isfinite(fc.rms)
    end

end
