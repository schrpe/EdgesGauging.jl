@testset "fitting" begin

    # ── fit_line_tls ──────────────────────────────────────────────────────────

    @testset "fit_line_tls: diagonal line" begin
        pts = [(x, x + 1.0) for x in 1.0:10.0]
        A, B, C = fit_line_tls(pts)
        # y = x+1  ↔  x - y + 1 = 0  ↔  A=1/√2, B=-1/√2, C=1/√2
        @test A^2 + B^2 ≈ 1.0 atol=1e-10
        for p in pts
            @test abs(A*p[1] + B*p[2] + C) < 1e-10
        end
    end

    @testset "fit_line_tls: horizontal line y=5" begin
        pts = [(x, 5.0) for x in 1.0:10.0]
        A, B, C = fit_line_tls(pts)
        @test A^2 + B^2 ≈ 1.0 atol=1e-10
        @test all(abs(A*p[1] + B*p[2] + C) < 1e-10 for p in pts)
    end

    @testset "fit_line_tls: vertical line x=3" begin
        pts = [(3.0, y) for y in 1.0:10.0]
        A, B, C = fit_line_tls(pts)
        @test A^2 + B^2 ≈ 1.0 atol=1e-10
        @test all(abs(A*p[1] + B*p[2] + C) < 1e-10 for p in pts)
    end

    @testset "fit_line_tls: too few points throws" begin
        @test_throws ArgumentError fit_line_tls([(1.0, 2.0)])
    end

    # ── fit_circle_kasa ─────────────────────────────────────────────────────

    @testset "fit_circle_kasa: unit circle" begin
        angles = range(0, 2π, length=20)[1:end-1]
        pts = [(cos(θ), sin(θ)) for θ in angles]
        cx, cy, r = fit_circle_kasa(pts)
        @test cx ≈ 0.0 atol=1e-8
        @test cy ≈ 0.0 atol=1e-8
        @test r  ≈ 1.0 atol=1e-8
    end

    @testset "fit_circle_kasa: offset circle" begin
        angles = range(0, 2π, length=30)[1:end-1]
        pts = [(10.0 + 5.0*cos(θ), -3.0 + 5.0*sin(θ)) for θ in angles]
        cx, cy, r = fit_circle_kasa(pts)
        @test cx ≈ 10.0 atol=1e-6
        @test cy ≈ -3.0 atol=1e-6
        @test r  ≈  5.0 atol=1e-6
    end

    @testset "fit_circle_kasa: too few points throws" begin
        @test_throws ArgumentError fit_circle_kasa([(1.0,0.0),(0.0,1.0)])
    end

    # ── fit_circle_taubin ───────────────────────────────────────────────────

    @testset "fit_circle_taubin: unit circle" begin
        angles = range(0, 2π, length=20)[1:end-1]
        pts = [(cos(θ), sin(θ)) for θ in angles]
        cx, cy, r = fit_circle_taubin(pts)
        @test cx ≈ 0.0 atol=1e-8
        @test cy ≈ 0.0 atol=1e-8
        @test r  ≈ 1.0 atol=1e-8
    end

    @testset "fit_circle_taubin: offset circle" begin
        angles = range(0, 2π, length=30)[1:end-1]
        pts = [(10.0 + 5.0*cos(θ), -3.0 + 5.0*sin(θ)) for θ in angles]
        cx, cy, r = fit_circle_taubin(pts)
        @test cx ≈ 10.0 atol=1e-6
        @test cy ≈ -3.0 atol=1e-6
        @test r  ≈  5.0 atol=1e-6
    end

    @testset "fit_circle_taubin: lower bias than Kåsa on a noisy partial arc" begin
        rng = MersenneTwister(42)
        true_r = 50.0
        true_cx, true_cy = 7.0, -2.0
        # 60° arc — the regime where Kåsa's radius bias is most visible.
        θs = range(0.0, π/3, length=40)
        σ  = 0.05
        pts = [(true_cx + true_r*cos(θ) + σ*randn(rng),
                true_cy + true_r*sin(θ) + σ*randn(rng)) for θ in θs]
        _, _, r_kasa   = fit_circle_kasa(pts)
        _, _, r_taubin = fit_circle_taubin(pts)
        @test abs(r_taubin - true_r) < abs(r_kasa - true_r)
    end

    @testset "fit_circle_taubin: too few points throws" begin
        @test_throws ArgumentError fit_circle_taubin([(1.0,0.0),(0.0,1.0)])
    end

    @testset "fit_circle_taubin: degenerate (coincident) points throw" begin
        @test_throws ArgumentError fit_circle_taubin([(1.0,2.0),(1.0,2.0),(1.0,2.0)])
    end

    # ── fit_circle_lm ───────────────────────────────────────────────────────

    @testset "fit_circle_lm: converges from offset start on a clean circle" begin
        angles = range(0, 2π, length=24)[1:end-1]
        pts = [(2.0 + 3.0*cos(θ), -1.0 + 3.0*sin(θ)) for θ in angles]
        cx, cy, r = fit_circle_lm(pts; cx0=1.5, cy0=-0.5, r0=2.5)
        @test cx ≈  2.0 atol=1e-9
        @test cy ≈ -1.0 atol=1e-9
        @test r  ≈  3.0 atol=1e-9
    end

    @testset "fit_circle_lm: refines a Taubin estimate on noisy partial arc" begin
        rng = MersenneTwister(7)
        true_r = 25.0
        θs = range(0.2, π/2 + 0.2, length=50)   # 90° arc
        σ  = 0.1
        pts = [(4.0 + true_r*cos(θ) + σ*randn(rng),
                6.0 + true_r*sin(θ) + σ*randn(rng)) for θ in θs]
        cx_t, cy_t, r_t = fit_circle_taubin(pts)
        cx_l, cy_l, r_l = fit_circle_lm(pts; cx0=cx_t, cy0=cy_t, r0=r_t)
        rms(c, p, r) = sqrt(sum((sqrt((q[1]-c[1])^2 + (q[2]-c[2])^2) - r)^2 for q in p) / length(p))
        rms_taubin = rms((cx_t, cy_t), pts, r_t)
        rms_lm     = rms((cx_l, cy_l), pts, r_l)
        @test rms_lm <= rms_taubin + 1e-12
    end

    @testset "fit_circle_lm: too few points throws" begin
        @test_throws ArgumentError fit_circle_lm([(1.0,0.0),(0.0,1.0)]; cx0=0.0, cy0=0.0, r0=1.0)
    end

    @testset "fit_circle_lm: point at the centre does not break the iteration" begin
        # One sample exactly at the seeded centre — its Jacobian row is skipped.
        angles = range(0, 2π, length=12)[1:end-1]
        pts = Tuple{Float64,Float64}[(cos(θ), sin(θ)) for θ in angles]
        push!(pts, (0.0, 0.0))
        cx, cy, r = fit_circle_lm(pts; cx0=0.0, cy0=0.0, r0=1.0)
        @test isfinite(cx) && isfinite(cy) && isfinite(r)
    end

    # ── fit_parabola ──────────────────────────────────────────────────────────

    @testset "fit_parabola: exact quadratic" begin
        xs = -5.0:1.0:5.0
        ys = [2x^2 - 3x + 7 for x in xs]
        a, b, c = fit_parabola(collect(xs), collect(ys))
        @test a ≈  2.0 atol=1e-8
        @test b ≈ -3.0 atol=1e-8
        @test c ≈  7.0 atol=1e-8
    end

    @testset "fit_parabola: flat (a≈0, b≈0)" begin
        xs = 1.0:10.0
        ys = fill(42.0, 10)
        a, b, c = fit_parabola(collect(xs), collect(ys))
        @test a ≈ 0.0 atol=1e-8
        @test b ≈ 0.0 atol=1e-8
        @test c ≈ 42.0 atol=1e-8
    end

    @testset "fit_parabola: too few points throws" begin
        @test_throws ArgumentError fit_parabola([1.0, 2.0], [1.0, 2.0])
    end

    @testset "fit_parabola: mismatched lengths throws" begin
        @test_throws ArgumentError fit_parabola([1.0, 2.0, 3.0], [1.0, 2.0])
    end

end
