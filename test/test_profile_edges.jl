@testset "profile_edges" begin

    # ── helpers ──────────────────────────────────────────────────────────────

    # Smoothed step edge: 0 for x < pos, amplitude for x ≥ pos
    function step_profile(n, pos, amplitude=200.0; noise=0.0, rng=MersenneTwister(0))
        p = [x < pos ? 0.0 : amplitude for x in 1.0:n]
        noise > 0 && (p .+= randn(rng, n) .* noise)
        return p
    end

    # ── basic rising edge ─────────────────────────────────────────────────────

    @testset "rising edge, subpixel accuracy" begin
        true_pos = 30.7
        profile = step_profile(60, true_pos)
        r = gauge_edges_in_profile(profile, 2.0, 10.0, POLARITY_POSITIVE, SELECT_BEST)
        @test length(r.edges) == 1
        @test abs(r.edges[1].position - true_pos) < 0.5   # subpixel
        @test r.edges[1].strength > 0
        @test length(r.smoothed) == 60
        @test length(r.gradient) == 60
    end

    # ── falling edge ─────────────────────────────────────────────────────────

    @testset "falling edge (POLARITY_NEGATIVE)" begin
        # Step from 200 → 0 at position 25
        profile = step_profile(50, 25.0)
        profile = 200.0 .- profile   # invert: falling edge
        r = gauge_edges_in_profile(profile, 2.0, 10.0, POLARITY_NEGATIVE, SELECT_BEST)
        @test length(r.edges) == 1
        @test abs(r.edges[1].position - 25.0) <= 0.5
    end

    # ── POLARITY_ANY finds both edges ─────────────────────────────────────────

    @testset "POLARITY_ANY, SELECT_ALL — two edges" begin
        # Pulse: dark | bright (col 20–40) | dark
        profile = zeros(60)
        profile[20:40] .= 200.0
        r = gauge_edges_in_profile(profile, 1.5, 20.0, POLARITY_ANY, SELECT_ALL)
        @test length(r.edges) >= 2
        positions = sort([e.position for e in r.edges])
        @test positions[1] < 25        # rising edge ≈ 20
        @test positions[2] > 35        # falling edge ≈ 40
    end

    # ── edge selectors ────────────────────────────────────────────────────────

    @testset "SELECT_FIRST returns leftmost edge" begin
        profile = zeros(80)
        profile[15:30] .= 100.0
        profile[50:65] .= 100.0
        r_all   = gauge_edges_in_profile(profile, 1.5, 10.0, POLARITY_ANY, SELECT_ALL)
        r_first = gauge_edges_in_profile(profile, 1.5, 10.0, POLARITY_ANY, SELECT_FIRST)
        @test length(r_first.edges) == 1
        @test r_first.edges[1].position ≈ r_all.edges[1].position
    end

    @testset "SELECT_LAST returns rightmost edge" begin
        profile = zeros(80)
        profile[15:30] .= 100.0
        profile[50:65] .= 100.0
        r_all  = gauge_edges_in_profile(profile, 1.5, 10.0, POLARITY_ANY, SELECT_ALL)
        r_last = gauge_edges_in_profile(profile, 1.5, 10.0, POLARITY_ANY, SELECT_LAST)
        @test length(r_last.edges) == 1
        @test r_last.edges[1].position ≈ r_all.edges[last(eachindex(r_all.edges))].position
    end

    @testset "SELECT_BEST returns strongest edge" begin
        # Weak pulse at 15–20, strong pulse at 50–60
        profile = zeros(80)
        profile[15:20] .= 50.0     # weak
        profile[50:60] .= 200.0    # strong
        r = gauge_edges_in_profile(profile, 1.5, 5.0, POLARITY_POSITIVE, SELECT_BEST)
        @test length(r.edges) == 1
        @test r.edges[1].position > 40   # should find the stronger (right) edge
    end

    # ── sigma = 0: no smoothing ───────────────────────────────────────────────

    @testset "sigma=0 skips smoothing" begin
        profile = step_profile(50, 25.0)
        r = gauge_edges_in_profile(profile, 0.0, 5.0, POLARITY_POSITIVE, SELECT_BEST)
        @test length(r.edges) == 1
        # smoothed should equal input (no filter applied)
        @test r.smoothed ≈ profile
    end

    # ── flat profile — no edges ───────────────────────────────────────────────

    @testset "flat profile returns no edges" begin
        profile = fill(100.0, 50)
        r = gauge_edges_in_profile(profile, 2.0, 1.0, POLARITY_ANY, SELECT_ALL)
        @test isempty(r.edges)
    end

    # ── threshold filters weak edges ──────────────────────────────────────────

    @testset "edge below threshold is suppressed" begin
        profile = step_profile(50, 25.0, 10.0)   # very small amplitude
        r_low  = gauge_edges_in_profile(profile, 1.0, 0.5,  POLARITY_POSITIVE, SELECT_ALL)
        r_high = gauge_edges_in_profile(profile, 1.0, 50.0, POLARITY_POSITIVE, SELECT_ALL)
        @test !isempty(r_low.edges)
        @test isempty(r_high.edges)
    end

    # ── very short profiles ───────────────────────────────────────────────────

    @testset "profile length < 3 returns no edges (no crash)" begin
        for n in 0:2
            profile = rand(n)
            r = gauge_edges_in_profile(profile, 1.0, 0.1)
            @test isempty(r.edges)
        end
    end

    # ── edge at profile boundary ──────────────────────────────────────────────

    @testset "edge near boundary is detected without crash" begin
        # Step at position 3 (close to start)
        profile = step_profile(30, 3.0)
        r = gauge_edges_in_profile(profile, 0.5, 5.0, POLARITY_POSITIVE, SELECT_ALL)
        @test length(r.edges) >= 1
    end

    # ── noisy profile still yields roughly correct position ───────────────────

    @testset "subpixel accuracy under noise" begin
        rng = MersenneTwister(7)
        true_pos = 40.0
        profile = step_profile(80, true_pos, 200.0; noise=5.0, rng=rng)
        r = gauge_edges_in_profile(profile, 2.5, 10.0, POLARITY_POSITIVE, SELECT_BEST)
        @test length(r.edges) == 1
        @test abs(r.edges[1].position - true_pos) < 1.5
    end

    # ── polarity mismatch returns no edges ────────────────────────────────────

    @testset "polarity mismatch suppresses edge" begin
        profile = step_profile(50, 25.0)   # rising edge
        r = gauge_edges_in_profile(profile, 2.0, 5.0, POLARITY_NEGATIVE, SELECT_ALL)
        @test isempty(r.edges)
    end

    # ── NaN tolerance ────────────────────────────────────────────────────────

    @testset "NaN-free profile still detects edge (regression)" begin
        # A profile with no NaNs must take the fast path and produce the
        # same numerical result as before NaN-tolerance was introduced.
        true_pos = 30.7
        profile = step_profile(60, true_pos)
        r = gauge_edges_in_profile(profile, 2.0, 10.0, POLARITY_POSITIVE, SELECT_BEST)
        @test length(r.edges) == 1
        @test abs(r.edges[1].position - true_pos) < 0.5
        # No NaNs in any output buffer
        @test !any(isnan, r.smoothed)
        @test !any(isnan, r.gradient)
    end

    @testset "NaN-padded profile: edge still found in valid interior" begin
        # Same step edge, but with NaN padding at both ends — simulates a
        # profile extracted along a path that runs partly out of the image.
        true_pos = 30.7
        profile = step_profile(60, true_pos)
        profile[1:5]    .= NaN
        profile[56:60]  .= NaN
        r = gauge_edges_in_profile(profile, 2.0, 10.0, POLARITY_POSITIVE, SELECT_BEST)
        @test length(r.edges) == 1
        @test abs(r.edges[1].position - true_pos) < 0.7   # NaN-aware kernel
        # Smoothed array should be NaN exactly where the kernel saw no valid
        # samples — at the NaN-padded ends — and finite in the interior.
        @test isnan(r.smoothed[1])
        @test isnan(r.smoothed[end])
        @test !isnan(r.smoothed[30])
    end

    @testset "all-NaN profile: no edges, no crash" begin
        profile = fill(NaN, 50)
        r = gauge_edges_in_profile(profile, 2.0, 1.0, POLARITY_ANY, SELECT_ALL)
        @test isempty(r.edges)
        @test all(isnan, r.smoothed)
    end

end
