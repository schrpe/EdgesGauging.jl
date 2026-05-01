@testset "profiles" begin

    # ── helpers ───────────────────────────────────────────────────────────────

    # n×n image where pixel (r, c) holds the value 10*r + c
    ramp_img(n=10) = Float64[10*r + c for r in 1:n, c in 1:n]

    # n×n image filled with the constant `v`
    const_img(n=10, v=7.0) = fill(v, n, n)

    # n×n image with a bright disc of radius `R` centred at (cr, cc)
    disc_img(n=21, cr=11.0, cc=11.0, R=5.0; bright=100.0) =
        Float64[hypot(r - cr, c - cc) < R ? bright : 0.0 for r in 1:n, c in 1:n]

    # ── line, slim, all three interpolation modes ────────────────────────────

    @testset "line slim, NEAREST: returns exact pixel values" begin
        img = ramp_img(10)
        # Horizontal segment along row 3, cols 1..5
        p = extract_line_profile(img, (3.0, 1.0), (3.0, 5.0);
                                 width=1, interp=INTERP_NEAREST)
        @test p == [31.0, 32.0, 33.0, 34.0, 35.0]
    end

    @testset "line slim, BILINEAR on ramp: exact at integer cols" begin
        img = ramp_img(10)
        p = extract_line_profile(img, (3.0, 1.0), (3.0, 5.0);
                                 width=1, interp=INTERP_BILINEAR)
        @test p ≈ [31.0, 32.0, 33.0, 34.0, 35.0] atol=1e-12
    end

    @testset "line slim, BICUBIC on ramp: ≈ exact at interior integer cols" begin
        img = ramp_img(10)
        # BSpline(Cubic(Reflect(OnCell()))) reconstructs exactly at integer
        # positions in the interior; positions at exactly index 1 or n suffer
        # from the asymmetric Reflect prefilter at the corner cell, so we
        # sample cols 2..6 only (inside the stable region).
        p = extract_line_profile(img, (3.0, 2.0), (3.0, 6.0);
                                 width=1, interp=INTERP_BICUBIC)
        @test p ≈ [32.0, 33.0, 34.0, 35.0, 36.0] atol=1e-8
    end

    @testset "line slim: monotone profile across diagonal ramp" begin
        img = ramp_img(10)
        # Diagonal interior of the image — avoid the corner cells where
        # bicubic Reflect prefilter is asymmetric (see test above).
        p = extract_line_profile(img, (2.0, 2.0), (9.0, 9.0);
                                 width=1, interp=INTERP_BICUBIC)
        @test issorted(p)
        # Auto-density: hypot(7, 7) ≈ 9.90 → ceil(L)+1 = 11
        @test length(p) == 11
    end

    # ── line, wide strip ──────────────────────────────────────────────────────

    @testset "line width=3 on constant image: profile is constant" begin
        img = const_img(10, 7.0)
        p = extract_line_profile(img, (5.0, 2.0), (5.0, 8.0);
                                 width=3, interp=INTERP_NEAREST)
        @test all(==(7.0), p)
    end

    @testset "line width=5 on constant image with NaN-handling" begin
        img = const_img(10, 7.0)
        # Centre on row 1 → strip extends to rows -1..3.  Rows 1,2,3 valid; -1,0 OOB.
        p = extract_line_profile(img, (1.0, 2.0), (1.0, 8.0);
                                 width=5, interp=INTERP_NEAREST)
        # Mean over valid samples only → all entries should still equal 7
        @test all(==(7.0), p)
    end

    @testset "line width even (4): centred offsets [-1.5, -0.5, 0.5, 1.5]" begin
        img = const_img(10, 7.0)
        p = extract_line_profile(img, (5.0, 2.0), (5.0, 8.0);
                                 width=4, interp=INTERP_NEAREST)
        @test all(==(7.0), p)
    end

    # ── line: out-of-bounds extension ─────────────────────────────────────────

    @testset "line slim extends past right edge: trailing NaN" begin
        img = ramp_img(5)
        # Segment row=3, cols 4..7 — col 6,7 are OOB
        p = extract_line_profile(img, (3.0, 4.0), (3.0, 7.0);
                                 n_samples=4, interp=INTERP_NEAREST)
        @test p[1] == 34.0
        @test p[2] == 35.0
        @test isnan(p[3])
        @test isnan(p[4])
    end

    @testset "line slim extends past left edge: leading NaN" begin
        img = ramp_img(5)
        # Segment row=3, cols -1..2 — col -1, 0 are OOB
        p = extract_line_profile(img, (3.0, -1.0), (3.0, 2.0);
                                 n_samples=4, interp=INTERP_NEAREST)
        @test isnan(p[1])
        @test isnan(p[2])
        @test p[3] == 31.0
        @test p[4] == 32.0
    end

    @testset "line width=3, single centreline position fully OOB" begin
        img = const_img(10, 7.0)
        # Segment runs along row 5, cols 0,1,2.  At col=0 with width=3 around row 5:
        # all three perpendicular samples (rows 4,5,6) at col=0 are OOB → NaN.
        p = extract_line_profile(img, (5.0, 0.0), (5.0, 2.0);
                                 n_samples=3, width=3, interp=INTERP_NEAREST)
        @test isnan(p[1])
        @test p[2] == 7.0
        @test p[3] == 7.0
    end

    # ── line degenerate cases ─────────────────────────────────────────────────

    @testset "line degenerate: p0 == p1" begin
        img = ramp_img(10)
        # Same start and end → constant profile of that one pixel value
        p = extract_line_profile(img, (3.0, 4.0), (3.0, 4.0);
                                 n_samples=5, interp=INTERP_NEAREST)
        @test all(==(34.0), p)
    end

    @testset "line: explicit n_samples=1 returns single sample" begin
        img = ramp_img(10)
        p = extract_line_profile(img, (3.0, 4.0), (7.0, 8.0);
                                 n_samples=1, interp=INTERP_NEAREST)
        @test length(p) == 1
        @test p[1] == 34.0   # t=0 → start point
    end

    @testset "line: n_samples=0 auto-density ≈ ceil(L)+1" begin
        img = ramp_img(20)
        p = extract_line_profile(img, (1.0, 1.0), (1.0, 11.0);
                                 width=1, interp=INTERP_NEAREST)
        # L = 10 → expected length = ceil(10) + 1 = 11
        @test length(p) == 11
    end

    # ── arc ───────────────────────────────────────────────────────────────────

    @testset "arc: full circle inside disc → constant profile" begin
        img = disc_img(21, 11.0, 11.0, 5.0; bright=100.0)
        # Sample at radius=3 inside the disc → all bright
        p = extract_arc_profile(img, (11.0, 11.0), 3.0, 0.0, 2π;
                                n_samples=24, interp=INTERP_NEAREST)
        @test all(==(100.0), p)
    end

    @testset "arc: arc-length-uniform auto-density doubles with radius" begin
        img = const_img(40, 1.0)
        p1 = extract_arc_profile(img, (20.0, 20.0), 5.0, 0.0, π;
                                 width=1, interp=INTERP_NEAREST)
        p2 = extract_arc_profile(img, (20.0, 20.0), 10.0, 0.0, π;
                                 width=1, interp=INTERP_NEAREST)
        # Auto-density: ceil(arc_len)+1 — twice the radius → ≈ twice the length
        @test length(p2) ≈ 2 * length(p1) atol=2
    end

    @testset "arc width>1 across disc edge: averaged step" begin
        img = disc_img(41, 21.0, 21.0, 8.0; bright=100.0)
        # Walk along a ray of constant angle, but use arc with very small angular range
        # (just one point on the arc) — this is an arc-length test of the step
        # transition.  Use radius=8 (right at the boundary), width=5 (samples at
        # radii 6,7,8,9,10; first 3 inside disc=100, last 2 outside disc=0).
        p = extract_arc_profile(img, (21.0, 21.0), 8.0, 0.0, π/4;
                                n_samples=4, width=5, interp=INTERP_NEAREST)
        # Mean of (100, 100, 100, 0, 0) = 60.  Allow some tolerance because
        # rounding of pixel positions on the disc boundary makes a few
        # boundary pixels behave differently.
        for v in p
            @test 0.0 < v < 100.0   # genuinely intermediate, not pure 0 or 100
        end
    end

    @testset "arc degenerate: start_angle == end_angle" begin
        img = disc_img(21, 11.0, 11.0, 5.0; bright=100.0)
        p = extract_arc_profile(img, (11.0, 11.0), 3.0, 0.5, 0.5;
                                n_samples=4, interp=INTERP_NEAREST)
        # All four sample positions identical → all the same value
        @test length(p) == 4
        @test all(==(p[1]), p)
    end

    @testset "arc with centre near image edge: leading/trailing NaN" begin
        img = const_img(10, 7.0)
        # Centre at (1,1), radius 3 — most of the arc is OOB
        p = extract_arc_profile(img, (1.0, 1.0), 3.0, 0.0, 2π;
                                n_samples=20, interp=INTERP_NEAREST)
        @test any(isnan, p)        # somewhere on the arc we leave the image
        @test any(==(7.0), p)      # somewhere on the arc we are inside
    end

    # ── pixel-centre convention ───────────────────────────────────────────────

    @testset "pixel-centre convention: integer index hits centre" begin
        img = ramp_img(10)
        # All three modes must agree on the centre of pixel [3, 4]
        v_n = extract_line_profile(img, (3.0, 4.0), (3.0, 4.0);
                                   n_samples=1, interp=INTERP_NEAREST)
        v_l = extract_line_profile(img, (3.0, 4.0), (3.0, 4.0);
                                   n_samples=1, interp=INTERP_BILINEAR)
        v_c = extract_line_profile(img, (3.0, 4.0), (3.0, 4.0);
                                   n_samples=1, interp=INTERP_BICUBIC)
        @test v_n[1] ≈ 34.0 atol=1e-12
        @test v_l[1] ≈ 34.0 atol=1e-12
        @test v_c[1] ≈ 34.0 atol=1e-10
    end

    @testset "pixel-centre convention: NEAREST honours [0.5, n+0.5] extent" begin
        img = ramp_img(10)
        # (0.5, 0.5) is the outer corner of pixel [1,1] → still in-bounds for nearest
        v_in = extract_line_profile(img, (0.5, 0.5), (0.5, 0.5);
                                    n_samples=1, interp=INTERP_NEAREST)
        # floor(0.5 + 0.5) = 1 → image[1,1] = 11
        @test v_in[1] == 11.0

        # (0.4, 0.4) is just outside the image → NaN
        v_out = extract_line_profile(img, (0.4, 0.4), (0.4, 0.4);
                                     n_samples=1, interp=INTERP_NEAREST)
        @test isnan(v_out[1])
    end

    # ── error handling ────────────────────────────────────────────────────────

    @testset "argument validation" begin
        img = const_img(10)
        @test_throws ArgumentError extract_line_profile(img, (1.0, 1.0), (5.0, 5.0); width=0)
        @test_throws ArgumentError extract_line_profile(img, (1.0, 1.0), (5.0, 5.0); n_samples=-1)
        @test_throws ArgumentError extract_arc_profile(img, (5.0, 5.0), -1.0, 0.0, π)
        @test_throws ArgumentError extract_arc_profile(img, (5.0, 5.0), 3.0, 0.0, π; width=0)
    end

    # ── end-to-end with NaN-tolerant edge detection ──────────────────────────

    @testset "extract → gauge_edges_in_profile with NaN-padded profile" begin
        # Synthetic 1-D step embedded in a 2-D image, then sampled along an
        # arbitrary line that runs partly outside the image.  The NaN-tolerant
        # smoother should let us still find the edge sub-pixel.
        ncols = 60
        nrows = 20
        true_col = 30.7
        img = Float64[c < true_col ? 0.0 : 200.0 for _ in 1:nrows, c in 1:ncols]

        # Horizontal sweep that starts a few columns past the right edge and
        # ends a few columns before the left edge — gives leading and trailing
        # NaNs.
        p = extract_line_profile(img, (10.0, -3.0), (10.0, ncols + 4.0);
                                 width=1, interp=INTERP_BICUBIC)
        @test any(isnan, p)
        @test any(!isnan, p)

        r = gauge_edges_in_profile(p, 2.0, 10.0, POLARITY_POSITIVE, SELECT_BEST)
        @test length(r.edges) == 1
        # Profile starts at col=-3, so true_col=30.7 maps to profile index
        # (true_col - (-3)) + 1 = 34.7 in pixel-centred parametrisation.
        # Auto n_samples = ceil(ncols+7)+1 = 68 over a range of 67 → step ≈ 1.0
        # so profile index ≈ position along centreline ≈ true_col + 4
        # We don't tightly check the exact mapping — we test the edge can be
        # found on a NaN-padded profile, and is roughly in the middle.
        idx = r.edges[1].position
        @test 30 < idx < 40
    end
end
