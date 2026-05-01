@testset "edge_points" begin

    # ── helpers ───────────────────────────────────────────────────────────────

    # 100×100 image with a vertical edge (dark left / bright right) at `col`
    function vimg(col=50, nrows=100, ncols=100; amplitude=200.0)
        [c < col ? 0.0 : amplitude for _ in 1:nrows, c in 1:ncols]
    end

    # 200×200 image with a bright disc of radius `r` centred at (rc, cc)
    function disc_image(rc=100, cc=100, r=40, nrows=200, ncols=200; amplitude=200.0)
        [sqrt((row-rc)^2 + (col-cc)^2) < r ? amplitude : 0.0
         for row in 1:nrows, col in 1:ncols]
    end

    # ── gauge_edge_points_info ────────────────────────────────────────────────

    @testset "multi-strip: one strip per spacing step" begin
        img = vimg(50)
        roi = (10, 5, 90, 95)
        strips = gauge_edge_points_info(img, roi, LEFT_TO_RIGHT, 10.0, 3, 1.5, 20.0,
                                        POLARITY_POSITIVE, SELECT_FIRST)
        # rows 10–90 with spacing 10 → strips centred at rows 10,20,...,90 → 9 strips
        @test length(strips) >= 8
        # every strip should detect at least one edge near column 50
        for strip in strips
            @test length(strip) >= 1
            for edge in strip
                @test abs(edge.x - 50.0) < 2.0
            end
        end
    end

    @testset "multi-strip: TOP_TO_BOTTOM finds horizontal edge" begin
        img = [r < 50 ? 0.0 : 200.0 for r in 1:100, _ in 1:100]
        roi = (5, 10, 95, 90)
        strips = gauge_edge_points_info(img, roi, TOP_TO_BOTTOM, 10.0, 3, 1.5, 20.0,
                                        POLARITY_POSITIVE, SELECT_FIRST)
        @test length(strips) >= 8
        for strip in strips
            @test length(strip) >= 1
            for edge in strip
                @test abs(edge.y - 50.0) < 2.0
            end
        end
    end

    @testset "multi-strip: empty strips when threshold too high" begin
        img = vimg(50)
        roi = (10, 5, 90, 95)
        strips = gauge_edge_points_info(img, roi, LEFT_TO_RIGHT, 10.0, 3, 1.5, 10_000.0)
        @test all(isempty(s) for s in strips)
    end

    # ── gauge_circular_edge_points_info ───────────────────────────────────────

    @testset "circular scan: detects disc boundary near known radius" begin
        img = disc_image(100, 100, 40)
        # Scan from centre outward: bright→dark transition at r≈40
        edges = gauge_circular_edge_points_info(
            img, (100.0, 100.0),
            0.0, 2π, deg2rad(5.0), 80,
            1.5, 20.0, POLARITY_NEGATIVE, SELECT_FIRST)
        @test length(edges) >= 60   # most rays should find the edge
        # All edges should be near radius 40 from centre
        radii = [sqrt((e.x - 100)^2 + (e.y - 100)^2) for e in edges]
        @test mean(radii)       ≈ 40.0 atol=2.0
        @test maximum(abs.(radii .- 40.0)) < 4.0
        # Radial scans use scan_index=nothing since they have no strip concept
        @test all(isnothing(e.scan_index) for e in edges)

        # threaded=true must produce the same edges in the same order
        threaded = gauge_circular_edge_points_info(
            img, (100.0, 100.0),
            0.0, 2π, deg2rad(5.0), 80,
            1.5, 20.0, POLARITY_NEGATIVE, SELECT_FIRST; threaded=true)
        @test [(e.x, e.y, e.strength) for e in edges] ==
              [(e.x, e.y, e.strength) for e in threaded]
    end

    @testset "circular scan: partial arc (π radians)" begin
        img = disc_image(100, 100, 35)
        edges = gauge_circular_edge_points_info(
            img, (100.0, 100.0),
            0.0, π, deg2rad(5.0), 70,
            1.5, 20.0, POLARITY_NEGATIVE, SELECT_FIRST)
        # Half circle → roughly half the rays of a full scan
        @test length(edges) >= 25
        # All x coords should be ≥ centre (right half, angles 0..π covers x≥centre and x≤centre)
        # Actually 0..π covers the upper semicircle (sin>0 for angles 0..π)
    end

    @testset "circular scan: spacing_radians=0 returns empty" begin
        img = disc_image()
        edges = gauge_circular_edge_points_info(
            img, (100.0, 100.0), 0.0, 2π, 0.0, 80, 1.5, 20.0)
        @test isempty(edges)
    end

    # ── gauge_ring_edge_points_info ───────────────────────────────────────────

    @testset "ring scan: detects edge within annulus" begin
        img = disc_image(100, 100, 40)
        edges = gauge_ring_edge_points_info(
            img, (100.0, 100.0),
            20.0, 60.0,       # annulus: inner=20, outer=60
            0.0, 2π, deg2rad(5.0),
            1.5, 20.0, POLARITY_NEGATIVE, SELECT_FIRST)
        @test length(edges) >= 60
        radii = [sqrt((e.x - 100)^2 + (e.y - 100)^2) for e in edges]
        @test all(r -> 18.0 <= r <= 62.0, radii)   # within annulus (with tolerance)
        @test mean(radii) ≈ 40.0 atol=3.0
    end

    @testset "ring scan: edge outside annulus is not reported" begin
        # Disc radius=40; ring only covers r∈[50,80] → no edge in that range
        img = disc_image(100, 100, 40)
        edges = gauge_ring_edge_points_info(
            img, (100.0, 100.0),
            50.0, 80.0,
            0.0, 2π, deg2rad(5.0),
            1.5, 5.0, POLARITY_ANY, SELECT_FIRST)
        # Profile in [50,80] is all dark (0) → no edge above threshold
        @test isempty(edges)
    end

    @testset "ring scan: inner_radius >= outer_radius throws" begin
        img = disc_image()
        @test_throws ArgumentError gauge_ring_edge_points_info(
            img, (100.0, 100.0), 50.0, 30.0, 0.0, 2π, deg2rad(5.0), 1.5, 10.0)
        @test_throws ArgumentError gauge_ring_edge_points_info(
            img, (100.0, 100.0), 40.0, 40.0, 0.0, 2π, deg2rad(5.0), 1.5, 10.0)
    end

    @testset "ring scan: spacing_radians=0 returns empty" begin
        img = disc_image()
        edges = gauge_ring_edge_points_info(
            img, (100.0, 100.0), 20.0, 60.0, 0.0, 2π, 0.0, 1.5, 10.0)
        @test isempty(edges)
    end

    # ── interp kwarg on radial / ring scans ──────────────────────────────────

    @testset "circular scan: bilinear interp finds disc boundary" begin
        img = disc_image(100, 100, 40)
        edges = gauge_circular_edge_points_info(
            img, (100.0, 100.0),
            0.0, 2π, deg2rad(5.0), 80,
            1.5, 20.0, POLARITY_NEGATIVE, SELECT_FIRST;
            interp=INTERP_BILINEAR)
        @test length(edges) >= 60
        radii = [sqrt((e.x - 100)^2 + (e.y - 100)^2) for e in edges]
        @test mean(radii) ≈ 40.0 atol=2.0
    end

    @testset "circular scan: nearest interp finds disc boundary (coarser)" begin
        img = disc_image(100, 100, 40)
        edges = gauge_circular_edge_points_info(
            img, (100.0, 100.0),
            0.0, 2π, deg2rad(5.0), 80,
            1.5, 20.0, POLARITY_NEGATIVE, SELECT_FIRST;
            interp=INTERP_NEAREST)
        @test length(edges) >= 60
        radii = [sqrt((e.x - 100)^2 + (e.y - 100)^2) for e in edges]
        @test mean(radii) ≈ 40.0 atol=3.0   # coarser tolerance for nearest
    end

    @testset "circular scan: OOB rays produce NaN, edge still found" begin
        # Disc near the corner; a few rays will leave the image partway through.
        img = disc_image(40, 40, 25, 100, 100)
        # Profile length 90 from centre (40, 40) — many rays go OOB
        edges = gauge_circular_edge_points_info(
            img, (40.0, 40.0),
            0.0, 2π, deg2rad(10.0), 90,
            1.5, 20.0, POLARITY_NEGATIVE, SELECT_FIRST)
        # Despite OOB samples on some rays, the disc boundary at r=25 is well
        # inside the image and should be detected on most rays.
        @test length(edges) >= 20
    end

    @testset "ring scan: interp kwarg is accepted" begin
        img = disc_image(100, 100, 40)
        for mode in (INTERP_NEAREST, INTERP_BILINEAR, INTERP_BICUBIC)
            edges = gauge_ring_edge_points_info(
                img, (100.0, 100.0),
                20.0, 60.0,
                0.0, 2π, deg2rad(5.0),
                1.5, 20.0, POLARITY_NEGATIVE, SELECT_FIRST;
                interp=mode)
            @test length(edges) >= 50
        end
    end

end
