@testset "image_edges" begin

    # Synthetic image: dark left half, bright right half, vertical edge at column `edge_col`
    function vertical_edge_image(nrows, ncols, edge_col; amplitude=200.0, noise=0.0, rng=MersenneTwister(0))
        img = [c < edge_col ? 0.0 : amplitude for _ in 1:nrows, c in 1:ncols]
        noise > 0 && (img .+= randn(rng, nrows, ncols) .* noise)
        return img
    end

    # Synthetic image: dark top half, bright bottom half, horizontal edge at row `edge_row`
    function horizontal_edge_image(nrows, ncols, edge_row; amplitude=200.0)
        return [r < edge_row ? 0.0 : amplitude for r in 1:nrows, _ in 1:ncols]
    end

    # ── LEFT_TO_RIGHT ─────────────────────────────────────────────────────────

    @testset "LEFT_TO_RIGHT: detects vertical edge" begin
        img = vertical_edge_image(50, 80, 40)
        roi = (5, 5, 45, 75)
        edges = gauge_edges(img, roi, LEFT_TO_RIGHT, 1.5, 20.0, POLARITY_POSITIVE, SELECT_FIRST)
        @test length(edges) == 41   # rows 5–45
        cols = [e.x for e in edges]
        @test all(abs(c - 40.0) < 1.5 for c in cols)
        @test all(e.y >= 5 && e.y <= 45 for e in edges)
    end

    # ── RIGHT_TO_LEFT ─────────────────────────────────────────────────────────

    @testset "RIGHT_TO_LEFT: same edge found, coordinate not mirrored" begin
        img = vertical_edge_image(50, 80, 40)
        roi = (5, 5, 45, 75)
        edges_ltr = gauge_edges(img, roi, LEFT_TO_RIGHT,  1.5, 20.0, POLARITY_POSITIVE, SELECT_FIRST)
        edges_rtl = gauge_edges(img, roi, RIGHT_TO_LEFT,  1.5, 20.0, POLARITY_NEGATIVE, SELECT_FIRST)
        # RTL scans dark→bright reversed, so the step appears as a rising edge in the
        # reversed profile (POLARITY_NEGATIVE from RTL perspective = actual falling from left)
        @test length(edges_rtl) == length(edges_ltr)
        # Detected column should be consistent between both scan directions
        mean_ltr = mean(e.x for e in edges_ltr)
        mean_rtl = mean(e.x for e in edges_rtl)
        @test abs(mean_ltr - mean_rtl) < 2.0
    end

    # ── TOP_TO_BOTTOM ─────────────────────────────────────────────────────────

    @testset "TOP_TO_BOTTOM: detects horizontal edge" begin
        img = horizontal_edge_image(80, 50, 40)
        roi = (5, 5, 75, 45)
        edges = gauge_edges(img, roi, TOP_TO_BOTTOM, 1.5, 20.0, POLARITY_POSITIVE, SELECT_FIRST)
        @test length(edges) == 41   # cols 5–45
        rows = [e.y for e in edges]
        @test all(abs(r - 40.0) < 1.5 for r in rows)
        @test all(e.x >= 5 && e.x <= 45 for e in edges)
    end

    # ── BOTTOM_TO_TOP ─────────────────────────────────────────────────────────

    @testset "BOTTOM_TO_TOP: same edge found, coordinate correct" begin
        img = horizontal_edge_image(80, 50, 40)
        roi = (5, 5, 75, 45)
        edges = gauge_edges(img, roi, BOTTOM_TO_TOP, 1.5, 20.0, POLARITY_NEGATIVE, SELECT_FIRST)
        @test !isempty(edges)
        rows = [e.y for e in edges]
        @test all(abs(r - 40.0) < 1.5 for r in rows)
    end

    # ── ROI clamping ──────────────────────────────────────────────────────────

    @testset "ROI larger than image is clamped (no crash)" begin
        img = vertical_edge_image(50, 80, 40)
        roi_oversized = (-10, -10, 200, 200)   # way outside
        @test_nowarn gauge_edges(img, roi_oversized, LEFT_TO_RIGHT, 1.5, 20.0)
    end

    @testset "ROI rows inverted (r1 > r2) is handled" begin
        img = vertical_edge_image(50, 80, 40)
        roi_inv = (45, 5, 5, 75)   # r1 > r2, should be swapped internally
        edges = gauge_edges(img, roi_inv, LEFT_TO_RIGHT, 1.5, 20.0, POLARITY_POSITIVE, SELECT_FIRST)
        @test !isempty(edges)
    end

    # ── SELECT_ALL returns multiple edges per profile ─────────────────────────

    @testset "SELECT_ALL returns multiple edges in same row" begin
        # Image with two vertical edges: bright stripe between cols 30 and 50
        img = zeros(40, 80)
        img[:, 30:50] .= 200.0
        roi = (5, 5, 35, 75)
        edges = gauge_edges(img, roi, LEFT_TO_RIGHT, 1.5, 20.0, POLARITY_ANY, SELECT_ALL)
        # Each row should yield 2 edges (rising at ~30, falling at ~50)
        row_counts = Dict{Float64,Int}()
        for e in edges
            row_counts[e.y] = get(row_counts, e.y, 0) + 1
        end
        @test all(v >= 2 for v in values(row_counts))
    end

    # ── flat image — no edges ─────────────────────────────────────────────────

    @testset "flat image returns no edges" begin
        img = fill(128.0, 50, 80)
        roi = (1, 1, 50, 80)
        edges = gauge_edges(img, roi, LEFT_TO_RIGHT, 2.0, 1.0)
        @test isempty(edges)
    end

    # ── scan_index is set correctly ───────────────────────────────────────────

    @testset "scan_index increases with each profile" begin
        img = vertical_edge_image(20, 60, 30)
        roi = (1, 1, 20, 60)
        edges = gauge_edges(img, roi, LEFT_TO_RIGHT, 1.5, 20.0, POLARITY_POSITIVE, SELECT_FIRST)
        indices = [e.scan_index for e in edges]
        @test indices == sort(indices)
        @test indices[1] == 1
    end

    # ── threaded matches serial ───────────────────────────────────────────────

    @testset "threaded=true produces identical output" begin
        img = vertical_edge_image(40, 80, 50)
        roi = (1, 1, 40, 80)
        for orient in (LEFT_TO_RIGHT, RIGHT_TO_LEFT, TOP_TO_BOTTOM, BOTTOM_TO_TOP)
            serial   = gauge_edges(img, roi, orient, 1.5, 20.0)
            threaded = gauge_edges(img, roi, orient, 1.5, 20.0; threaded=true)
            @test length(serial) == length(threaded)
            @test [(e.x, e.y, e.scan_index) for e in serial] ==
                  [(e.x, e.y, e.scan_index) for e in threaded]
        end
    end

end
