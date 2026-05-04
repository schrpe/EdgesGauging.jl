#=
Run from the repo root:

    julia --project=docs docs/generate_readme_example.jl

Reads `test/blob.tif`, demonstrates `gauge_line` and `gauge_circle`,
prints the results, and writes an annotated `test/blob_example.png`
that is referenced from README.md.
=#

using FileIO
using Images
using Random
using EdgesGauging

function main()
    Random.seed!(0)                    # RANSAC samples randomly — fix seed
                                       # so the README numbers reproduce.
    img_path = joinpath(@__DIR__, "..", "test", "blob.tif")
    raw      = load(img_path)
    img      = Float64.(Gray.(raw))     # `gauge_*` works on a Float64 matrix

    # ── Line gauge ────────────────────────────────────────────────────────────
    # The slanted right edge of the part runs from (row≈140, col≈1060) down to
    # roughly (row≈310, col≈1290) before flattening into a vertical run. A
    # LEFT_TO_RIGHT scan over an ROI on that diagonal produces one strong
    # bright→dark transition per row.
    # ROI is positioned so the upper strips graze the part's peak vertex (the
    # right edge has not yet started its diagonal there) and the lower strips
    # touch the slant→vertical transition. Both regions deviate from a single
    # straight line, producing a few RANSAC outliers.
    line_roi     = (30, 380, 140, 530)           # (r0, c0, r1, c1)
    line_spacing = 15.0                          # strip spacing (px)
    line_thick_strips = 3                        # strip thickness (px)
    line_sigma   = 2.0                           # Gaussian smoothing
    line_thresh  = 0.05                          # min |gradient|
    line_fit = gauge_line(
        img, line_roi, LEFT_TO_RIGHT,
        line_spacing, line_thick_strips,
        line_sigma, line_thresh;
        polarity = POLARITY_NEGATIVE,            # bright → dark
        selector = SELECT_BEST,                  # one edge per strip
        inlier_threshold = 1.0,                  # RANSAC inlier distance (px)
    )

    # Tangent direction of `Ax + By + C = 0` is `(-B, A)`; orientation is the
    # angle that direction makes with the +x axis, normalised to [-90°, 90°].
    angle_rad = atan(line_fit.A, -line_fit.B)
    angle_rad = mod(angle_rad + π/2, π) - π/2
    angle_deg = rad2deg(angle_rad)
    println("─ Line fit ────────────────────────────────────────────────")
    println("  A,B,C    = ", round(line_fit.A; digits=4), ", ",
                              round(line_fit.B; digits=4), ", ",
                              round(line_fit.C; digits=2))
    println("  angle    = ", round(angle_deg; digits=2), "° (from +x axis)")
    println("  inliers  = ", length(line_fit.inliers), " / ",
                             length(line_fit.inliers) + length(line_fit.outliers))

    # Re-collect the same edge points the gauge consumed, so we can colour
    # inliers vs outliers on the overlay. `gauge_edge_points_info` is purely
    # deterministic (no RNG), so this does not perturb the RANSAC seed.
    line_strips = gauge_edge_points_info(
        img, line_roi, LEFT_TO_RIGHT,
        line_spacing, line_thick_strips,
        line_sigma, line_thresh,
        POLARITY_NEGATIVE, SELECT_BEST,
    )
    line_pts = [(e.x, e.y) for strip in line_strips for e in strip]   # (col, row)

    # ── Circle gauge ──────────────────────────────────────────────────────────
    # Large round hole near the top of the part. The approximate centre is
    # enough — the radial scan + RANSAC find the true centre and radius to
    # sub-pixel accuracy.
    centre_rc      = (135.0, 374.0)              # (row, col) — rough estimate
    profile_length = 60                          # pixels along each radial ray
                                                 # (must exceed the true radius)
    spacing_rad    = deg2rad(30.0)               # angular step between rays
    cc = CircleConstraints{Float64}(min_radius = 20.0, max_radius = 70.0)
    circle_fit = gauge_circle(
        img, centre_rc,
        0.0, 2π,                                 # full 360°
        spacing_rad,
        profile_length,
        2.0,                                     # sigma
        0.05;                                    # threshold
        polarity = POLARITY_POSITIVE,            # outward ray: dark hole → bright metal
        selector = SELECT_BEST,
        inlier_threshold = 1.0,
        constraints = cc,
        refine = true,                           # geometric LM refinement
    )
    println()
    println("─ Circle fit ──────────────────────────────────────────────")
    println("  centre (col,row) = (",
            round(circle_fit.cx; digits=3), ", ",
            round(circle_fit.cy; digits=3), ")")
    println("  radius (px)      = ", round(circle_fit.r;  digits=3))
    println("  inliers          = ", length(circle_fit.inliers), " / ",
            length(circle_fit.inliers) + length(circle_fit.outliers))

    circle_edges = gauge_circular_edge_points_info(
        img, centre_rc,
        0.0, 2π, spacing_rad, profile_length,
        2.0, 0.05,
        POLARITY_POSITIVE, SELECT_BEST,
    )
    circle_pts = [(e.x, e.y) for e in circle_edges]                  # (col, row)

    # ── Annotated overlay ─────────────────────────────────────────────────────
    overlay = RGB{Float64}.(Gray.(img))

    function setpx!(canvas, r, c, color)
        if 1 <= r <= size(canvas, 1) && 1 <= c <= size(canvas, 2)
            canvas[r, c] = color
        end
    end
    function stamp!(canvas, r, c, color; thick = 2)
        for dr in -thick:thick, dc in -thick:thick
            if dr*dr + dc*dc <= thick*thick
                setpx!(canvas, r + dr, c + dc, color)
            end
        end
    end
    # Hollow ring stamp — draws only the rim of the disc so the underlying
    # pixel content (e.g. the fitted line/circle) remains visible inside.
    function ring!(canvas, r, c, color; outer = 4, inner = 2)
        o2, i2 = outer*outer, inner*inner
        for dr in -outer:outer, dc in -outer:outer
            d = dr*dr + dc*dc
            if i2 < d <= o2
                setpx!(canvas, r + dr, c + dc, color)
            end
        end
    end
    function draw_segment!(canvas, r0, c0, r1, c1, color; thick = 1)
        n = max(abs(r1 - r0), abs(c1 - c0)) + 1
        for k in 0:n-1
            t = n == 1 ? 0.0 : k / (n - 1)
            stamp!(canvas, round(Int, r0 + t*(r1 - r0)),
                           round(Int, c0 + t*(c1 - c0)), color; thick)
        end
    end

    # Stamp sizes for the 480×640 canvas. Points are drawn LAST and slightly
    # thicker than the fit so they remain visible: when the fit is sub-pixel
    # accurate, the points would otherwise sit completely under the fit stroke.
    line_thick   = 1
    circle_thick = 1
    point_thick  = 3

    # Colour palette (kept distinct so each layer reads cleanly).
    roi_color     = RGB(0.0, 0.85, 1.0)   # cyan — measurement windows
    inlier_color  = RGB(0.0, 1.0, 0.3)    # bright green — RANSAC inliers
    outlier_color = RGB(1.0, 0.4, 0.0)    # orange — RANSAC outliers
    fit_color     = RGB(1.0, 1.0, 0.0)    # yellow — fitted line and circle

    # 1) Line measurement window (the ROI rectangle) — cyan outline
    r0, c0, r1, c1 = line_roi
    draw_segment!(overlay, r0, c0, r0, c1, roi_color; thick = 1)
    draw_segment!(overlay, r1, c0, r1, c1, roi_color; thick = 1)
    draw_segment!(overlay, r0, c0, r1, c0, roi_color; thick = 1)
    draw_segment!(overlay, r0, c1, r1, c1, roi_color; thick = 1)

    # 2) Circle measurement window (max ray extent) — solid cyan circle
    cy0, cx0 = centre_rc
    n_disc = max(360, ceil(Int, 2π * profile_length))
    for k in 0:n_disc-1
        θ = 2π * k / n_disc
        r = cy0 + profile_length * sin(θ)
        c = cx0 + profile_length * cos(θ)
        stamp!(overlay, round(Int, r), round(Int, c), roi_color; thick = 1)
    end

    # 3) Fitted line clipped to the ROI's row span — yellow (thin)
    for r in r0:r1
        # Ax + By + C = 0  with x = col, y = row  ⇒  c = -(B*r + C)/A   (A != 0)
        if abs(line_fit.A) > 1e-9
            c = -(line_fit.B * r + line_fit.C) / line_fit.A
            stamp!(overlay, r, round(Int, c), fit_color; thick = line_thick)
        end
    end

    # 4) Fitted circle — yellow (thin)
    n_arc = max(720, ceil(Int, 2π * circle_fit.r))
    for k in 0:n_arc-1
        θ = 2π * k / n_arc
        r = circle_fit.cy + circle_fit.r * sin(θ)
        c = circle_fit.cx + circle_fit.r * cos(θ)
        stamp!(overlay, round(Int, r), round(Int, c), fit_color; thick = circle_thick)
    end

    # 5) Edge points last. Inliers as hollow rings — they sit *on* the fit by
    #    construction, so a hollow marker keeps the fit line/circle visible
    #    inside each marker. Outliers, which lie off the fit, are filled.
    for i in line_fit.outliers
        x, y = line_pts[i]
        stamp!(overlay, round(Int, y), round(Int, x), outlier_color; thick = point_thick)
    end
    for i in line_fit.inliers
        x, y = line_pts[i]
        ring!(overlay, round(Int, y), round(Int, x), inlier_color;
              outer = point_thick + 1, inner = point_thick - 1)
    end
    for i in circle_fit.outliers
        x, y = circle_pts[i]
        stamp!(overlay, round(Int, y), round(Int, x), outlier_color; thick = point_thick)
    end
    for i in circle_fit.inliers
        x, y = circle_pts[i]
        ring!(overlay, round(Int, y), round(Int, x), inlier_color;
              outer = point_thick + 1, inner = point_thick - 1)
    end

    out_path = joinpath(@__DIR__, "..", "test", "blob_example.png")
    save(out_path, overlay)
    println()
    println("Annotated overlay saved to ",
            relpath(out_path, joinpath(@__DIR__, "..")),
            "  (size ", size(overlay), ")")
end

main()
