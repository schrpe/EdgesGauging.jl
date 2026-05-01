"""
High-level gauging functions that chain edge-point collection with RANSAC fitting.

Mirrors the C++ `gauge_line()` and `gauge_circle()`.

Each function follows the same three-step pipeline:
  1. Detect edge points (strips or radial profiles).
  2. Fit a geometric model with constraint-aware RANSAC ([`ransac2`](@ref)).
  3. Return a typed fit result ([`LineFit`](@ref) / [`CircleFit`](@ref)).
"""

# ── gauge_line ────────────────────────────────────────────────────────────────

"""
    gauge_line(image, roi, orientation, spacing, thickness, sigma, threshold;
               polarity=POLARITY_ANY, selector=SELECT_FIRST,
               constraints=LineConstraints{Float64}(),
               confidence=0.99, inlier_threshold=1.0, max_iter=10_000)
               -> LineFit{Float64}

Detect edge points along parallel strips within `roi`, then fit a line with
RANSAC outlier rejection.

The function chains [`gauge_edge_points_info`](@ref) → [`ransac2`](@ref) →
[`LineFit`](@ref).

# Arguments
- `roi`, `orientation`, `spacing`, `thickness`, `sigma`, `threshold`,
  `polarity`, `selector`: passed to [`gauge_edge_points_info`](@ref).
- `constraints`: [`LineConstraints`](@ref) to enforce angle and inlier limits.
- `confidence`: RANSAC confidence level (default 0.99).
- `inlier_threshold`: maximum perpendicular distance (pixels) to count a point
  as an inlier (default 1.0).
- `max_iter`: hard cap on RANSAC iterations (default 10 000).

# Returns
A [`LineFit`](@ref) with normalised coefficients `(A, B, C)`, inlier/outlier
index vectors, and RMS residual.

Throws [`GaugingError`](@ref) with `reason = :too_few_points` when edge
detection yields fewer than 2 points, or `reason = :ransac_failed` when RANSAC
cannot find a line that satisfies `constraints`.

# Examples
```jldoctest
julia> img = [col < 30 ? 0.0 : 200.0 for _ in 1:50, col in 1:60];

julia> fit = gauge_line(img, (5,5,45,55), LEFT_TO_RIGHT, 5.0, 3, 1.5, 20.0);

julia> fit isa LineFit{Float64}
true

julia> fit.rms < 1.0
true

julia> abs(-fit.C / fit.A - 30.0) < 2.0   # line passes near col 30
true
```
"""
function gauge_line(
    image       :: AbstractMatrix,
    roi         :: NTuple{4,Int},
    orientation :: ScanOrientation,
    spacing     :: Real,
    thickness   :: Int,
    sigma       :: Real,
    threshold   :: Real;
    polarity         :: EdgePolarity          = POLARITY_ANY,
    selector         :: EdgeSelector          = SELECT_FIRST,
    constraints      :: LineConstraints       = LineConstraints{Float64}(),
    confidence       :: Real                  = 0.99,
    inlier_threshold :: Real                  = 1.0,
    max_iter         :: Int                   = 10_000,
) :: LineFit{Float64}

    strip_results = gauge_edge_points_info(image, roi, orientation,
                                           spacing, thickness,
                                           sigma, threshold, polarity, selector)
    pts = _flatten_to_xy(strip_results)
    length(pts) < 2 && throw(GaugingError(:too_few_points,
        "gauge_line: too few edge points detected ($(length(pts)))"))

    model, inliers, outliers = ransac2(pts, LineModel, inlier_threshold, constraints;
                                       min_inliers=constraints.min_inlier_count,
                                       confidence=confidence, max_iter=max_iter)

    isnothing(model) && throw(GaugingError(:ransac_failed,
        "gauge_line: RANSAC could not find a valid line"))

    rms = rms_error(model, pts[inliers])
    return LineFit{Float64}(model.A, model.B, model.C, inliers, outliers, rms)
end

# ── gauge_circle ──────────────────────────────────────────────────────────────

"""
    gauge_circle(image, center, start_angle, angular_span, spacing_radians,
                 profile_length, sigma, threshold;
                 polarity=POLARITY_ANY, selector=SELECT_FIRST,
                 constraints=CircleConstraints{Float64}(),
                 confidence=0.99, inlier_threshold=1.0, max_iter=10_000,
                 refine=false)
                 -> CircleFit{Float64}

Detect edge points along radial profiles from `center_rc` and fit a circle with
RANSAC outlier rejection.

The function chains [`gauge_circular_edge_points_info`](@ref) → [`ransac2`](@ref)
(which rejects arc-incomplete candidates via [`data_constraints_met`](@ref))
→ [`CircleFit`](@ref).

# Arguments
- `center_rc`: `(row, col)` of the approximate circle centre (1-based). The
  `_rc` suffix is a reminder that this is **(row, col)**, not `(x, y)`.
- `start_angle`, `angular_span`, `spacing_radians`, `profile_length`: passed to
  [`gauge_circular_edge_points_info`](@ref).
- `sigma`, `threshold`, `polarity`, `selector`: edge detection parameters.
- `constraints`: [`CircleConstraints`](@ref) to enforce radius limits and arc
  completeness.
- `confidence`, `inlier_threshold`, `max_iter`: RANSAC parameters.
- `refine`: when `true`, run a geometric Levenberg-Marquardt refit
  ([`fit_circle_lm`](@ref)) on the final inlier set after RANSAC, starting
  from the algebraic Taubin estimate.  This minimises orthogonal distances
  rather than the (biased) algebraic residual and gives the maximum-likelihood
  centre and radius under isotropic noise.  Adds a few extra iterations of
  cost; default `false`.

# Returns
A [`CircleFit`](@ref) with `(cx, cy, r)`, inlier/outlier indices, and RMS.

Throws [`GaugingError`](@ref) with `reason`:
- `:too_few_points` — fewer than 3 edge points detected.
- `:ransac_failed` — no circle satisfies the constraints (including the arc
  completeness check, which is now enforced *during* RANSAC via
  [`data_constraints_met`](@ref) rather than post-hoc).

# Examples
```jldoctest
julia> img = [sqrt((r-25.0)^2+(c-25.0)^2) < 15.0 ? 200.0 : 0.0 for r in 1:50, c in 1:50];

julia> cc = CircleConstraints{Float64}(min_radius=10.0, max_radius=20.0, min_completeness=0.5);

julia> fit = gauge_circle(img, (25.0,25.0), 0.0, 2π, deg2rad(5.0), 25,
                          1.5, 20.0; polarity=POLARITY_NEGATIVE, constraints=cc);

julia> fit isa CircleFit{Float64}
true

julia> abs(fit.r - 15.0) < 2.0
true

julia> fit.rms < 2.0
true
```
"""
function gauge_circle(
    image           :: AbstractMatrix,
    center_rc       :: NTuple{2,Real},
    start_angle     :: Real,
    angular_span    :: Real,
    spacing_radians :: Real,
    profile_length  :: Int,
    sigma           :: Real,
    threshold       :: Real;
    polarity         :: EdgePolarity           = POLARITY_ANY,
    selector         :: EdgeSelector           = SELECT_FIRST,
    constraints      :: CircleConstraints      = CircleConstraints{Float64}(),
    confidence       :: Real                   = 0.99,
    inlier_threshold :: Real                   = 1.0,
    max_iter         :: Int                    = 10_000,
    refine           :: Bool                   = false,
) :: CircleFit{Float64}

    edge_pts = gauge_circular_edge_points_info(image, center_rc,
                   start_angle, angular_span, spacing_radians, profile_length,
                   sigma, threshold, polarity, selector)
    pts = [(e.x, e.y) for e in edge_pts]
    length(pts) < 3 && throw(GaugingError(:too_few_points,
        "gauge_circle: too few edge points detected ($(length(pts)))"))

    model, inliers, outliers = ransac2(pts, CircleModel, inlier_threshold, constraints;
                                       min_inliers=constraints.min_inlier_count,
                                       confidence=confidence, max_iter=max_iter)

    isnothing(model) && throw(GaugingError(:ransac_failed,
        "gauge_circle: RANSAC could not find a valid circle"))

    if refine
        cx_lm, cy_lm, r_lm = fit_circle_lm(pts[inliers];
                                           cx0=model.cx, cy0=model.cy, r0=model.r)
        model = CircleModel{Float64}(cx_lm, cy_lm, r_lm)
    end

    rms = rms_error(model, pts[inliers])
    return CircleFit{Float64}(model.cx, model.cy, model.r, inliers, outliers, rms)
end

# ── Internal helpers ──────────────────────────────────────────────────────────

"""Flatten a vector-of-vectors of ImageEdge into (x,y) tuples."""
function _flatten_to_xy(strip_results::Vector{Vector{ImageEdge{Float64}}})
    pts = Tuple{Float64,Float64}[]
    for strip in strip_results
        for e in strip
            push!(pts, (e.x, e.y))
        end
    end
    return pts
end
