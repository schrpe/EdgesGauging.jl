"""
Multi-strip and radial edge-point scanning.

Mirrors the C++ functions:
  - `gauge_edge_points_info`          : rectangular multi-strip scan
  - `gauge_circular_edge_points_info` : radial scan from a centre point
  - `gauge_ring_edge_points_info`     : radial scan restricted to an annulus
"""

# Sampling helpers (`_make_interp`, `_sample_radial_profile`,
# `_sample_ring_profile`) live in profiles.jl, which is included before this
# file in the module entry point.

# ── Multi-strip rectangular scan ─────────────────────────────────────────────

"""
    gauge_edge_points_info(image, roi, orientation, spacing, thickness,
                           sigma, threshold,
                           polarity=POLARITY_ANY, selector=SELECT_FIRST)
                           -> Vector{Vector{ImageEdge{Float64}}}

Scan a rectangular ROI with multiple parallel measurement strips and return
the detected edge points per strip.

The ROI is divided into strips whose centres are spaced `spacing` pixels apart
along the scan-perpendicular direction. Each strip is `thickness` pixels wide
and is scanned with [`gauge_edges_info`](@ref).

This two-level structure lets callers inspect per-strip results (e.g. to
detect broken or missing edges) before passing all points to RANSAC.

# Arguments
- `roi`: `(row_start, col_start, row_end, col_end)`, 1-based inclusive.
- `orientation`: scan direction for each strip profile.
- `spacing`: distance between adjacent strip centres (pixels).
- `thickness`: full width of each strip (pixels).
- `sigma`, `threshold`, `polarity`, `selector`: forwarded to
  [`gauge_edges_in_profile`](@ref) for each row or column profile.

# Returns
One inner `Vector{ImageEdge{Float64}}` per strip, in scan order along the
perpendicular axis.

# Examples
```jldoctest
julia> img = [col < 5 ? 0.0 : 100.0 for _ in 1:9, col in 1:8];

julia> strips = gauge_edge_points_info(img, (1,1,9,8), LEFT_TO_RIGHT,
                                       3.0, 1, 0.0, 5.0,
                                       POLARITY_POSITIVE, SELECT_FIRST);

julia> length(strips) >= 3
true

julia> all(!isempty(s) for s in strips)
true

julia> all(all(e.x == 4.5 for e in s) for s in strips)
true
```
"""
function gauge_edge_points_info(
    image       :: AbstractMatrix,
    roi         :: NTuple{4,Int},
    orientation :: ScanOrientation,
    spacing     :: Real,
    thickness   :: Int,
    sigma       :: Real,
    threshold   :: Real,
    polarity    :: EdgePolarity = POLARITY_ANY,
    selector    :: EdgeSelector = SELECT_FIRST,
) :: Vector{Vector{ImageEdge{Float64}}}

    r1, c1, r2, c2 = _clamp_roi(roi, size(image))
    half = max(1, thickness ÷ 2)

    results = Vector{ImageEdge{Float64}}[]

    if orientation == LEFT_TO_RIGHT || orientation == RIGHT_TO_LEFT
        # Strips run parallel to columns, scanned across rows
        centre_row = Float64(r1)
        while centre_row <= Float64(r2)
            strip_r1 = clamp(round(Int, centre_row - half), r1, r2)
            strip_r2 = clamp(round(Int, centre_row + half), r1, r2)
            strip_roi = (strip_r1, c1, strip_r2, c2)
            push!(results, gauge_edges_info(image, strip_roi, orientation,
                                            sigma, threshold, polarity, selector))
            centre_row += spacing
        end
    else
        # Strips run parallel to rows, scanned across columns
        centre_col = Float64(c1)
        while centre_col <= Float64(c2)
            strip_c1 = clamp(round(Int, centre_col - half), c1, c2)
            strip_c2 = clamp(round(Int, centre_col + half), c1, c2)
            strip_roi = (r1, strip_c1, r2, strip_c2)
            push!(results, gauge_edges_info(image, strip_roi, orientation,
                                            sigma, threshold, polarity, selector))
            centre_col += spacing
        end
    end

    return results
end

# ── Circular (radial) scan ────────────────────────────────────────────────────

"""
    gauge_circular_edge_points_info(image, center, start_angle, angular_span,
                                    spacing_radians, profile_length,
                                    sigma, threshold,
                                    polarity=POLARITY_ANY, selector=SELECT_FIRST;
                                    threaded=false, interp=INTERP_BICUBIC)
                                    -> Vector{ImageEdge{Float64}}

Detect edge points by casting radial profiles from `center_rc` at evenly-spaced
angles and calling [`gauge_edges_in_profile`](@ref) on each profile.

Profiles are sampled using the chosen [`InterpolationMode`](@ref) (bicubic by
default for best sub-pixel stability); the interpolant is built once and
reused across all rays.  Samples that fall outside the image yield `NaN`,
which `gauge_edges_in_profile` handles via NaN-aware smoothing.

# Arguments
- `center_rc`: `(row, col)` of the scan centre (1-based). The `_rc` suffix is
  a reminder that this is **(row, col)**, not `(x, y)` — detected edges in the
  returned `ImageEdge` are exposed as `(x=col, y=row)` for Cartesian use.
- `start_angle`: angle of the first ray in radians (`0` = rightward / +col
  direction).
- `angular_span`: total angular range to cover (radians). Use `2π` for a full
  360° scan.
- `spacing_radians`: angular step between consecutive rays. Returns an empty
  vector when `spacing_radians == 0`.
- `profile_length`: number of pixels sampled along each ray.
- `sigma`, `threshold`, `polarity`, `selector`: forwarded to
  [`gauge_edges_in_profile`](@ref).
- `interp`: pixel interpolation method — [`INTERP_NEAREST`](@ref),
  [`INTERP_BILINEAR`](@ref), or [`INTERP_BICUBIC`](@ref) (default).

# Returns
All detected edge points across all rays, collected into a single flat vector
in angular order.

# Examples
```jldoctest
julia> img = [sqrt((r-10.0)^2+(c-10.0)^2) < 5.0 ? 100.0 : 0.0 for r in 1:20, c in 1:20];

julia> edges = gauge_circular_edge_points_info(img, (10.0,10.0),
                   0.0, 2π, deg2rad(30.0), 15, 0.5, 5.0,
                   POLARITY_NEGATIVE, SELECT_FIRST);

julia> length(edges) >= 8
true
```

```jldoctest
julia> img = ones(20, 20);

julia> isempty(gauge_circular_edge_points_info(img, (10.0,10.0), 0.0, 2π, 0.0, 10, 1.0, 1.0))
true
```
"""
function gauge_circular_edge_points_info(
    image           :: AbstractMatrix,
    center_rc       :: NTuple{2,Real},
    start_angle     :: Real,
    angular_span    :: Real,
    spacing_radians :: Real,
    profile_length  :: Int,
    sigma           :: Real,
    threshold       :: Real,
    polarity        :: EdgePolarity = POLARITY_ANY,
    selector        :: EdgeSelector = SELECT_FIRST;
    threaded        :: Bool             = false,
    interp          :: InterpolationMode = INTERP_BICUBIC,
) :: Vector{ImageEdge{Float64}}

    extp = _make_interp(image, interp)
    row_c, col_c = Float64(center_rc[1]), Float64(center_rc[2])

    edges = ImageEdge{Float64}[]
    angle_start = Float64(start_angle)
    span        = Float64(angular_span)
    step        = abs(Float64(spacing_radians))
    step == 0.0 && return edges

    # Precompute the ray angles up-front so the loop iterates over a fixed
    # count (required for `Threads.@threads`).  Generate them with the same
    # while-loop arithmetic as the original serial code so floating-point
    # accumulation rounds identically.
    angles = Float64[]
    angle_stop = angle_start + span + step / 2
    a = angle_start
    while a <= angle_stop
        push!(angles, a)
        a += step
    end
    n_rays = length(angles)

    if threaded
        buckets = Vector{Vector{ImageEdge{Float64}}}(undef, n_rays)
        Threads.@threads for k in 1:n_rays
            angle = angles[k]
            profile = _sample_radial_profile(extp, row_c, col_c, angle, profile_length)
            bucket = ImageEdge{Float64}[]
            for e in _gauge_edges_in_profile(profile, sigma, threshold, polarity, selector)
                r = row_c + (e.position - 1.0) * sin(angle)
                c = col_c + (e.position - 1.0) * cos(angle)
                push!(bucket, ImageEdge{Float64}(c, r, e.strength, nothing))
            end
            buckets[k] = bucket
        end
        for b in buckets
            append!(edges, b)
        end
        return edges
    end

    for k in 1:n_rays
        angle = angles[k]
        profile = _sample_radial_profile(extp, row_c, col_c, angle, profile_length)
        for e in _gauge_edges_in_profile(profile, sigma, threshold, polarity, selector)
            r = row_c + (e.position - 1.0) * sin(angle)
            c = col_c + (e.position - 1.0) * cos(angle)
            push!(edges, ImageEdge{Float64}(c, r, e.strength, nothing))
        end
    end

    return edges
end

# ── Ring (annular) scan ───────────────────────────────────────────────────────

"""
    gauge_ring_edge_points_info(image, center, inner_radius, outer_radius,
                                start_angle, angular_span, spacing_radians,
                                sigma, threshold,
                                polarity=POLARITY_ANY, selector=SELECT_FIRST;
                                threaded=false, interp=INTERP_BICUBIC)
                                -> Vector{ImageEdge{Float64}}

Like [`gauge_circular_edge_points_info`](@ref) but limits each radial profile
to the annular region between `inner_radius` and `outer_radius` pixels from
`center_rc`.

This is useful when the feature of interest lies within a known distance range
from a reference point, and avoids false detections from structures inside the
inner boundary.

Pass `interp` to choose the pixel interpolation method ([`INTERP_NEAREST`](@ref),
[`INTERP_BILINEAR`](@ref), or [`INTERP_BICUBIC`](@ref) — default).  Samples
that fall outside the image yield `NaN` and are handled by the NaN-aware
edge-detection pipeline.

Throws `ArgumentError` if `inner_radius >= outer_radius`.

# Examples
```jldoctest
julia> img = [sqrt((r-10.0)^2+(c-10.0)^2) < 5.0 ? 100.0 : 0.0 for r in 1:20, c in 1:20];

julia> edges = gauge_ring_edge_points_info(img, (10.0,10.0), 3.0, 8.0,
                   0.0, 2π, deg2rad(30.0), 0.5, 5.0,
                   POLARITY_NEGATIVE, SELECT_FIRST);

julia> length(edges) >= 8
true
```

```jldoctest
julia> img = ones(20, 20);

julia> gauge_ring_edge_points_info(img, (10.0,10.0), 5.0, 3.0,
                                   0.0, 2π, deg2rad(30.0), 1.0, 1.0)
ERROR: ArgumentError: inner_radius must be less than outer_radius
[...]
```
"""
function gauge_ring_edge_points_info(
    image           :: AbstractMatrix,
    center_rc       :: NTuple{2,Real},
    inner_radius    :: Real,
    outer_radius    :: Real,
    start_angle     :: Real,
    angular_span    :: Real,
    spacing_radians :: Real,
    sigma           :: Real,
    threshold       :: Real,
    polarity        :: EdgePolarity = POLARITY_ANY,
    selector        :: EdgeSelector = SELECT_FIRST;
    threaded        :: Bool             = false,
    interp          :: InterpolationMode = INTERP_BICUBIC,
) :: Vector{ImageEdge{Float64}}

    inner_radius >= outer_radius &&
        throw(ArgumentError("inner_radius must be less than outer_radius"))

    extp      = _make_interp(image, interp)
    row_c     = Float64(center_rc[1])
    col_c     = Float64(center_rc[2])
    r_inner   = Float64(inner_radius)
    r_outer   = Float64(outer_radius)
    n_samples = max(1, round(Int, r_outer - r_inner))

    edges = ImageEdge{Float64}[]
    angle_start = Float64(start_angle)
    span        = Float64(angular_span)
    step        = abs(Float64(spacing_radians))
    step == 0.0 && return edges

    # Precompute ray angles up-front to support `Threads.@threads`.  Uses the
    # same while-loop arithmetic as the original serial code to preserve
    # floating-point equivalence.
    angles = Float64[]
    angle_stop = angle_start + span + step / 2
    a = angle_start
    while a <= angle_stop
        push!(angles, a)
        a += step
    end
    n_rays = length(angles)

    if threaded
        buckets = Vector{Vector{ImageEdge{Float64}}}(undef, n_rays)
        Threads.@threads for k in 1:n_rays
            angle = angles[k]
            profile = _sample_ring_profile(extp, row_c, col_c, angle,
                                           r_inner, r_outer, n_samples)
            bucket = ImageEdge{Float64}[]
            for e in _gauge_edges_in_profile(profile, sigma, threshold, polarity, selector)
                r_dist = r_inner + (e.position - 1.0) * (r_outer - r_inner) / (n_samples - 1)
                r = row_c + r_dist * sin(angle)
                c = col_c + r_dist * cos(angle)
                push!(bucket, ImageEdge{Float64}(c, r, e.strength, nothing))
            end
            buckets[k] = bucket
        end
        for b in buckets
            append!(edges, b)
        end
        return edges
    end

    for k in 1:n_rays
        angle = angles[k]
        profile = _sample_ring_profile(extp, row_c, col_c, angle,
                                       r_inner, r_outer, n_samples)
        for e in _gauge_edges_in_profile(profile, sigma, threshold, polarity, selector)
            # profile index 1 corresponds to inner_radius
            r_dist = r_inner + (e.position - 1.0) * (r_outer - r_inner) / (n_samples - 1)
            r = row_c + r_dist * sin(angle)
            c = col_c + r_dist * cos(angle)
            push!(edges, ImageEdge{Float64}(c, r, e.strength, nothing))
        end
    end

    return edges
end

# Sampling helpers (`_sample_radial_profile`, `_sample_ring_profile`,
# `_make_interp`) live in profiles.jl, included before this file.
# `_clamp_roi` is defined in image_edges.jl (loaded first by the module).
