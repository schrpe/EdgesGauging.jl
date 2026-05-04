"""
Profile extraction along arbitrary geometric paths.

Provides public functions [`extract_line_profile`](@ref) and
[`extract_arc_profile`](@ref) that sample an image along a straight segment or
a circular arc, with selectable interpolation method (nearest / bilinear /
bicubic), configurable strip width, and explicit `NaN` for out-of-bounds
samples.

Also hosts the shared sampling helpers used by the radial / ring scans in
`edge_points.jl` so all profile-extraction paths share a common interpolant
factory and out-of-bounds policy.

# Coordinate convention
Integer indices lie at **pixel centres** — `image[i, j]` is the value at
`(row, col) = (i, j)`, and the pixel itself spatially covers
`(i-0.5, j-0.5)` … `(i+0.5, j+0.5)`. The full image extends from
`(0.5, 0.5)` to `(nrows+0.5, ncols+0.5)`. Sample positions outside that range
yield `NaN`.
"""

using Interpolations: interpolate, extrapolate, BSpline, Linear,
                      Cubic, Reflect, OnCell

# ── Public API ───────────────────────────────────────────────────────────────

"""
    extract_line_profile(image, p0_rc, p1_rc;
                         width=1, n_samples=0, interp=INTERP_BICUBIC)
                         -> Vector{Float64}

Sample the image along the straight line segment from `p0_rc` to `p1_rc` and
return the resulting 1-D intensity profile.

# Arguments
- `image`: 2-D array (matrix). Any element type convertible to `Float64`.
- `p0_rc`, `p1_rc`: segment endpoints as `(row, col)` tuples (1-based,
  pixel-centre convention; see [`InterpolationMode`](@ref)). May lie outside
  the image — out-of-bounds samples become `NaN`.
- `width::Int = 1`: number of samples taken perpendicular to the segment at
  each centreline position. `1` extracts a slim profile; values > 1 average
  `width` parallel samples spaced 1 px apart and centred on the segment.
- `n_samples::Int = 0`: number of samples along the segment. `0` means
  *auto* — picks `max(2, ⌈hypot(Δrow, Δcol)⌉ + 1)` so the centreline samples
  are spaced ≈ 1 px apart.
- `interp::InterpolationMode = INTERP_BICUBIC`: sampling method —
  [`INTERP_NEAREST`](@ref InterpolationMode), [`INTERP_BILINEAR`](@ref InterpolationMode), or
  [`INTERP_BICUBIC`](@ref InterpolationMode). Bicubic is recommended for sub-pixel edge
  detection because the resulting gradient profile is smooth across pixel
  boundaries.

# Strip aggregation
For `width > 1`, samples at the same centreline position but different
perpendicular offsets are averaged. Out-of-bounds perpendicular samples are
excluded from the mean — the position only becomes `NaN` if **all**
perpendicular samples are outside the image. This keeps strips that graze the
image border usable.

# Returns
A `Vector{Float64}` of length `n_samples` (or the auto-derived count). Entries
where every sample at that centreline position fell outside the image are
`NaN`.

# Examples
```jldoctest
julia> img = Float64[10*r + c for r in 1:5, c in 1:5];

julia> p = extract_line_profile(img, (3.0, 1.0), (3.0, 5.0); width=1, interp=INTERP_NEAREST);

julia> p == [31.0, 32.0, 33.0, 34.0, 35.0]
true
```

```jldoctest
julia> img = fill(7.0, 5, 5);

julia> p = extract_line_profile(img, (3.0, 1.0), (3.0, 5.0); width=3, interp=INTERP_NEAREST);

julia> all(==(7.0), p)
true
```

```jldoctest
julia> img = Float64[10*r + c for r in 1:5, c in 1:5];

julia> p = extract_line_profile(img, (3.0, 4.0), (3.0, 7.0); n_samples=4, interp=INTERP_NEAREST);

julia> p[1:2] == [34.0, 35.0]
true

julia> isnan(p[3]) && isnan(p[4])
true
```
"""
function extract_line_profile(
    image      :: AbstractMatrix,
    p0_rc      :: NTuple{2,Real},
    p1_rc      :: NTuple{2,Real};
    width      :: Int                = 1,
    n_samples  :: Int                = 0,
    interp     :: InterpolationMode  = INTERP_BICUBIC,
) :: Vector{Float64}

    width < 1 && throw(ArgumentError("width must be ≥ 1"))
    n_samples < 0 && throw(ArgumentError("n_samples must be ≥ 0"))

    r0, c0 = Float64(p0_rc[1]), Float64(p0_rc[2])
    r1, c1 = Float64(p1_rc[1]), Float64(p1_rc[2])
    Δr, Δc = r1 - r0, c1 - c0
    L      = hypot(Δr, Δc)

    n = n_samples == 0 ? max(2, ceil(Int, L) + 1) : n_samples

    # Perpendicular unit vector (rotate tangent 90°).  When L == 0 the segment
    # is degenerate; keep perp = (0, 0) so all `width` strip samples coincide
    # at the single point — the result is just that point's value, repeated.
    if L > 0
        perp_dr = -Δc / L
        perp_dc =  Δr / L
    else
        perp_dr = 0.0
        perp_dc = 0.0
    end

    extp    = _make_interp(image, interp)
    offsets = _perpendicular_offsets(width)
    profile = Vector{Float64}(undef, n)

    for k in 1:n
        t   = n == 1 ? 0.0 : (k - 1) / (n - 1)
        r_k = r0 + t * Δr
        c_k = c0 + t * Δc
        profile[k] = _aggregate_strip(extp, r_k, c_k, perp_dr, perp_dc, offsets)
    end
    return profile
end

"""
    extract_arc_profile(image, center_rc, radius, start_angle, end_angle;
                        width=1, n_samples=0, interp=INTERP_BICUBIC)
                        -> Vector{Float64}

Sample the image along a circular arc and return the resulting 1-D intensity
profile.

# Arguments
- `image`: 2-D array (matrix).
- `center_rc`: arc centre as a `(row, col)` tuple (1-based, pixel-centre
  convention).
- `radius`: arc radius in pixels.
- `start_angle`, `end_angle`: angles in **radians**, measured counter-clockwise
  from the +column axis (`0` = pointing right). The arc traverses from
  `start_angle` to `end_angle`; pass `end_angle < start_angle` for a
  clockwise arc.
- `width::Int = 1`: number of samples taken perpendicular to the arc at each
  centreline position. The perpendicular to an arc is the **radial**
  direction, so `width > 1` widens the band radially (inside ↔ outside the
  nominal radius). Samples are 1 px apart and centred on the arc.
- `n_samples::Int = 0`: number of samples along the arc. `0` means *auto* —
  picks `max(2, ⌈|end_angle - start_angle| · radius⌉ + 1)`, giving an
  arc-length-uniform spacing of ≈ 1 px regardless of radius.
- `interp::InterpolationMode = INTERP_BICUBIC`: sampling method.

# Returns
A `Vector{Float64}`. Entries become `NaN` where every perpendicular sample at
that arc position lies outside the image.

# Examples
```jldoctest
julia> img = [hypot(r - 10.0, c - 10.0) < 5.0 ? 100.0 : 0.0 for r in 1:20, c in 1:20];

julia> p = extract_arc_profile(img, (10.0, 10.0), 3.0, 0.0, 2π; n_samples=24, interp=INTERP_NEAREST);

julia> all(==(100.0), p)
true
```

```jldoctest
julia> img = ones(20, 20);

julia> p1 = extract_arc_profile(img, (10.0, 10.0), 3.0, 0.0, π);   # auto-density

julia> p2 = extract_arc_profile(img, (10.0, 10.0), 6.0, 0.0, π);

julia> isapprox(length(p2), 2 * length(p1); atol=2)
true
```
"""
function extract_arc_profile(
    image       :: AbstractMatrix,
    center_rc   :: NTuple{2,Real},
    radius      :: Real,
    start_angle :: Real,
    end_angle   :: Real;
    width       :: Int               = 1,
    n_samples   :: Int               = 0,
    interp      :: InterpolationMode = INTERP_BICUBIC,
) :: Vector{Float64}

    width < 1 && throw(ArgumentError("width must be ≥ 1"))
    n_samples < 0 && throw(ArgumentError("n_samples must be ≥ 0"))
    radius < 0 && throw(ArgumentError("radius must be ≥ 0"))

    row_c   = Float64(center_rc[1])
    col_c   = Float64(center_rc[2])
    rad     = Float64(radius)
    a0      = Float64(start_angle)
    a1      = Float64(end_angle)
    arc_len = abs(a1 - a0) * rad

    n = n_samples == 0 ? max(2, ceil(Int, arc_len) + 1) : n_samples

    extp    = _make_interp(image, interp)
    offsets = _perpendicular_offsets(width)
    profile = Vector{Float64}(undef, n)

    for k in 1:n
        t      = n == 1 ? 0.0 : (k - 1) / (n - 1)
        θ      = a0 + t * (a1 - a0)
        sinθ   = sin(θ)
        cosθ   = cos(θ)
        r_k    = row_c + rad * sinθ
        c_k    = col_c + rad * cosθ
        # Perpendicular to the arc = radial direction.
        profile[k] = _aggregate_strip(extp, r_k, c_k, sinθ, cosθ, offsets)
    end
    return profile
end

# ── Internal helpers (shared with edge_points.jl) ────────────────────────────

"""
    _make_interp(image, mode::InterpolationMode)

Build an interpolant that evaluates to `NaN` outside the image.  For
`INTERP_BILINEAR` and `INTERP_BICUBIC` this returns an
`Interpolations.Extrapolation` wrapping the corresponding B-spline; for
`INTERP_NEAREST` it returns a closure performing manual nearest-pixel lookup
with `floor(Int, r + 0.5)`, which exactly matches the "integer = pixel centre"
convention regardless of `Interpolations.jl` version differences in the
default grid semantics of `BSpline(Constant())`.
"""
function _make_interp(image::AbstractMatrix, mode::InterpolationMode)
    A = Float64.(image)
    if mode == INTERP_NEAREST
        nrows, ncols = size(A)
        # Closure: nearest pixel with pixel-centre semantics; OOB → NaN.
        return (r, c) -> begin
            ri = floor(Int, r + 0.5)
            ci = floor(Int, c + 0.5)
            (ri < 1 || ri > nrows || ci < 1 || ci > ncols) ? NaN : A[ri, ci]
        end
    elseif mode == INTERP_BILINEAR
        itp = interpolate(A, BSpline(Linear()))
        return extrapolate(itp, NaN)
    else  # INTERP_BICUBIC
        itp = interpolate(A, BSpline(Cubic(Reflect(OnCell()))))
        return extrapolate(itp, NaN)
    end
end

"""
    _in_bounds(r, c, nrows, ncols) -> Bool

True when `(r, c)` lies within the geometric extent of the image, i.e.
`0.5 ≤ r ≤ nrows + 0.5` and `0.5 ≤ c ≤ ncols + 0.5`.  Diagnostic helper —
the actual sampling path uses the `_make_interp` wrapper, which already
returns `NaN` outside its valid range.
"""
function _in_bounds(r::Real, c::Real, nrows::Int, ncols::Int) :: Bool
    return 0.5 <= r <= nrows + 0.5 && 0.5 <= c <= ncols + 0.5
end

"""
    _perpendicular_offsets(width) -> Vector{Float64}

Centred perpendicular sample offsets for a strip of `width` samples spaced 1 px
apart.  `width = 1` returns `[0.0]`; `width = 3` returns `[-1.0, 0.0, 1.0]`;
`width = 4` returns `[-1.5, -0.5, 0.5, 1.5]`.
"""
function _perpendicular_offsets(width::Int) :: Vector{Float64}
    width < 1 && throw(ArgumentError("width must be ≥ 1"))
    return [Float64(k) - (width - 1) / 2.0 for k in 0:(width - 1)]
end

"""
    _aggregate_strip(extp, r0, c0, dr, dc, offsets) -> Float64

Sample at `(r0 + off·dr, c0 + off·dc)` for every `off in offsets` and return
the mean of the in-bounds samples.  Returns `NaN` when every sample is `NaN`
(all perpendicular offsets fall outside the image).
"""
function _aggregate_strip(
    extp,
    r0      :: Float64,
    c0      :: Float64,
    dr      :: Float64,
    dc      :: Float64,
    offsets :: Vector{Float64},
) :: Float64
    n_valid = 0
    s = 0.0
    @inbounds for off in offsets
        v = extp(r0 + off * dr, c0 + off * dc)
        if !isnan(v)
            s += v
            n_valid += 1
        end
    end
    return n_valid > 0 ? s / n_valid : NaN
end

"""
    _sample_radial_profile(extp, row_c, col_c, angle, n) -> Vector{Float64}

Sample `n` pixels along a ray from `(row_c, col_c)` at `angle` (radians,
CCW from +col axis), spacing 1 px apart starting at the centre.  Used by
[`gauge_circular_edge_points_info`](@ref).  Out-of-bounds samples are `NaN`
(the `extp` wrapper takes care of that).
"""
function _sample_radial_profile(
    extp,
    row_c :: Float64,
    col_c :: Float64,
    angle :: Float64,
    n     :: Int,
) :: Vector{Float64}
    profile = Vector{Float64}(undef, n)
    sinθ = sin(angle)
    cosθ = cos(angle)
    @inbounds for k in 0:(n - 1)
        r = row_c + k * sinθ
        c = col_c + k * cosθ
        profile[k + 1] = extp(r, c)
    end
    return profile
end

"""
    _sample_ring_profile(extp, row_c, col_c, angle, r_inner, r_outer, n)
        -> Vector{Float64}

Sample `n` pixels along a ray from `(row_c, col_c)` at `angle`, restricted to
the radial range `[r_inner, r_outer]`.  Used by
[`gauge_ring_edge_points_info`](@ref).  Out-of-bounds samples are `NaN`.
"""
function _sample_ring_profile(
    extp,
    row_c   :: Float64,
    col_c   :: Float64,
    angle   :: Float64,
    r_inner :: Float64,
    r_outer :: Float64,
    n       :: Int,
) :: Vector{Float64}
    profile = Vector{Float64}(undef, n)
    sinθ = sin(angle)
    cosθ = cos(angle)
    step = n > 1 ? (r_outer - r_inner) / (n - 1) : 0.0
    @inbounds for k in 0:(n - 1)
        dist = r_inner + k * step
        r = row_c + dist * sinθ
        c = col_c + dist * cosθ
        profile[k + 1] = extp(r, c)
    end
    return profile
end
