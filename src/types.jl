"""
Shared enumerations, result structs, and constraint structs for Edges.
"""

# ── Enumerations ────────────────────────────────────────────────────────────

"""
    EdgePolarity

Specifies which gradient direction constitutes a detectable edge.

| Value               | Meaning                                     |
|:------------------- |:------------------------------------------- |
| `POLARITY_POSITIVE` | Dark-to-bright (rising) transitions only.   |
| `POLARITY_NEGATIVE` | Bright-to-dark (falling) transitions only.  |
| `POLARITY_ANY`      | Both rising and falling transitions.         |

# Examples
```jldoctest
julia> POLARITY_POSITIVE isa EdgePolarity
true

julia> length(instances(EdgePolarity))
3
```
"""
@enum EdgePolarity begin
    POLARITY_POSITIVE   # rising edge: dark → bright
    POLARITY_NEGATIVE   # falling edge: bright → dark
    POLARITY_ANY        # both rising and falling
end

"""
    EdgeSelector

Controls how many edges are returned from a single 1-D profile scan.

| Value          | Meaning                                              |
|:-------------- |:---------------------------------------------------- |
| `SELECT_FIRST` | Return only the first edge (leftmost / topmost).     |
| `SELECT_LAST`  | Return only the last edge (rightmost / bottommost).  |
| `SELECT_BEST`  | Return only the strongest edge (highest gradient).   |
| `SELECT_ALL`   | Return all edges above the strength threshold.       |

# Examples
```jldoctest
julia> SELECT_ALL isa EdgeSelector
true

julia> length(instances(EdgeSelector))
4
```
"""
@enum EdgeSelector begin
    SELECT_FIRST
    SELECT_LAST
    SELECT_BEST     # edge with the highest strength
    SELECT_ALL
end

"""
    InterpolationMode

Selects how an image is sampled at sub-pixel positions when extracting profiles
along arbitrary geometric paths.

| Value             | Method                | Sub-pixel quality                              |
|:----------------- |:--------------------- |:---------------------------------------------- |
| `INTERP_NEAREST`  | Piecewise constant    | Returns the nearest pixel value (no blending). |
| `INTERP_BILINEAR` | 2×2 linear blend      | C0 continuous; small bias at pixel boundaries. |
| `INTERP_BICUBIC`  | 4×4 cubic B-spline    | C2 continuous; best sub-pixel edge stability.  |

`INTERP_BICUBIC` is the recommended default for sub-pixel edge detection
because the gradient profile remains smooth across pixel boundaries, which
keeps the parabolic sub-pixel fit unbiased.

# Coordinate convention
Integer indices lie at **pixel centres**: `image[i, j]` is the value at
`(row, col) = (i, j)`, and the pixel itself spatially covers
`(i-0.5, j-0.5)` … `(i+0.5, j+0.5)`. The full image extends from
`(0.5, 0.5)` to `(nrows+0.5, ncols+0.5)`. Sample positions outside that range
yield `NaN`.

# Examples
```jldoctest
julia> INTERP_BICUBIC isa InterpolationMode
true

julia> length(instances(InterpolationMode))
3
```
"""
@enum InterpolationMode begin
    INTERP_NEAREST     # piecewise constant
    INTERP_BILINEAR    # 2×2 linear blend (C0)
    INTERP_BICUBIC     # 4×4 cubic B-spline (C2)
end

"""
    ScanOrientation

Primary scan direction for 2-D image edge detection.

| Value            | Profiles run along… | Scan moves…   |
|:---------------- |:------------------- |:------------- |
| `LEFT_TO_RIGHT`  | Rows (horizontal)   | Left to right |
| `RIGHT_TO_LEFT`  | Rows (horizontal)   | Right to left |
| `TOP_TO_BOTTOM`  | Columns (vertical)  | Top to bottom |
| `BOTTOM_TO_TOP`  | Columns (vertical)  | Bottom to top |

# Examples
```jldoctest
julia> LEFT_TO_RIGHT isa ScanOrientation
true

julia> length(instances(ScanOrientation))
4
```
"""
@enum ScanOrientation begin
    LEFT_TO_RIGHT
    RIGHT_TO_LEFT
    TOP_TO_BOTTOM
    BOTTOM_TO_TOP
end

# ── 1D / Profile results ─────────────────────────────────────────────────────

"""
    EdgeResult{T<:AbstractFloat}

A single edge detected in a 1-D intensity profile.

# Fields
- `position::T`: Sub-pixel position along the profile (1-based index units).
  Determined by fitting a parabola to the local gradient extremum.
- `strength::T`: Absolute gradient value at the extremum — proportional to
  the contrast of the edge.

# Examples
```jldoctest
julia> e = EdgeResult{Float64}(12.3, 95.0);

julia> e.position
12.3

julia> e.strength
95.0
```
"""
struct EdgeResult{T<:AbstractFloat}
    position::T    # subpixel position along the profile (1-based index units)
    strength::T    # |gradient| value at the extremum
end

"""
    ProfileEdgesResult{T<:AbstractFloat}

Full return value of [`gauge_edges_in_profile`](@ref).

# Fields
- `edges::Vector{EdgeResult{T}}`: Detected edges, ordered by position.
- `smoothed::Vector{T}`: Gaussian-smoothed copy of the input profile.
- `gradient::Vector{T}`: Discrete gradient ([-1,0,1] kernel) of `smoothed`.
"""
struct ProfileEdgesResult{T<:AbstractFloat}
    edges::Vector{EdgeResult{T}}
    smoothed::Vector{T}    # Gaussian-smoothed profile
    gradient::Vector{T}    # [-1,0,1] derivative of the smoothed profile
end

# ── 2D image edge ─────────────────────────────────────────────────────────────

"""
    ImageEdge{T<:AbstractFloat}

A single edge detected in a 2-D image, returned by [`gauge_edges_info`](@ref)
and related scanning functions.

# Fields
- `x::T`: Column coordinate (horizontal, 1-based).
- `y::T`: Row coordinate (vertical, 1-based).
- `strength::T`: Absolute gradient value at the edge — indicates contrast.
- `scan_index::Union{Int,Nothing}`: Index of the row or column profile that
  produced this edge (useful for tracing which strip produced each point), or
  `nothing` for scans that do not have a meaningful strip index (e.g. radial
  scans produced by [`gauge_circular_edge_points_info`](@ref) and
  [`gauge_ring_edge_points_info`](@ref)).

# Examples
```jldoctest
julia> e = ImageEdge{Float64}(40.5, 12.0, 88.0, 3);

julia> e.x
40.5

julia> e.scan_index
3

julia> radial = ImageEdge{Float64}(40.5, 12.0, 88.0, nothing);

julia> isnothing(radial.scan_index)
true
```
"""
struct ImageEdge{T<:AbstractFloat}
    x::T
    y::T
    strength::T
    scan_index::Union{Int,Nothing}   # nothing for radial scans
end

# ── Geometric fit results ─────────────────────────────────────────────────────

"""
    LineFit{T<:AbstractFloat}

Result of [`gauge_line`](@ref): a RANSAC-fitted infinite line.

The line is represented in normalised implicit form `Ax + By + C = 0`
(A² + B² = 1), so the perpendicular distance from any point `(x, y)` to the
line equals `|Ax + By + C|` directly.

# Fields
- `A`, `B`, `C`: Normalised line coefficients (A² + B² = 1).
- `inliers::Vector{Int}`: Indices into the edge-point array of inlier points.
- `outliers::Vector{Int}`: Indices of outlier points.
- `rms::T`: Root-mean-square perpendicular distance of inliers from the line.

# Examples
```jldoctest
julia> f = LineFit{Float64}(0.0, 1.0, -5.0, [1,2,3], [4], 0.02);

julia> f.A^2 + f.B^2     # normalised
1.0

julia> f.rms
0.02
```
"""
struct LineFit{T<:AbstractFloat}
    A::T; B::T; C::T
    inliers::Vector{Int}
    outliers::Vector{Int}
    rms::T
end

"""
    CircleFit{T<:AbstractFloat}

Result of [`gauge_circle`](@ref): a RANSAC-fitted circle.

# Fields
- `cx`, `cy`: Centre coordinates (column, row; 1-based).
- `r`: Radius in pixels.
- `inliers::Vector{Int}`: Indices of inlier edge points.
- `outliers::Vector{Int}`: Indices of outlier edge points.
- `rms::T`: Root-mean-square radial distance of inliers from the circle.

# Examples
```jldoctest
julia> f = CircleFit{Float64}(100.0, 100.0, 45.0, collect(1:60), Int[], 0.3);

julia> f.r
45.0

julia> isempty(f.outliers)
true
```
"""
struct CircleFit{T<:AbstractFloat}
    cx::T; cy::T; r::T
    inliers::Vector{Int}
    outliers::Vector{Int}
    rms::T
end

# ── RANSAC constraint structs ─────────────────────────────────────────────────

"""
    LineConstraints{T<:AbstractFloat}

Geometric constraints applied during RANSAC line fitting (used by
[`gauge_line`](@ref) and [`ransac2`](@ref)).

All fields have keyword-argument constructors with the defaults shown.

| Field              | Default | Meaning                                             |
|:------------------ |:------- |:--------------------------------------------------- |
| `min_angle`        | `-π/2`  | Minimum allowed line orientation angle (radians).   |
| `max_angle`        | `π/2`   | Maximum allowed line orientation angle (radians).   |
| `min_inlier_ratio` | `0.1`   | Minimum fraction of points that must be inliers.    |
| `min_inlier_count` | `2`     | Absolute minimum number of inlier points required.  |

The angle is the orientation of the line, measured from the positive x-axis
and normalised to `[-π/2, π/2]`.  A horizontal line has angle 0; a vertical
line has angle `±π/2`.

# Examples
```jldoctest
julia> c = LineConstraints{Float64}(min_angle=0.0, max_angle=Float64(π/4));

julia> c.min_angle
0.0

julia> c.min_inlier_count   # default
2
```
"""
Base.@kwdef struct LineConstraints{T<:AbstractFloat}
    min_angle::T          = T(-π/2)
    max_angle::T          = T( π/2)
    min_inlier_ratio::T   = T(0.1)
    min_inlier_count::Int = 2
end

"""
    LineSegmentConstraints{T<:AbstractFloat}

Constraints for [`LineSegmentModel`](@ref) RANSAC fitting. Extends
[`LineConstraints`](@ref) with segment-length limits.

| Field              | Default | Meaning                                           |
|:------------------ |:------- |:------------------------------------------------- |
| `min_angle`        | `-π/2`  | Minimum allowed orientation angle (radians).      |
| `max_angle`        | `π/2`   | Maximum allowed orientation angle (radians).      |
| `min_inlier_ratio` | `0.1`   | Minimum inlier fraction.                          |
| `min_inlier_count` | `2`     | Minimum number of inliers.                        |
| `min_length`       | `0.0`   | Minimum projected span of inlier points (pixels). |
| `max_length`       | `Inf`   | Maximum projected span (pixels).                  |
"""
Base.@kwdef struct LineSegmentConstraints{T<:AbstractFloat}
    min_angle::T           = T(-π/2)
    max_angle::T           = T( π/2)
    min_inlier_ratio::T    = T(0.1)
    min_inlier_count::Int  = 2
    min_length::T          = T(0.0)
    max_length::T          = T(Inf)
end

"""
    CircleConstraints{T<:AbstractFloat}

Geometric constraints applied during RANSAC circle fitting (used by
[`gauge_circle`](@ref) and [`ransac2`](@ref)).

| Field              | Default | Meaning                                           |
|:------------------ |:------- |:------------------------------------------------- |
| `min_radius`       | `0.0`   | Minimum allowed circle radius (pixels).           |
| `max_radius`       | `Inf`   | Maximum allowed circle radius (pixels).           |
| `min_completeness` | `0.1`   | Minimum arc completeness fraction (0–1).          |
| `min_inlier_count` | `3`     | Absolute minimum number of inlier points.         |

Arc completeness measures what fraction of the full 360° is covered by inlier
points, computed in 30° sectors by [`arc_completeness`](@ref).

# Examples
```jldoctest
julia> c = CircleConstraints{Float64}(min_radius=10.0, max_radius=50.0);

julia> c.min_radius
10.0

julia> c.min_completeness   # default
0.1
```
"""
Base.@kwdef struct CircleConstraints{T<:AbstractFloat}
    min_radius::T          = T(0.0)
    max_radius::T          = T(Inf)
    min_completeness::T    = T(0.1)   # fraction of arc that must be covered
    min_inlier_count::Int  = 3
end

# ── Gauging error ─────────────────────────────────────────────────────────────

"""
    GaugingError(reason, msg) <: Exception

Raised by [`gauge_line`](@ref) and [`gauge_circle`](@ref) when a gauging
pipeline cannot produce a valid fit.  The `reason` field is a `Symbol` so
callers can branch on the failure mode programmatically.

# Fields
- `reason::Symbol` — one of:
  - `:too_few_points` — edge detection returned fewer points than the fitter requires.
  - `:ransac_failed` — RANSAC exhausted its iteration budget without finding a
    model that satisfies the constraints.  This includes arc-completeness
    failure for circles (enforced inside RANSAC via [`data_constraints_met`](@ref)).
- `msg::String` — human-readable detail (safe to show to users).

# Examples
```jldoctest
julia> e = GaugingError(:too_few_points, "only 1 point detected");

julia> e.reason
:too_few_points

julia> e isa Exception
true
```
"""
struct GaugingError <: Exception
    reason::Symbol
    msg::String
end

Base.showerror(io::IO, e::GaugingError) =
    print(io, "GaugingError(:", e.reason, "): ", e.msg)
