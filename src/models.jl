"""
Concrete model types for RANSAC fitting.

Each model type `M` must implement the following interface so that the generic
[`ransac`](@ref) / [`ransac2`](@ref) engines can use it:

| Function                          | Returns      | Purpose                        |
|:--------------------------------- |:------------ |:------------------------------ |
| `sample_size(::Type{M})`          | `Int`        | Minimum points to fit model    |
| `fit_model(::Type{M}, pts)`       | `M`          | Fit model to a point sample    |
| `point_distance(m::M, pt)`        | `Float64`    | Point-to-model distance        |
| `constraints_met(m::M, c)`        | `Bool`       | Validate geometric constraints |

Plain functions (not abstract-type methods) let Julia specialise each method
at compile time for the concrete model — no runtime dispatch overhead.
"""

using LinearAlgebra: norm

# ── LineModel ─────────────────────────────────────────────────────────────────

"""
    LineModel(A, B, C)

Fitted infinite line `Ax + By + C = 0` (A² + B² = 1), as used inside RANSAC.

The normalisation A² + B² = 1 means `point_distance` is the true perpendicular
distance from the point to the line.  Use [`LineConstraints`](@ref) to restrict
allowed orientations when calling [`ransac2`](@ref).

See also [`fit_line_tls`](@ref), [`gauge_line`](@ref).

# Examples
```jldoctest
julia> m = LineModel(0.0, 1.0, -5.0);   # y = 5

julia> point_distance(m, (3.0, 5.0))    # on the line
0.0

julia> point_distance(m, (0.0, 3.0))    # 2 pixels below
2.0

julia> constraints_met(m, LineConstraints{Float64}())
true
```
"""
struct LineModel{T<:AbstractFloat}
    A::T; B::T; C::T
end

# Julia auto-generates `LineModel(::T, ::T, ::T) where T<:AbstractFloat`; this
# extra method promotes mixed-type calls like `LineModel(1, 2.0, 3)`.
LineModel(A, B, C) = LineModel(promote(float(A), float(B), float(C))...)

"""
    sample_size(::Type{M}) -> Int

Return the minimum number of points required to fit model type `M` uniquely.

| Model type         | Minimum points |
|:------------------ |:-------------- |
| `LineModel`        | 2              |
| `LineSegmentModel` | 2              |
| `CircleModel`      | 3              |

# Examples
```jldoctest
julia> sample_size(LineModel)
2

julia> sample_size(CircleModel)
3
```
"""
sample_size(::Type{<:LineModel}) = 2

"""
    fit_model(::Type{M}, pts) -> M

Fit model type `M` to the point collection `pts` and return the fitted model.

Each concrete model delegates to its dedicated low-level fitting function:
- `LineModel` / `LineSegmentModel` → [`fit_line_tls`](@ref)
- `CircleModel` → [`fit_circle_taubin`](@ref)

Throws if `pts` contains fewer points than [`sample_size`](@ref)`(M)` requires,
or if the point configuration is degenerate.

# Examples
```jldoctest
julia> pts = [(x, 2.0*x) for x in 1.0:5.0];

julia> m = fit_model(LineModel, pts);

julia> m isa LineModel
true

julia> all(point_distance(m, p) < 1e-10 for p in pts)
true
```
"""
# Default element type is Float64 when no parameter is supplied.
fit_model(::Type{LineModel}, pts::AbstractVector) = fit_model(LineModel{Float64}, pts)

function fit_model(::Type{LineModel{T}}, pts::AbstractVector) where {T<:AbstractFloat}
    A, B, C = fit_line_tls(pts)
    return LineModel{T}(T(A), T(B), T(C))
end

"""
    point_distance(m, pt) -> Float64

Return the geometric distance from point `pt` (a length-2 indexable) to
model `m`.

- **`LineModel`**: perpendicular distance `|Ax + By + C|` (denominator is 1
  because the normal is normalised).
- **`LineSegmentModel`**: same perpendicular distance as `LineModel`.
- **`CircleModel`**: `|√((x−cx)²+(y−cy)²) − r|`, i.e. radial distance from
  the circle boundary.

# Examples
```jldoctest
julia> m = CircleModel(0.0, 0.0, 10.0);

julia> point_distance(m, (10.0, 0.0))   # exactly on the circle
0.0

julia> point_distance(m, (7.0, 0.0))    # 3 pixels inside
3.0

julia> point_distance(m, (13.0, 0.0))   # 3 pixels outside
3.0
```
"""
function point_distance(m::LineModel, pt)
    # Promotion lets `point_distance(::LineModel{Float32}, ::Tuple{Float32,Float32})`
    # stay in Float32 end-to-end.  Denominator is 1 because A² + B² = 1.
    return abs(m.A * pt[1] + m.B * pt[2] + m.C)
end

"""
    constraints_met(m, c) -> Bool

Return `true` when the model `m` satisfies all constraints in `c`.

| Model / Constraint pair                         | Checks                         |
|:----------------------------------------------- |:------------------------------ |
| `LineModel` + `LineConstraints`                 | Orientation angle within range |
| `LineSegmentModel` + `LineSegmentConstraints`   | Same + segment length          |
| `CircleModel` + `CircleConstraints`             | Radius within `[min, max]`     |

# Examples
```jldoctest
julia> m = CircleModel(0.0, 0.0, 25.0);

julia> constraints_met(m, CircleConstraints{Float64}(min_radius=10.0, max_radius=50.0))
true

julia> constraints_met(m, CircleConstraints{Float64}(min_radius=30.0, max_radius=50.0))
false
```
"""
function constraints_met(m::LineModel, c::LineConstraints)
    angle = atan(m.B, m.A)   # angle of the line normal; line angle = angle ± π/2
    line_angle = angle + π/2
    # Normalise to [-π/2, π/2]
    while line_angle >  π/2; line_angle -= π; end
    while line_angle < -π/2; line_angle += π; end
    return c.min_angle <= line_angle <= c.max_angle
end

# ── LineSegmentModel ──────────────────────────────────────────────────────────

"""
    LineSegmentModel(A, B, C)

Fitted line segment — identical algebraic representation to [`LineModel`](@ref)
but paired with [`LineSegmentConstraints`](@ref) to also enforce limits on the
projected span of inlier points via [`segment_length`](@ref).

# Examples
```jldoctest
julia> pts = [(Float64(x), 0.0) for x in 1:10];

julia> A, B, C = fit_line_tls(pts);

julia> m = LineSegmentModel(A, B, C);

julia> round(segment_length(m, pts), digits=10)
9.0
```
"""
struct LineSegmentModel{T<:AbstractFloat}
    A::T; B::T; C::T
end

LineSegmentModel(A, B, C) = LineSegmentModel(promote(float(A), float(B), float(C))...)

sample_size(::Type{<:LineSegmentModel}) = 2

fit_model(::Type{LineSegmentModel}, pts::AbstractVector) = fit_model(LineSegmentModel{Float64}, pts)

function fit_model(::Type{LineSegmentModel{T}}, pts::AbstractVector) where {T<:AbstractFloat}
    A, B, C = fit_line_tls(pts)
    return LineSegmentModel{T}(T(A), T(B), T(C))
end

point_distance(m::LineSegmentModel, pt) = abs(m.A * pt[1] + m.B * pt[2] + m.C)

function constraints_met(m::LineSegmentModel, c::LineSegmentConstraints)
    angle = atan(m.B, m.A) + π/2
    while angle >  π/2; angle -= π; end
    while angle < -π/2; angle += π; end
    return c.min_angle <= angle <= c.max_angle
end

"""
    segment_length(m::LineSegmentModel, inlier_pts) -> Float64

Return the projected span of `inlier_pts` along the line direction of `m`.

This is the length of the smallest segment of the fitted line that contains
all inlier projections — i.e. `max(proj) − min(proj)` where the projection
is taken along the line's tangent direction `(-B, A)`.

Returns `0.0` for an empty inlier set.

# Examples
```jldoctest
julia> m = LineSegmentModel(0.0, 1.0, -3.0);   # horizontal line y = 3

julia> pts = [(1.0, 3.0), (5.0, 3.0), (9.0, 3.0)];

julia> segment_length(m, pts)
8.0

julia> segment_length(m, [])
0.0
```
"""
function segment_length(m::LineSegmentModel, inlier_pts::AbstractVector)
    isempty(inlier_pts) && return 0.0
    # Direction along the line (perpendicular to normal)
    dx, dy = -m.B, m.A
    projs = [dx * pt[1] + dy * pt[2] for pt in inlier_pts]
    return maximum(projs) - minimum(projs)
end

# ── CircleModel ───────────────────────────────────────────────────────────────

"""
    CircleModel(cx, cy, r)

Fitted circle with centre `(cx, cy)` and radius `r`.

Use [`CircleConstraints`](@ref) to enforce radius and arc-completeness limits
when calling [`ransac2`](@ref).

See also [`fit_circle_taubin`](@ref), [`gauge_circle`](@ref),
[`arc_completeness`](@ref).

# Examples
```jldoctest
julia> m = CircleModel(0.0, 0.0, 10.0);

julia> point_distance(m, (10.0, 0.0))
0.0

julia> constraints_met(m, CircleConstraints{Float64}(min_radius=5.0, max_radius=15.0))
true

julia> constraints_met(m, CircleConstraints{Float64}(min_radius=11.0, max_radius=20.0))
false
```
"""
struct CircleModel{T<:AbstractFloat}
    cx::T; cy::T; r::T
end

CircleModel(cx, cy, r) = CircleModel(promote(float(cx), float(cy), float(r))...)

sample_size(::Type{<:CircleModel}) = 3

fit_model(::Type{CircleModel}, pts::AbstractVector) = fit_model(CircleModel{Float64}, pts)

function fit_model(::Type{CircleModel{T}}, pts::AbstractVector) where {T<:AbstractFloat}
    cx, cy, r = fit_circle_taubin(pts)
    return CircleModel{T}(T(cx), T(cy), T(r))
end

function point_distance(m::CircleModel, pt)
    dx = pt[1] - m.cx
    dy = pt[2] - m.cy
    return abs(sqrt(dx^2 + dy^2) - m.r)
end

function constraints_met(m::CircleModel, c::CircleConstraints)
    return c.min_radius <= m.r <= c.max_radius
end

"""
    arc_completeness(m::CircleModel, inlier_pts; sector_deg=30.0) -> Float64

Compute what fraction of the full 360° circle boundary is covered by
`inlier_pts`, measured in angular sectors of `sector_deg` degrees.

Returns a value in `[0, 1]` — `1.0` means all sectors contain at least one
inlier; `0.0` means the point set is empty.

Used by [`gauge_circle`](@ref) to reject partial-arc detections that do not
meet the [`CircleConstraints`](@ref) `min_completeness` requirement.

# Examples
```jldoctest
julia> m = CircleModel(0.0, 0.0, 10.0);

julia> pts_full = [(10.0*cos(deg2rad(d)), 10.0*sin(deg2rad(d))) for d in 15:30:345];

julia> arc_completeness(m, pts_full) ≈ 1.0
true

julia> pts_half = [(10.0*cos(deg2rad(d)), 10.0*sin(deg2rad(d))) for d in 15:30:175];

julia> comp = arc_completeness(m, pts_half);

julia> 0.4 <= comp <= 0.6
true

julia> arc_completeness(m, [])
0.0
```
"""
function arc_completeness(m::CircleModel, inlier_pts::AbstractVector;
                          sector_deg::Float64=30.0)
    isempty(inlier_pts) && return 0.0
    n_sectors = round(Int, 360.0 / sector_deg)

    # Fast path: up to 64 sectors fits in a single UInt64 bitmask — no allocation.
    # The default 30° → 12 sectors and any realistic sector_deg ≥ ~6° lands here.
    if n_sectors <= 64
        mask = UInt64(0)
        for pt in inlier_pts
            θ = atan(Float64(pt[2]) - m.cy, Float64(pt[1]) - m.cx)
            θ_deg = mod(rad2deg(θ), 360.0)
            sector = clamp(floor(Int, θ_deg / sector_deg), 0, n_sectors - 1)
            mask |= (UInt64(1) << sector)
        end
        return count_ones(mask) / n_sectors
    end

    occupied = falses(n_sectors)
    for pt in inlier_pts
        θ = atan(Float64(pt[2]) - m.cy, Float64(pt[1]) - m.cx)
        θ_deg = mod(rad2deg(θ), 360.0)
        sector = clamp(floor(Int, θ_deg / sector_deg) + 1, 1, n_sectors)
        occupied[sector] = true
    end
    return sum(occupied) / n_sectors
end

# ── Data-dependent constraints (checked per-candidate inside `ransac2`) ──────

"""
    data_constraints_met(m, c, inlier_pts) -> Bool

Per-candidate constraint check evaluated on a model `m`'s inlier set during
[`ransac2`](@ref).  Unlike [`constraints_met`](@ref), this has access to the
inlier points themselves so constraints that depend on the data (e.g. arc
completeness) can be enforced while RANSAC is still searching, letting it
reject an inlier-majority-but-invalid candidate in favour of a smaller
valid one.

Default implementation returns `true`.  Override for specific `(Model,
Constraint)` pairs as needed.
"""
data_constraints_met(m, c, inlier_pts) = true

"""
Override for circles: require `arc_completeness ≥ c.min_completeness`.
"""
function data_constraints_met(m::CircleModel, c::CircleConstraints, inlier_pts)
    isempty(inlier_pts) && return false
    return arc_completeness(m, inlier_pts) >= c.min_completeness
end

"""
Override for line segments: require `min_length ≤ segment_length ≤ max_length`.
"""
function data_constraints_met(m::LineSegmentModel, c::LineSegmentConstraints, inlier_pts)
    isempty(inlier_pts) && return false
    len = segment_length(m, inlier_pts)
    return c.min_length <= len <= c.max_length
end

# ── RMS error helpers ─────────────────────────────────────────────────────────

"""
    rms_error(m, pts) -> Float64

Return the root-mean-square distance of all points in `pts` from model `m`,
using [`point_distance`](@ref).

Returns `0.0` for an empty point set.

# Examples
```jldoctest
julia> m = CircleModel(0.0, 0.0, 5.0);

julia> pts = [(5.0*cos(θ), 5.0*sin(θ)) for θ in range(0, 2π, length=9)[1:end-1]];

julia> rms_error(m, pts) < 1e-10
true

julia> rms_error(m, [])
0.0
```
"""
function rms_error(m::LineModel, pts::AbstractVector)
    isempty(pts) && return 0.0
    sqrt(sum(point_distance(m, pt)^2 for pt in pts) / length(pts))
end

function rms_error(m::CircleModel, pts::AbstractVector)
    isempty(pts) && return 0.0
    sqrt(sum(point_distance(m, pt)^2 for pt in pts) / length(pts))
end
