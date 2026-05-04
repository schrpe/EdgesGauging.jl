"""
2D image edge detection.

`gauge_edges_info` dispatches over rows or columns of an image ROI, calls
the 1D [`gauge_edges_in_profile`](@ref) for each profile, and converts the
resulting 1D subpixel positions back to 2D image coordinates.

Mirrors the C++ `gauge_edges_info()` function.
"""

# ── Public API ───────────────────────────────────────────────────────────────

"""
    gauge_edges_info(image, roi, orientation, sigma, threshold,
                     polarity=POLARITY_ANY, selector=SELECT_ALL;
                     threaded=false)
                     -> Vector{ImageEdge{Float64}}

Detect edges across all row or column profiles within a rectangular ROI.

For each 1-D profile extracted from the image (one per row for
`LEFT_TO_RIGHT`/`RIGHT_TO_LEFT`, one per column for `TOP_TO_BOTTOM`/
`BOTTOM_TO_TOP`), [`gauge_edges_in_profile`](@ref) is called and the
resulting sub-pixel positions are converted to 2-D image coordinates.

# Arguments
- `image`: 2-D array (matrix). Any element type convertible to `Float64`.
- `roi`: `(row_start, col_start, row_end, col_end)`, 1-based inclusive.
  Values are clamped to image bounds, and `row_start > row_end` is swapped.
- `orientation`: scan direction — one of [`LEFT_TO_RIGHT`](@ref ScanOrientation),
  [`RIGHT_TO_LEFT`](@ref ScanOrientation), [`TOP_TO_BOTTOM`](@ref ScanOrientation),
  [`BOTTOM_TO_TOP`](@ref ScanOrientation).
- `sigma`, `threshold`, `polarity`, `selector`: forwarded to
  [`gauge_edges_in_profile`](@ref) for each extracted profile.
- `threaded`: if `true`, scans are processed in parallel with `Threads.@threads`.
  Requires `Threads.nthreads() > 1` to actually speed up. Output order is
  preserved (edges are collected per-scan into buckets and merged in scan-index
  order at the end).

# Returns
A `Vector{ImageEdge{Float64}}` with `x` = column, `y` = row (1-based), in
scan order (row or column index increases monotonically).

# Examples
```jldoctest
julia> img = [col < 5 ? 0.0 : 100.0 for _ in 1:4, col in 1:8];

julia> edges = gauge_edges_info(img, (1,1,4,8), LEFT_TO_RIGHT, 0.0, 5.0,
                                POLARITY_POSITIVE, SELECT_FIRST);

julia> length(edges)    # one edge per row
4

julia> all(e.x == 4.5 for e in edges)   # step between col 4 and 5
true

julia> edges[1].scan_index
1
```

```jldoctest
julia> img = fill(128.0, 20, 20);

julia> isempty(gauge_edges_info(img, (1,1,20,20), LEFT_TO_RIGHT, 1.0, 1.0))
true
```
"""
function gauge_edges_info(
    image       :: AbstractMatrix,
    roi         :: NTuple{4,Int},
    orientation :: ScanOrientation,
    sigma       :: Real,
    threshold   :: Real,
    polarity    :: EdgePolarity = POLARITY_ANY,
    selector    :: EdgeSelector = SELECT_ALL;
    threaded    :: Bool = false,
) :: Vector{ImageEdge{Float64}}

    r1, c1, r2, c2 = _clamp_roi(roi, size(image))
    sub = @view image[r1:r2, c1:c2]

    edges = ImageEdge{Float64}[]

    if orientation == LEFT_TO_RIGHT || orientation == RIGHT_TO_LEFT
        _scan_rows(sub, r1, c1, orientation, sigma, threshold, polarity, selector, edges, threaded)
    else
        _scan_cols(sub, c1, r1, orientation, sigma, threshold, polarity, selector, edges, threaded)
    end

    return edges
end

# ── Internal helpers ──────────────────────────────────────────────────────────

function _clamp_roi(roi::NTuple{4,Int}, sz::Tuple{Int,Int})
    nrows, ncols = sz
    r1 = clamp(roi[1], 1, nrows)
    c1 = clamp(roi[2], 1, ncols)
    r2 = clamp(roi[3], 1, nrows)
    c2 = clamp(roi[4], 1, ncols)
    r1 > r2 && ((r1, r2) = (r2, r1))
    c1 > c2 && ((c1, c2) = (c2, c1))
    return r1, c1, r2, c2
end

function _scan_rows(
    sub, r1, c1,
    orientation, sigma, threshold, polarity, selector,
    edges, threaded::Bool,
)
    nr_sub, nc_sub = size(sub)

    if threaded
        # Parallel path: per-scan bucket; each thread allocates its own profile
        # buffer (cost amortised by the work in _gauge_edges_in_profile).
        buckets = Vector{Vector{ImageEdge{Float64}}}(undef, nr_sub)
        Threads.@threads for scan_idx in 1:nr_sub
            row_img = r1 + scan_idx - 1
            profile = Vector{Float64}(undef, nc_sub)
            @inbounds for j in 1:nc_sub
                profile[j] = sub[scan_idx, j]
            end
            orientation == RIGHT_TO_LEFT && reverse!(profile)

            bucket = ImageEdge{Float64}[]
            for e in _gauge_edges_in_profile(profile, sigma, threshold, polarity, selector)
                col_in_sub = orientation == RIGHT_TO_LEFT ? (nc_sub + 1 - e.position) : e.position
                col_img    = c1 - 1 + col_in_sub
                push!(bucket, ImageEdge{Float64}(col_img, Float64(row_img), e.strength, scan_idx))
            end
            buckets[scan_idx] = bucket
        end
        for b in buckets
            append!(edges, b)
        end
        return
    end

    # Serial path: reuse a single profile buffer — `_gauge_edges_in_profile`
    # never retains the input (see profile_edges.jl: `_smooth` always returns a
    # fresh vector), so overwriting it in place between iterations is safe.
    profile = Vector{Float64}(undef, nc_sub)
    for scan_idx in 1:nr_sub
        row_img = r1 + scan_idx - 1

        @inbounds for j in 1:nc_sub
            profile[j] = sub[scan_idx, j]
        end
        orientation == RIGHT_TO_LEFT && reverse!(profile)

        for e in _gauge_edges_in_profile(profile, sigma, threshold, polarity, selector)
            col_in_sub = orientation == RIGHT_TO_LEFT ? (nc_sub + 1 - e.position) : e.position
            col_img    = c1 - 1 + col_in_sub
            push!(edges, ImageEdge{Float64}(col_img, Float64(row_img), e.strength, scan_idx))
        end
    end
end

function _scan_cols(
    sub, c1, r1,
    orientation, sigma, threshold, polarity, selector,
    edges, threaded::Bool,
)
    nr_sub, nc_sub = size(sub)

    if threaded
        buckets = Vector{Vector{ImageEdge{Float64}}}(undef, nc_sub)
        Threads.@threads for scan_idx in 1:nc_sub
            col_img = c1 + scan_idx - 1
            profile = Vector{Float64}(undef, nr_sub)
            @inbounds for i in 1:nr_sub
                profile[i] = sub[i, scan_idx]
            end
            orientation == BOTTOM_TO_TOP && reverse!(profile)

            bucket = ImageEdge{Float64}[]
            for e in _gauge_edges_in_profile(profile, sigma, threshold, polarity, selector)
                row_in_sub = orientation == BOTTOM_TO_TOP ? (nr_sub + 1 - e.position) : e.position
                row_img    = r1 - 1 + row_in_sub
                push!(bucket, ImageEdge{Float64}(Float64(col_img), row_img, e.strength, scan_idx))
            end
            buckets[scan_idx] = bucket
        end
        for b in buckets
            append!(edges, b)
        end
        return
    end

    profile = Vector{Float64}(undef, nr_sub)
    for scan_idx in 1:nc_sub
        col_img = c1 + scan_idx - 1

        @inbounds for i in 1:nr_sub
            profile[i] = sub[i, scan_idx]
        end
        orientation == BOTTOM_TO_TOP && reverse!(profile)

        for e in _gauge_edges_in_profile(profile, sigma, threshold, polarity, selector)
            row_in_sub = orientation == BOTTOM_TO_TOP ? (nr_sub + 1 - e.position) : e.position
            row_img    = r1 - 1 + row_in_sub
            push!(edges, ImageEdge{Float64}(Float64(col_img), row_img, e.strength, scan_idx))
        end
    end
end
