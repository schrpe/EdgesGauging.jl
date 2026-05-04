"""
1D profile edge detection.

Mirrors the C++ `gauge_edges_info_in_profile()`:
  1. Gaussian smooth the profile.
  2. Compute gradient with the [-1, 0, 1] kernel.
  3. Find gradient extrema with parabolic sub-pixel interpolation.
  4. Filter by polarity and strength threshold.
  5. Apply edge selector (first / last / best / all).
"""

using Images: imfilter, KernelFactors, centered

# ── Public API ───────────────────────────────────────────────────────────────

"""
    gauge_edges_in_profile(profile, sigma, threshold,
                           polarity=POLARITY_ANY, selector=SELECT_ALL)
                           -> ProfileEdgesResult{Float64}

Detect edges in a 1-D intensity profile using Gaussian smoothing, a
symmetric gradient kernel, and parabolic sub-pixel interpolation.

# Arguments
- `profile`: 1-D array of pixel intensities (any numeric element type).
- `sigma`: Gaussian smoothing standard deviation (pixels).  Pass `0` or a
  non-positive value to skip smoothing entirely.
- `threshold`: minimum edge strength (`|gradient|` at the extremum) to report.
  Edges weaker than this are discarded.
- `polarity`: which gradient directions to detect — [`POLARITY_POSITIVE`](@ref EdgePolarity)
  (dark→bright), [`POLARITY_NEGATIVE`](@ref EdgePolarity) (bright→dark), or
  [`POLARITY_ANY`](@ref EdgePolarity) (both).
- `selector`: how many edges to return per profile — [`SELECT_FIRST`](@ref EdgeSelector),
  [`SELECT_LAST`](@ref EdgeSelector), [`SELECT_BEST`](@ref EdgeSelector) (strongest), or
  [`SELECT_ALL`](@ref EdgeSelector).

# Returns
A [`ProfileEdgesResult`](@ref) containing:
- `edges`: detected edges, each with a sub-pixel `position` (1-based) and
  `strength`.
- `smoothed`: Gaussian-smoothed copy of the input profile.
- `gradient`: discrete `[-1, 0, 1]` derivative of the smoothed profile.

Profiles shorter than 3 elements return an empty edge list without error.

# Sub-pixel interpolation
The extremum position is refined by fitting a parabola through the three
samples around the gradient peak:

    offset = 0.5 * (g[i-1] - g[i+1]) / (g[i-1] - 2·g[i] + g[i+1])

clamped to ±0.5 so the result stays within the local sample interval.

# Examples
```jldoctest
julia> p = [0.0, 0.0, 0.0, 100.0, 100.0, 100.0];

julia> r = gauge_edges_in_profile(p, 0.0, 5.0, POLARITY_POSITIVE, SELECT_FIRST);

julia> length(r.edges)
1

julia> r.edges[1].position   # step sits between indices 3 and 4 → 3.5
3.5

julia> r.edges[1].strength > 0
true
```

```jldoctest
julia> r = gauge_edges_in_profile(fill(128.0, 50), 2.0, 1.0);

julia> isempty(r.edges)   # flat profile → no edges
true
```

```jldoctest
julia> r = gauge_edges_in_profile(Float64[], 1.0, 1.0);

julia> isempty(r.edges)   # empty profile → no crash
true
```
"""
function gauge_edges_in_profile(
    profile   :: AbstractVector,
    sigma     :: Real,
    threshold :: Real,
    polarity  :: EdgePolarity = POLARITY_ANY,
    selector  :: EdgeSelector = SELECT_ALL,
) :: ProfileEdgesResult{Float64}

    # No-op when the caller already has a Vector{Float64}; the downstream
    # helpers never mutate `p`, and `_smooth` always returns a fresh vector
    # (either `copy(p)` when σ ≤ 0, or the `imfilter` output), so the result's
    # `.smoothed` field is independent of the caller's input.
    p = profile isa Vector{Float64} ? profile : Float64.(profile)
    # Early-return path aliases `p` as `smoothed` — copy defensively so the
    # caller can mutate their profile without affecting the returned result.
    length(p) < 3 && return ProfileEdgesResult{Float64}(EdgeResult{Float64}[], copy(p), zeros(length(p)))

    smoothed = _smooth(p, sigma)
    grad     = _gradient(smoothed)
    edges    = _find_edges(grad, polarity, Float64(threshold))
    selected = _apply_selector(edges, selector)

    return ProfileEdgesResult{Float64}(selected, smoothed, grad)
end

"""
    _gauge_edges_in_profile(profile, sigma, threshold, polarity, selector)
        -> Vector{EdgeResult{Float64}}

Internal fast-path equivalent of [`gauge_edges_in_profile`](@ref) that returns
just the selected edges — no `ProfileEdgesResult` wrapper, and the smoothed /
gradient buffers become garbage as soon as this function returns.  Used by
the 2-D scanning functions that call this in a tight inner loop and never
read `.smoothed` or `.gradient`.
"""
function _gauge_edges_in_profile(
    profile   :: AbstractVector,
    sigma     :: Real,
    threshold :: Real,
    polarity  :: EdgePolarity,
    selector  :: EdgeSelector,
) :: Vector{EdgeResult{Float64}}

    p = profile isa Vector{Float64} ? profile : Float64.(profile)
    length(p) < 3 && return EdgeResult{Float64}[]

    smoothed = _smooth(p, sigma)
    grad     = _gradient(smoothed)
    edges    = _find_edges(grad, polarity, Float64(threshold))
    return _apply_selector(edges, selector)
end

# ── Internal helpers ──────────────────────────────────────────────────────────

"""
Apply 1D Gaussian filter; returns a copy of the profile unchanged when
sigma ≤ 0.

When the profile contains `NaN` values (e.g. from out-of-bounds samples in a
profile extracted by [`extract_line_profile`](@ref) or
[`extract_arc_profile`](@ref)), a normalised-convolution path is used: NaN
samples are excluded from the local weighted average, and the result is
`NaN` only at positions where the kernel sees no valid samples at all.
NaN-free profiles take a fast path that is bit-identical to the previous
behaviour.
"""
function _smooth(profile::Vector{Float64}, sigma::Real)
    sigma <= 0 && return copy(profile)
    kernel = KernelFactors.gaussian((Float64(sigma),))
    if !any(isnan, profile)
        return imfilter(profile, kernel, "reflect")
    end
    # NaN-aware path: replace NaN with 0, build presence mask, filter both,
    # divide.  Output is NaN where the local kernel saw no valid samples.
    mask  = .!isnan.(profile)
    data  = ifelse.(mask, profile, 0.0)
    num   = imfilter(data,           kernel, "reflect")
    den   = imfilter(Float64.(mask), kernel, "reflect")
    out   = similar(profile)
    @inbounds for i in eachindex(out)
        out[i] = den[i] > 0 ? num[i] / den[i] : NaN
    end
    return out
end

"""Symmetric [-1, 0, 1] gradient kernel, applied with reflect boundary."""
function _gradient(smoothed::Vector{Float64})
    kernel = centered([-1.0, 0.0, 1.0])
    return imfilter(smoothed, kernel, "reflect")
end

"""
Find all gradient extrema in `grad` that match `polarity` and lie above
`threshold`.  Returns sub-pixel positions via parabolic interpolation.
"""
function _find_edges(
    grad      :: Vector{Float64},
    polarity  :: EdgePolarity,
    threshold :: Float64,
) :: Vector{EdgeResult{Float64}}

    edges = EdgeResult{Float64}[]
    n = length(grad)
    n < 3 && return edges

    for i in 2:(n - 1)
        gp, gc, gn = grad[i-1], grad[i], grad[i+1]
        # Skip positions where any of the three samples is NaN — sub-pixel
        # interpolation needs all three to be finite.
        (isnan(gp) || isnan(gc) || isnan(gn)) && continue

        is_max = gc > 0 && gc >= gp && gc >= gn && (gc > gp || gc > gn)
        is_min = gc < 0 && gc <= gp && gc <= gn && (gc < gp || gc < gn)

        want = (polarity == POLARITY_ANY      && (is_max || is_min)) ||
               (polarity == POLARITY_POSITIVE &&  is_max)            ||
               (polarity == POLARITY_NEGATIVE &&  is_min)

        !want && continue
        abs(gc) < threshold && continue

        # Parabolic sub-pixel interpolation for the extremum location.
        # Fit parabola through (i-1, gp), (i, gc), (i+1, gn):
        #   offset = 0.5 * (gp - gn) / (gp - 2*gc + gn)
        denom  = gp - 2.0 * gc + gn
        offset = abs(denom) > eps() ? 0.5 * (gp - gn) / denom : 0.0
        # Clamp to ±0.5 to stay within the local interval
        offset = clamp(offset, -0.5, 0.5)

        position = Float64(i) + offset   # 1-based
        push!(edges, EdgeResult{Float64}(position, abs(gc)))
    end

    return edges
end

"""Apply the edge selector to a sorted (left-to-right) edge list."""
function _apply_selector(
    edges :: Vector{EdgeResult{Float64}},
    sel   :: EdgeSelector,
) :: Vector{EdgeResult{Float64}}

    isempty(edges) && return edges
    sel == SELECT_ALL   && return edges
    sel == SELECT_FIRST && return [first(edges)]
    sel == SELECT_LAST  && return [last(edges)]
    # SELECT_BEST: highest strength
    return [argmax(e -> e.strength, edges)]
end
