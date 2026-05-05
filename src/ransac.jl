"""
Generic RANSAC engine.

This module provides two implementations:
  - `ransac`  : classic Fischler-Bolles RANSAC with adaptive iteration count.
  - `ransac2` : extended RANSAC that additionally validates model and data
                constraints after each candidate fit, and adapts its iteration
                budget from the best consensus set found so far.

Both functions are generic over the model type `M`; they call the interface
functions defined in models.jl:
  - `sample_size(::Type{M}) :: Int`
  - `fit_model(::Type{M}, pts) :: M`
  - `point_distance(m::M, pt) :: Float64`
  - `constraints_met(m::M, constraints) :: Bool`
"""

using Random: AbstractRNG, default_rng, shuffle!
using LinearAlgebra: SingularException

# Errors expected from the low-level fitters on degenerate samples
# (e.g. too few points, collinear points making the Taubin system singular).
# Anything else — interrupts, OOM, MethodError, programming bugs — must propagate.
_is_expected_fit_failure(e) = e isa ArgumentError || e isa SingularException

# ── Classic RANSAC ────────────────────────────────────────────────────────────

"""
    ransac(points, ModelType, inlier_threshold;
           confidence=0.99, max_iter=10_000, rng=default_rng())
           -> (model, inlier_indices, outlier_indices)

Run RANSAC (Fischler & Bolles 1981) to robustly fit `ModelType` to `points`.

The iteration count is determined adaptively using the formula

    N = ⌈log(1 − α) / log(1 − (1 − ε)^k)⌉

where `α` is `confidence`, `ε` is the estimated outlier fraction (updated
after each new best consensus set), and `k = sample_size(ModelType)`.

After the best consensus set is found, the model is refit on all inliers for
improved parameter accuracy.

# Arguments
- `points`: vector of points, each indexable as `pt[1]`, `pt[2]`.
- `ModelType`: concrete model type, e.g. `LineModel` or `CircleModel`.
- `inlier_threshold`: maximum [`point_distance`](@ref) to count as an inlier.
- `confidence`: desired probability of finding the correct model (default 0.99).
- `max_iter`: hard upper limit on iterations (default 10 000).
- `rng`: random-number generator; pass an explicit `MersenneTwister` for
  reproducibility.

# Returns
`(best_model, inlier_indices, outlier_indices)` where the indices refer into
the original `points` vector.

Returns `(nothing, Int[], collect(1:n))` when no model could be fitted (e.g.
fewer points than the minimum sample size, or all samples degenerate).

# Examples
```jldoctest
julia> pts = [(x, 2.0*x + 1.0) for x in 1.0:10.0];   # exact line y = 2x+1

julia> model, inl, outl = ransac(pts, LineModel, 0.1; rng=MersenneTwister(0));

julia> length(outl)   # no outliers on an exact line
0

julia> length(inl)
10
```

```jldoctest
julia> model, inl, outl = ransac(Tuple{Float64,Float64}[], LineModel, 0.5);

julia> isnothing(model)
true

julia> isempty(inl)
true
```
"""
function ransac(
    points          :: AbstractVector,
    ::Type{M},
    inlier_threshold :: Real;
    confidence  :: Real        = 0.99,
    max_iter    :: Int         = 10_000,
    rng         :: AbstractRNG = default_rng(),
) where M

    n   = length(points)
    dof = sample_size(M)
    n < dof && return (nothing, Int[], collect(1:n))

    thresh = Float64(inlier_threshold)
    best_inlier_idx = Int[]
    best_model      = nothing
    N               = max_iter   # adaptive upper bound on iterations

    idx_pool = collect(1:n)
    # Scratch buffer for the current candidate's inliers.  On a rejected
    # candidate we `empty!` and reuse; on a new best we swap references with
    # `best_inlier_idx` — zero allocation either way.
    scratch = Int[]

    for _ in 1:N
        # Random minimal sample
        sample_idx = _random_sample!(rng, idx_pool, dof)
        sample_pts = points[sample_idx]

        model = try
            fit_model(M, sample_pts)
        catch e
            _is_expected_fit_failure(e) || rethrow()
            continue
        end

        # Classify all points
        empty!(scratch)
        for i in 1:n
            point_distance(model, points[i]) <= thresh && push!(scratch, i)
        end

        if length(scratch) > length(best_inlier_idx)
            best_inlier_idx, scratch = scratch, best_inlier_idx
            best_model      = model

            # Adaptive iteration count update (Fischler-Bolles formula)
            ε      = 1.0 - length(best_inlier_idx) / n
            ε      = clamp(ε, 1e-6, 1.0 - 1e-6)
            logval = log(1.0 - (1.0 - ε)^dof)
            if logval < 0.0
                N = min(max_iter, ceil(Int, log(1.0 - Float64(confidence)) / logval))
            end
        end
    end

    isnothing(best_model) && return (nothing, Int[], collect(1:n))

    # Final refit on all inliers for improved accuracy.  If the full inlier set
    # is degenerate (collinear, coincident, …) keep the best minimal-sample fit.
    if length(best_inlier_idx) >= dof
        try
            best_model = fit_model(M, points[best_inlier_idx])
        catch e
            _is_expected_fit_failure(e) || rethrow()
        end
    end

    outlier_idx = setdiff(1:n, best_inlier_idx)
    return (best_model, best_inlier_idx, outlier_idx)
end

# ── RANSAC2: constraint-aware extended RANSAC ─────────────────────────────────

"""
    ransac2(points, ModelType, inlier_threshold, constraints;
            min_inliers=sample_size(ModelType),
            confidence=0.99, max_iter=10_000,
            initial_outlier_ratio=0.5, rng=default_rng())
            -> (model, inlier_indices, outlier_indices)

Extended RANSAC that validates model constraints after each candidate fit.

After each successful random fit, [`constraints_met`](@ref) is called to
reject geometrically inadmissible candidates (e.g. circles with wrong radius).
The best surviving consensus set is immediately refit on all its inliers, and
the adaptive iteration budget is updated from the new inlier count.

# Arguments
- `constraints`: a constraint struct matched to `ModelType` — passed verbatim
  to [`constraints_met`](@ref).
- `min_inliers`: minimum number of inliers required to accept a model
  (default: `sample_size(ModelType)`).
- `initial_outlier_ratio`: assumed outlier fraction used to seed the iteration
  budget before any good model is found (default 0.5).

All other arguments are as for [`ransac`](@ref).

# Examples
```jldoctest
julia> pts = [(x, 2.0*x + 1.0) for x in 1.0:10.0];

julia> c = LineConstraints{Float64}(min_angle=-Float64(π/2), max_angle=Float64(π/2));

julia> model, inl, _ = ransac2(pts, LineModel, 0.1, c; rng=MersenneTwister(0));

julia> length(inl)
10
```

```jldoctest
julia> pts = [(x, x) for x in 1.0:10.0];   # 45° line

julia> c_tight = LineConstraints{Float64}(min_angle=Float64(π/3), max_angle=Float64(π/2));

julia> model, inl, _ = ransac2(pts, LineModel, 0.1, c_tight;
                                rng=MersenneTwister(0), max_iter=200);

julia> isnothing(model)   # 45° line is below min_angle = 60°, so rejected
true
```
"""
function ransac2(
    points           :: AbstractVector,
    ::Type{M},
    inlier_threshold :: Real,
    constraints;
    min_inliers      :: Int        = sample_size(M),
    confidence       :: Real       = 0.99,
    max_iter         :: Int        = 10_000,
    initial_outlier_ratio :: Real  = 0.5,
    rng              :: AbstractRNG = default_rng(),
) where M

    n   = length(points)
    dof = sample_size(M)
    n < dof && return (nothing, Int[], collect(1:n))

    thresh = Float64(inlier_threshold)

    # Seed the initial iteration budget from the assumed outlier ratio
    ε0 = clamp(Float64(initial_outlier_ratio), 1e-6, 1.0 - 1e-6)
    log0 = log(1.0 - (1.0 - ε0)^dof)
    N = log0 < 0.0 ? min(max_iter, ceil(Int, log(1.0 - Float64(confidence)) / log0)) : max_iter

    best_inlier_idx = Int[]
    best_model      = nothing
    idx_pool        = collect(1:n)
    # Scratch buffers — reused across iterations; swap-on-new-best (see `ransac`
    # above for the pattern).  `reclassify` is used only when a refined model
    # triggers a reclassification pass.
    scratch    = Int[]
    reclassify = Int[]

    for _ in 1:N
        sample_idx = _random_sample!(rng, idx_pool, dof)
        sample_pts = points[sample_idx]

        model = try
            fit_model(M, sample_pts)
        catch e
            _is_expected_fit_failure(e) || rethrow()
            continue
        end

        # Model constraint check (radius range, angle range, etc.)
        constraints_met(model, constraints) || continue

        empty!(scratch)
        for i in 1:n
            point_distance(model, points[i]) <= thresh && push!(scratch, i)
        end

        length(scratch) < min_inliers && continue
        # Data-aware check (e.g. arc completeness for circles).  Models that
        # don't need this hit the generic `true` fallback with zero cost.
        data_constraints_met(model, constraints, view(points, scratch)) || continue

        if length(scratch) > length(best_inlier_idx)
            best_inlier_idx, scratch = scratch, best_inlier_idx
            best_model      = model

            # Refit on all inliers immediately so constraints can be re-checked.
            # If the refit itself is degenerate, keep the unrefined candidate.
            try
                candidate = fit_model(M, points[best_inlier_idx])
                if constraints_met(candidate, constraints)
                    best_model = candidate
                    # Reclassify with the refined model
                    empty!(reclassify)
                    for i in 1:n
                        point_distance(best_model, points[i]) <= thresh && push!(reclassify, i)
                    end
                    if length(reclassify) >= length(best_inlier_idx)
                        best_inlier_idx, reclassify = reclassify, best_inlier_idx
                    end
                end
            catch e
                _is_expected_fit_failure(e) || rethrow()
            end

            # Adaptive N update
            ε      = 1.0 - length(best_inlier_idx) / n
            ε      = clamp(ε, 1e-6, 1.0 - 1e-6)
            logval = log(1.0 - (1.0 - ε)^dof)
            if logval < 0.0
                N = min(N, ceil(Int, log(1.0 - Float64(confidence)) / logval))
            end
        end
    end

    isnothing(best_model) && return (nothing, Int[], collect(1:n))

    outlier_idx = setdiff(1:n, best_inlier_idx)
    return (best_model, best_inlier_idx, outlier_idx)
end

# ── Internal helpers ──────────────────────────────────────────────────────────

"""
Sample `k` distinct indices from `pool` without replacement, in place.

Uses a Fisher-Yates partial shuffle on `pool` itself — correctness does not
depend on the initial order, so callers can reuse the same `pool` across many
iterations and avoid per-iteration allocation.  Returns a `view` over the
sampled tail; copy if you need independent ownership.
"""
function _random_sample!(rng::AbstractRNG, pool::Vector{Int}, k::Int)
    n = length(pool)
    k > n && throw(ArgumentError("Cannot sample $k items from pool of size $n"))
    for i in n:-1:(n - k + 1)
        j = rand(rng, 1:i)
        pool[i], pool[j] = pool[j], pool[i]
    end
    return view(pool, (n - k + 1):n)
end
