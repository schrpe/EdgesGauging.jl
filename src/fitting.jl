"""
Low-level geometric fitting functions (no RANSAC — operate on a clean point set).

  - `fit_line_tls`      : Total Least Squares line via SVD
  - `fit_circle_kasa`   : Kåsa algebraic circle fit
  - `fit_circle_taubin` : Taubin algebraic circle fit (less biased on partial arcs)
  - `fit_circle_lm`     : Geometric Levenberg-Marquardt circle refinement
  - `fit_parabola`      : Quadratic least-squares via QR decomposition
"""

using LinearAlgebra: svd, qr, norm

# ── Line: Total Least Squares ─────────────────────────────────────────────────

"""
    fit_line_tls(points) -> (A, B, C)

Fit the line `Ax + By + C = 0` (with A² + B² = 1) to `points` using Total
Least Squares (equivalent to PCA / SVD of the centred point matrix).

`points` is a vector of `(x, y)` tuples or any length-2 indexable collection.
Returns `(A, B, C)` as `Float64`.

Sign convention: `A` is preferably positive; if `A == 0` then `B > 0`.

Throws `ArgumentError` if fewer than 2 points are provided.

# Examples
```jldoctest
julia> pts = [(0.0, 5.0), (1.0, 5.0), (2.0, 5.0)];   # horizontal line y = 5

julia> A, B, C = fit_line_tls(pts);

julia> round(A^2 + B^2, digits=10)   # unit normal
1.0

julia> all(abs(A*p[1] + B*p[2] + C) < 1e-10 for p in pts)
true
```

```jldoctest
julia> fit_line_tls([(1.0, 1.0)])   # too few points
ERROR: ArgumentError: fit_line_tls requires ≥ 2 points
[...]
```
"""
function fit_line_tls(points::AbstractVector)
    length(points) < 2 && throw(ArgumentError("fit_line_tls requires ≥ 2 points"))

    n = length(points)
    M = Matrix{Float64}(undef, n, 2)
    sx = 0.0
    sy = 0.0
    @inbounds for i in 1:n
        p = points[i]
        x = Float64(p[1]); y = Float64(p[2])
        M[i, 1] = x
        M[i, 2] = y
        sx += x
        sy += y
    end
    cx = sx / n
    cy = sy / n
    @inbounds for i in 1:n
        M[i, 1] -= cx
        M[i, 2] -= cy
    end

    # Last right singular vector of the centred matrix is the line normal.
    _, _, V = svd(M)
    A, B = V[1, end], V[2, end]   # normal direction (unit vector from SVD)
    C = -(A * cx + B * cy)

    # Normalise sign: A preferably positive; tie-break on B.
    if A < 0 || (A == 0 && B < 0)
        A, B, C = -A, -B, -C
    end

    return (A, B, C)
end

# ── Circle: Kåsa algebraic fit ────────────────────────────────────────────────

"""
    fit_circle_kasa(points) -> (cx, cy, r)

Fit a circle to `points` using the Kåsa (1976) algebraic method: minimise
`∑(x² + y² − 2·cx·x − 2·cy·y + cx² + cy² − r²)²` by solving a linear system
in the centre coordinates.

Kåsa is fast and numerically straightforward, but biased toward smaller
radii when samples cover only a short arc — the regime typical of
edge-gauging.  Use [`fit_circle_taubin`](@ref) for substantially lower
bias at essentially the same cost, or [`fit_circle_lm`](@ref) for a
geometric maximum-likelihood refinement.

Returns `(cx, cy, r)` as `Float64`.

Throws `ArgumentError` if fewer than 3 points are provided.

# Examples
```jldoctest
julia> angles = range(0, 2π, length=13)[1:end-1];

julia> pts = [(3.0 + 5.0*cos(θ), 4.0 + 5.0*sin(θ)) for θ in angles];

julia> cx, cy, r = fit_circle_kasa(pts);

julia> round(cx, digits=6)
3.0

julia> round(cy, digits=6)
4.0

julia> round(r, digits=6)
5.0
```

```jldoctest
julia> fit_circle_kasa([(0.0,0.0),(1.0,0.0)])   # too few points
ERROR: ArgumentError: fit_circle_kasa requires ≥ 3 points
[...]
```
"""
function fit_circle_kasa(points::AbstractVector)
    length(points) < 3 && throw(ArgumentError("fit_circle_kasa requires ≥ 3 points"))

    xs = Float64[p[1] for p in points]
    ys = Float64[p[2] for p in points]
    n  = length(xs)

    # Centre the data for numerical stability.
    mx = sum(xs) / n
    my = sum(ys) / n
    u  = xs .- mx
    v  = ys .- my

    # Moment sums of the centred coordinates.
    Suu = sum(u .^ 2)
    Svv = sum(v .^ 2)
    Suv = sum(u .* v)
    Suuu = sum(u .^ 3)
    Svvv = sum(v .^ 3)
    Suvv = sum(u .* v .^ 2)
    Svuu = sum(v .* u .^ 2)

    # Solve the 2×2 linear system for the circle centre shift from data centroid.
    A_mat = [Suu Suv; Suv Svv]
    b_vec = [0.5 * (Suuu + Suvv); 0.5 * (Svvv + Svuu)]
    ab = A_mat \ b_vec

    cx = ab[1] + mx
    cy = ab[2] + my
    r  = sqrt(sum(((xs .- cx) .^ 2) .+ ((ys .- cy) .^ 2)) / n)

    return (cx, cy, r)
end

# ── Circle: Taubin algebraic fit ──────────────────────────────────────────────

"""
    fit_circle_taubin(points) -> (cx, cy, r)

Fit a circle to `points` using Taubin's algebraic method (Taubin 1991;
Chernov & Lesort 2005), implemented as a 3-column SVD on centred and
Z-scaled coordinates.

Taubin minimises the same algebraic residual as Kåsa but normalises it by
the squared gradient `‖∇F‖²`, which removes the strong radius bias that
Kåsa exhibits on short arcs.  On a full circle the two methods agree;
on a partial arc Taubin's centre and radius are markedly closer to the
geometric optimum at essentially the same cost.

Returns `(cx, cy, r)` as `Float64`.

Throws `ArgumentError` if fewer than 3 points are given, all points
coincide, or the point set is collinear.

# Examples
```jldoctest
julia> angles = range(0, 2π, length=13)[1:end-1];

julia> pts = [(3.0 + 5.0*cos(θ), 4.0 + 5.0*sin(θ)) for θ in angles];

julia> cx, cy, r = fit_circle_taubin(pts);

julia> round(cx, digits=6)
3.0

julia> round(cy, digits=6)
4.0

julia> round(r, digits=6)
5.0
```

```jldoctest
julia> fit_circle_taubin([(0.0,0.0),(1.0,0.0)])   # too few points
ERROR: ArgumentError: fit_circle_taubin requires ≥ 3 points
[...]
```
"""
function fit_circle_taubin(points::AbstractVector)
    length(points) < 3 && throw(ArgumentError("fit_circle_taubin requires ≥ 3 points"))

    n  = length(points)
    xs = Float64[p[1] for p in points]
    ys = Float64[p[2] for p in points]

    mx = sum(xs) / n
    my = sum(ys) / n
    u  = xs .- mx
    v  = ys .- my
    Z  = u .^ 2 .+ v .^ 2
    Zmean = sum(Z) / n

    Zmean > 0 ||
        throw(ArgumentError("fit_circle_taubin: degenerate point configuration"))

    # Taubin scaling of the Z column: this is the key step that distinguishes
    # the method from Kåsa — it normalises the algebraic cost by ‖∇F‖².
    s  = 2 * sqrt(Zmean)
    Z0 = (Z .- Zmean) ./ s

    # SVD of the n×3 design matrix; the right singular vector corresponding
    # to the smallest singular value solves the constrained algebraic problem.
    M = hcat(Z0, u, v)
    F = svd(M)
    a1, a2, a3 = F.V[1, 3], F.V[2, 3], F.V[3, 3]

    # Undo the Z scaling and recover the constant term D of the algebraic form
    # A(x²+y²) + Bx + Cy + D = 0.
    A1 = a1 / s
    A4 = -Zmean * A1

    abs(A1) > eps(Float64) ||
        throw(ArgumentError("fit_circle_taubin: collinear or near-collinear points"))

    cx_c = -a2 / (2 * A1)
    cy_c = -a3 / (2 * A1)
    disc = a2 * a2 + a3 * a3 - 4 * A1 * A4
    r    = sqrt(max(disc, 0.0)) / (2 * abs(A1))

    return (cx_c + mx, cy_c + my, r)
end

# ── Circle: Levenberg-Marquardt geometric refinement ─────────────────────────

"""
    fit_circle_lm(points; cx0, cy0, r0, max_iter=50, tol=1e-10) -> (cx, cy, r)

Refine a circle estimate by minimising the sum of squared **orthogonal**
distances `Σ (√((xᵢ-cx)² + (yᵢ-cy)²) − r)²` with a damped Gauss-Newton
(Levenberg-Marquardt) iteration.  This is the maximum-likelihood estimator
under isotropic Gaussian noise on the points, in contrast to the algebraic
methods (Kåsa, Taubin) which minimise a different, biased cost.

Requires a starting estimate `(cx0, cy0, r0)` — typically the output of
[`fit_circle_taubin`](@ref).  Convergence is quadratic from a good start
and usually completes in fewer than ten iterations.

Returns `(cx, cy, r)` as `Float64`.

Throws `ArgumentError` if fewer than 3 points are given.  If the iteration
diverges (damping grows beyond a safe bound), the last accepted estimate
is returned rather than throwing.

# Examples
```jldoctest
julia> angles = range(0, 2π, length=20)[1:end-1];

julia> pts = [(3.0 + 5.0*cos(θ), 4.0 + 5.0*sin(θ)) for θ in angles];

julia> cx, cy, r = fit_circle_lm(pts; cx0=2.5, cy0=4.5, r0=4.5);

julia> round(cx, digits=8)
3.0

julia> round(cy, digits=8)
4.0

julia> round(r, digits=8)
5.0
```
"""
function fit_circle_lm(points::AbstractVector;
                       cx0::Real, cy0::Real, r0::Real,
                       max_iter::Int=50, tol::Real=1e-10)
    length(points) < 3 && throw(ArgumentError("fit_circle_lm requires ≥ 3 points"))

    xs = Float64[p[1] for p in points]
    ys = Float64[p[2] for p in points]
    n  = length(xs)

    cx, cy, r = Float64(cx0), Float64(cy0), Float64(r0)
    λ = 1e-3

    # Sum of squared orthogonal distances.  Inlined twice rather than
    # closed over a `cost` helper, since a nested function would capture
    # and overwrite the outer cost variable through Julia's closure scoping.
    S = 0.0
    @inbounds for i in 1:n
        dx = xs[i] - cx
        dy = ys[i] - cy
        di = sqrt(dx*dx + dy*dy) - r
        S += di*di
    end

    for _ in 1:max_iter
        # Build JᵀJ (3×3) and Jᵀd (3-vector) without forming the full Jacobian.
        # Residual: dᵢ = ρᵢ − r,   ρᵢ = √((xᵢ-cx)² + (yᵢ-cy)²)
        # Jacobian rows: [-(x-cx)/ρ, -(y-cy)/ρ, -1]
        a11 = a12 = a13 = a22 = a23 = a33 = 0.0
        b1  = b2  = b3  = 0.0
        @inbounds for i in 1:n
            dx = xs[i] - cx
            dy = ys[i] - cy
            ρ  = sqrt(dx*dx + dy*dy)
            ρ < eps(Float64) && continue   # skip points sitting on the centre
            jcx = -dx / ρ
            jcy = -dy / ρ
            jr  = -1.0
            di  = ρ - r
            a11 += jcx*jcx; a12 += jcx*jcy; a13 += jcx*jr
            a22 += jcy*jcy; a23 += jcy*jr
            a33 += jr*jr
            b1  += jcx*di
            b2  += jcy*di
            b3  += jr *di
        end

        H = [a11+λ  a12     a13;
             a12    a22+λ   a23;
             a13    a23     a33+λ]
        g = [b1, b2, b3]

        local Δ
        try
            Δ = -(H \ g)
        catch
            λ *= 10
            λ > 1e10 && break
            continue
        end

        cx_new = cx + Δ[1]
        cy_new = cy + Δ[2]
        r_new  = r  + Δ[3]
        S_new  = 0.0
        @inbounds for i in 1:n
            dx = xs[i] - cx_new
            dy = ys[i] - cy_new
            di = sqrt(dx*dx + dy*dy) - r_new
            S_new += di*di
        end

        if S_new < S
            improvement = (S - S_new) / max(S, eps(Float64))
            cx, cy, r = cx_new, cy_new, r_new
            S = S_new
            λ = max(λ / 10, 1e-15)
            improvement < tol && break
        else
            λ *= 10
            λ > 1e10 && break
        end
    end

    return (cx, cy, r)
end

# ── Parabola: quadratic least squares ─────────────────────────────────────────

"""
    fit_parabola(xs, ys) -> (a, b, c)

Fit the parabola `y = a·x² + b·x + c` to the data vectors `xs`, `ys` using
QR least-squares decomposition.

Returns `(a, b, c)` as `Float64`.

Throws `ArgumentError` if `xs` and `ys` differ in length or if fewer than
3 points are provided.

# Examples
```jldoctest
julia> a, b, c = fit_parabola([0.0, 1.0, 2.0], [1.0, 0.0, 3.0]);

julia> round(a, digits=10)
2.0

julia> round(b, digits=10)
-3.0

julia> round(c, digits=10)
1.0
```

```jldoctest
julia> fit_parabola([1.0, 2.0], [1.0, 4.0])   # too few points
ERROR: ArgumentError: fit_parabola requires ≥ 3 points
[...]
```
"""
function fit_parabola(xs::AbstractVector, ys::AbstractVector)
    length(xs) == length(ys) || throw(ArgumentError("xs and ys must have equal length"))
    length(xs) < 3 && throw(ArgumentError("fit_parabola requires ≥ 3 points"))

    x = Float64.(xs)
    y = Float64.(ys)
    A = [x .^ 2   x   ones(length(x))]
    coeffs = qr(A) \ y
    return (coeffs[1], coeffs[2], coeffs[3])
end
