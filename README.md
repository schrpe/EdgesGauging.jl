# Edges.jl

Sub-pixel edge detection and robust geometric fitting (lines, circles) for
machine-vision gauging tasks, written in pure Julia.

## What it does

- **1-D edge detection** in intensity profiles, with Gaussian smoothing and
  parabolic sub-pixel interpolation of gradient extrema. NaN-tolerant — works
  on profiles padded with NaN at the ends.
- **2-D edge detection** across rectangular ROIs, multi-strip scans, and
  radial / ring scans from a reference point.
- **Profile extraction along arbitrary paths** — sample an image along a line
  segment or a circular arc with selectable interpolation (nearest / bilinear /
  bicubic), configurable strip width, and explicit `NaN` for out-of-bounds
  samples.
- **Robust geometric fitting** via a generic RANSAC engine with constraint
  support (angle, radius, arc completeness, inlier counts).
- **Parametric element types** — `LineModel{T}`, `CircleModel{T}`, etc. so
  Float32 pipelines work end-to-end at the model layer.

## Install

```julia
] add https://github.com/schrpe/Edges.jl
```

(Once registered in the General registry: `] add Edges`.)

## Quick start

```julia
using Edges

# 1-D edge detection in a pixel profile
result = gauge_edges_in_profile(profile, 2.0, 0.1,
                                POLARITY_POSITIVE, SELECT_BEST)

# Fit a circle to an image region
cc  = CircleConstraints{Float64}(min_radius=10.0, max_radius=200.0)
fit = gauge_circle(image, (row_c, col_c), 0.0, 2π, deg2rad(3.0), 80, 2.0, 0.1;
                   constraints = cc)
println("Centre: (", fit.cx, ", ", fit.cy, ")  radius: ", fit.r)
```

See the docstrings of `gauge_edges_in_profile`, `gauge_edges_info`,
`gauge_line`, `gauge_circle` for full argument documentation, and `test/` for
end-to-end examples.

## Profile extraction along arbitrary paths

`extract_line_profile` and `extract_arc_profile` sample image intensity along
a straight segment or a circular arc, returning a 1-D `Vector{Float64}` ready
to feed into `gauge_edges_in_profile`:

```julia
# Slim profile across a feature, then sub-pixel edge detection
p = extract_line_profile(img, (row0, col0), (row1, col1);
                         width=1, interp=INTERP_BICUBIC)
r = gauge_edges_in_profile(p, 2.0, 10.0, POLARITY_POSITIVE, SELECT_BEST)

# Wider strip averages 5 parallel samples — useful when the feature is
# slightly tilted or noisy
p_strip = extract_line_profile(img, p0_rc, p1_rc; width=5)

# Arc-length-uniform sampling along an arc (auto-density)
p_arc = extract_arc_profile(img, (row_c, col_c), 40.0, 0.0, π/2)
```

Pick the interpolation method via `interp=`:

| Mode               | When to use                                            |
|:------------------ |:------------------------------------------------------ |
| `INTERP_NEAREST`   | Diagnostic / discrete-pixel validation; no sub-pixel.  |
| `INTERP_BILINEAR`  | Faster, slight sub-pixel bias at pixel boundaries.     |
| `INTERP_BICUBIC`   | Default; best sub-pixel edge stability.                |

Out-of-bounds samples are emitted as `NaN`. `gauge_edges_in_profile` is
NaN-tolerant: NaN-padded profiles flow straight through and the edge is
still found in the valid interior.

## Conventions

- Image arrays are `(row, col)` — matching Julia's column-major indexing.
- `center_rc` and segment-endpoint arguments are therefore `(row, col)` tuples.
- **Integer index = pixel centre.** `image[i, j]` is the value at
  `(row, col) = (i, j)`; the pixel itself spatially covers
  `(i-0.5, j-0.5)` … `(i+0.5, j+0.5)`. Sample positions outside
  `(0.5, 0.5)`…`(nrows+0.5, ncols+0.5)` yield `NaN`.
- Detected edges in `ImageEdge` are exposed as Cartesian `(x = col, y = row)`
  for consumers that prefer image-space coordinates.

## Error handling

`gauge_line` and `gauge_circle` throw `GaugingError` with a `reason::Symbol`
field when a pipeline fails:

- `:too_few_points` — edge detection produced fewer points than the fitter
  requires.
- `:ransac_failed` — RANSAC could not find a model satisfying the supplied
  constraints (radius range, angle range, arc completeness, …).

Catch on `e.reason` to branch on the failure mode:

```julia
try
    fit = gauge_circle(img, (row, col), 0, 2π, deg2rad(3), 80, 2.0, 0.1)
catch e
    e isa GaugingError && e.reason === :too_few_points || rethrow()
    # fall back to a coarser threshold / larger ROI / …
end
```

## Running the tests

```
julia --project=. test/runtests.jl
```

`Pkg.test()` also works on a clean environment; on some Julia 1.12 installs the
direct invocation above avoids a Pkg sandbox quirk around stdlib `Test`
resolution.
