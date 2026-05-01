```@meta
CurrentModule = EdgesGauging
```

# EdgesGauging.jl

Sub-pixel edge detection and robust geometric fitting (lines, circles) for
machine-vision gauging tasks, written in pure Julia.

## What it does

- **1-D edge detection** in intensity profiles, with Gaussian smoothing and
  parabolic sub-pixel interpolation of gradient extrema.
- **2-D edge detection** across rectangular ROIs, multi-strip scans, and
  radial / ring scans from a reference point.
- **Robust geometric fitting** via a generic RANSAC engine with constraint
  support (angle, radius, arc completeness, inlier counts).
- **Parametric element types** — `LineModel{T}`, `CircleModel{T}`, etc. so
  Float32 pipelines work end-to-end at the model layer.

## Quick start

```julia
using EdgesGauging

result = gauge_edges_in_profile(profile, 2.0, 0.1,
                                POLARITY_POSITIVE, SELECT_BEST)

cc  = CircleConstraints{Float64}(min_radius=10.0, max_radius=200.0)
fit = gauge_circle(image, (row_c, col_c), 0.0, 2π, deg2rad(3.0), 80, 2.0, 0.1;
                   constraints = cc)
```

## Conventions

- Image arrays are `(row, col)` — matching Julia's column-major indexing.
- `center_rc` arguments are therefore `(row, col)` tuples.
- Detected edges in [`ImageEdge`](@ref) are exposed as Cartesian
  `(x = col, y = row)` for consumers that prefer image-space coordinates.

See the [API reference](@ref api) for the complete public interface.
