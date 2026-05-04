"""
    EdgesGauging

Provides subpixel-accurate edge detection in images and robust geometric
fitting (line, circle) using RANSAC with outlier rejection.

## Quick start

```julia
using EdgesGauging

# 1D edge detection in a pixel profile
result = gauge_edges_in_profile(profile, 2.0, 0.1, POLARITY_POSITIVE, SELECT_BEST)

# Fit a circle to an image region
fit = gauge_circle(image, (row_c, col_c), 0.0, 2π, deg2rad(3.0), 80, 2.0, 0.1;
                   constraints = CircleConstraints{Float64}(min_radius=10.0, max_radius=200.0))
println("Centre: (", fit.cx, ", ", fit.cy, ")  radius: ", fit.r)
```
"""
module EdgesGauging

# ── Dependencies ──────────────────────────────────────────────────────────────
# Standard library
using LinearAlgebra
using Statistics
using Random

# Third-party (declared in Project.toml)
using Images
using Interpolations

# ── Source files ──────────────────────────────────────────────────────────────
# Load in dependency order: types first, low-level algorithms next, then
# high-level wrappers that call them.
include("types.jl")
include("profile_edges.jl")
include("fitting.jl")
include("models.jl")
include("ransac.jl")
include("image_edges.jl")
include("profiles.jl")     # shared sampling helpers + extract_line_/arc_profile
include("edge_points.jl")  # uses helpers from profiles.jl
include("gauging.jl")

# ── Public exports ────────────────────────────────────────────────────────────

# Enumerations
export EdgePolarity, POLARITY_POSITIVE, POLARITY_NEGATIVE, POLARITY_ANY
export EdgeSelector, SELECT_FIRST, SELECT_LAST, SELECT_BEST, SELECT_ALL
export ScanOrientation, LEFT_TO_RIGHT, RIGHT_TO_LEFT, TOP_TO_BOTTOM, BOTTOM_TO_TOP
export InterpolationMode, INTERP_NEAREST, INTERP_BILINEAR, INTERP_BICUBIC

# Result types
export EdgeResult, ProfileEdgesResult, ImageEdge
export LineFit, CircleFit

# Constraint types
export LineConstraints, LineSegmentConstraints, CircleConstraints

# Errors
export GaugingError

# Profile extraction along arbitrary geometric paths
export extract_line_profile, extract_arc_profile

# Edge detection
export gauge_edges_in_profile
export gauge_edges_info
export gauge_edge_points_info
export gauge_circular_edge_points_info
export gauge_ring_edge_points_info

# Fitting (low-level, no RANSAC)
export fit_line_tls, fit_circle_kasa, fit_circle_taubin, fit_circle_lm, fit_parabola

# High-level gauging (edge detection + RANSAC)
export gauge_line, gauge_circle

# Model types (useful when calling ransac / ransac2 directly)
export LineModel, LineSegmentModel, CircleModel
export sample_size, fit_model, point_distance, constraints_met, data_constraints_met
export arc_completeness, rms_error, segment_length

# RANSAC engine
export ransac, ransac2

end # module EdgesGauging
