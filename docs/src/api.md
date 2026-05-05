# [API reference](@id api)

```@meta
CurrentModule = EdgesGauging
```

## Module

```@docs
EdgesGauging
```

## Enumerations

```@docs
EdgePolarity
EdgeSelector
ScanOrientation
InterpolationMode
```

## Result types

```@docs
EdgeResult
ProfileEdgesResult
ImageEdge
LineFit
CircleFit
```

## Constraint types

```@docs
LineConstraints
LineSegmentConstraints
CircleConstraints
```

## Errors

```@docs
GaugingError
```

## Profile extraction

```@docs
extract_line_profile
extract_arc_profile
```

## Edge detection

```@docs
gauge_edges_in_profile
gauge_edges
gauge_edge_points
gauge_circular_edge_points
gauge_ring_edge_points
```

## Low-level fitting

```@docs
fit_line_tls
fit_circle_kasa
fit_circle_taubin
fit_circle_lm
fit_parabola
```

## High-level gauging

```@docs
gauge_line
gauge_circle
```

## RANSAC engine and model interface

```@docs
ransac
ransac2
sample_size
fit_model
point_distance
constraints_met
data_constraints_met
arc_completeness
rms_error
segment_length
LineModel
LineSegmentModel
CircleModel
```
