# [API reference](@id api)

```@meta
CurrentModule = EdgesGauging
```

## Enumerations

```@docs
EdgePolarity
EdgeSelector
ScanOrientation
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

## Edge detection

```@docs
gauge_edges_in_profile
gauge_edges_info
gauge_edge_points_info
gauge_circular_edge_points_info
gauge_ring_edge_points_info
```

## Low-level fitting

```@docs
fit_line_tls
fit_circle_kasa
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
