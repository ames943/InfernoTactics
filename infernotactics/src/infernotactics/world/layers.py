"""Canonical static raster-layer schema."""


STATIC_LAYER_NAMES = (
    "elevation",
    "slope",
    "building_density",
    "building_height",
    "road_mask",
    "fuel_density",
    "water_mask",
    "population_density",
)
LAYER_INDEX = {name: index for index, name in enumerate(STATIC_LAYER_NAMES)}
