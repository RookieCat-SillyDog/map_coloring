"""Data definitions: maps, graphs, and problem instances."""

from map_coloring.data.maps import (
    MAPS,
    MapConfig,
    get_map,
    get_edges,
    get_layout,
    get_color_domain,
    list_maps,
)

__all__ = [
    "MAPS",
    "MapConfig",
    "get_map",
    "get_edges",
    "get_layout",
    "get_color_domain",
    "list_maps",
]
