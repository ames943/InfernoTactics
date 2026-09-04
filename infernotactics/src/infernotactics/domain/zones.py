"""Canonical macro-zone geometry.

Every consumer must use this module instead of independently rebuilding zone
boundaries. The previous environment and CNN implementations both produced 32
zones, but disagreed about the cells contained by nearly every zone.
"""

from dataclasses import dataclass


DEFAULT_ZONE_ROWS = 4
DEFAULT_ZONE_COLS = 8


@dataclass(frozen=True, slots=True)
class ZoneBounds:
    zone_id: int
    row_start: int
    row_stop: int
    col_start: int
    col_stop: int

    @property
    def row_range(self) -> tuple[int, int]:
        return self.row_start, self.row_stop

    @property
    def col_range(self) -> tuple[int, int]:
        return self.col_start, self.col_stop

    @property
    def centroid(self) -> tuple[int, int]:
        return (
            (self.row_start + self.row_stop) // 2,
            (self.col_start + self.col_stop) // 2,
        )


def build_zone_boundaries(
    height: int,
    width: int,
    *,
    rows: int = DEFAULT_ZONE_ROWS,
    cols: int = DEFAULT_ZONE_COLS,
) -> tuple[ZoneBounds, ...]:
    """Partition a grid into equal-count rectangular macro-zones.

    Integer division assigns every cell exactly once, including remainder
    rows and columns when the grid shape is not divisible by the zone count.
    """
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    if rows > height or cols > width:
        raise ValueError("zone grid cannot contain more rows/cols than cells")

    zones: list[ZoneBounds] = []
    zone_id = 0
    for row_index in range(rows):
        row_start = row_index * height // rows
        row_stop = (row_index + 1) * height // rows
        for col_index in range(cols):
            col_start = col_index * width // cols
            col_stop = (col_index + 1) * width // cols
            zones.append(
                ZoneBounds(
                    zone_id=zone_id,
                    row_start=row_start,
                    row_stop=row_stop,
                    col_start=col_start,
                    col_stop=col_stop,
                )
            )
            zone_id += 1
    return tuple(zones)


def zone_id_for_cell(
    row: int,
    col: int,
    height: int,
    width: int,
    *,
    rows: int = DEFAULT_ZONE_ROWS,
    cols: int = DEFAULT_ZONE_COLS,
) -> int:
    """Return the canonical zone containing a validated grid cell."""
    if not 0 <= row < height or not 0 <= col < width:
        raise ValueError(f"cell ({row}, {col}) is outside {height}x{width} grid")
    zone_row = min(row * rows // height, rows - 1)
    zone_col = min(col * cols // width, cols - 1)
    return zone_row * cols + zone_col
