"""Build a non-executable schema-one scenario from explicitly sourced local inputs.

INPUTS.json (paths relative to the manifest) has schema_version=1, name,
mode (``hindcast`` or explicitly ``synthetic``), start_utc, duration_s,
bbox_wgs84=[west,south,east,north], buffer_m, sources, landscape, weather,
roads, resources and optional seed/provenance. Hindcasts use the fixed Palisades
bbox (-118.605,34.030,-118.440,34.150) and 3000 m buffer; synthetic fixtures may
use smaller domains. All outputs use a densified, outward-snapped EPSG:5070
30 m envelope, with a separate cell-centre core_mask.

Each sources[id] contains path, sha256, provider, version, license,
retrieved_utc, crs (original CRS, or 'none' for scalar tables), units,
valid_from_utc and valid_until_utc. Paths are local files, not URLs. Optional
source_url/source_sha256 retain acquisition lineage. NPZ raster sources also
contain transform=[a,b,c,d,e,f]; NPZ vectors declare
vector_basis='earth_east_north'. Source validity describes data applicability,
not download/publication time. Every local file is checked before decoding.
Mixed source units are a dictionary keyed by NPZ/JSON variable or GeoTIFF band
number as a string. Each reference must agree exactly with its original units.

landscape maps elevation, fuel_model, canopy_cover, canopy_height,
canopy_base_height, canopy_bulk_density and asset_value to references:
{source, band:1, units, scale, offset} for GeoTIFF, or
{source, variable, units, scale, offset[, time_index]} for NPZ/JSON.
Units name original values; scale and offset explicitly convert to SI (fuel
code, m, fraction, kg/m3, and nonnegative asset value). Terrain slope is rise/run
and aspect is downslope clockwise from grid north, derived from elevation.

weather is a nonempty list of contiguous hourly records with start_utc,
end_utc, u_wind, v_wind, temperature_c and all six moisture contract fields.
References use the same form; JSON numeric values and NPZ scalar/time-series
moisture values broadcast, not implicit meteorological spatial fallbacks.
Wind references require earth-relative east/north m/s, are reprojected as
components, rotated into grid coordinates, then converted to wind-FROM degrees.
No directions are interpolated. Missing bands, nodata, or hashes never receive
synthetic replacements. provenance.moisture must identify method ('nfdrs4' or
'explicit_priors'); explicit priors also require intervals and rationale and
always label the resulting historical inputs not ready.
The prior schema is provenance.moisture.uncertainty_manifest (or shared
provenance.uncertainty_manifest), containing moisture_priors[field] =
{distribution:'uniform', low, high, provenance}, with a nondegenerate,
nonnegative interval for every moisture field. Explicit prepared values must
lie within these ranges; they remain assumed_prior, not calibrated estimates.
Hindcast priors additionally require uncertainty_manifest.wind_priors entries
wind_speed_m_s and upwind_direction_deg in the same form. Sampling belongs to
the ensemble layer; preparation never substitutes an interval midpoint.

roads={source, crs:'EPSG:5070', buffer_m:20000, as_of_utc,
coverage_bounds:[xmin,ymin,xmax,ymax]} references JSON containing directed
nodes/edges in the existing road contract. Coverage must enclose the entire
buffered landscape plus 20 km. Hindcasts reject road snapshots after ignition.
resources is the existing explicit resource-contract list; no resources,
assets, roads, moisture, or weather are fabricated by this builder.

CLI: python -m src.world.prepare --inputs INPUTS.json --out PATH
"""

import argparse
from datetime import timedelta
import hashlib
import math
from pathlib import Path
import numpy as np
from pyproj import CRS, Proj, Transformer
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import Affine
from rasterio.warp import reproject
from shapely.geometry import Polygon, mapping

from .contracts import Scenario, WEATHER_KEYS, _utc
from .scenario import _decode_json, save_scenario

PALISADES_BBOX = (-118.605, 34.030, -118.440, 34.150)
CELL_SIZE_M = 30.0
LANDSCAPE_FIELDS = (
    'elevation', 'fuel_model', 'canopy_cover', 'canopy_height',
    'canopy_base_height', 'canopy_bulk_density', 'asset_value',
)
MOISTURE_FIELDS = WEATHER_KEYS[3:]
_SOURCE_REQUIRED = {
    'path', 'sha256', 'provider', 'version', 'license', 'retrieved_utc',
    'crs', 'units', 'valid_from_utc', 'valid_until_utc',
}


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{label} must be finite numeric data')
    return float(value)


def _grid(bbox, buffer_m):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError('bbox_wgs84 must be [west,south,east,north]')
    west, south, east, north = [_finite(v, 'bbox_wgs84') for v in bbox]
    if not (-180 <= west < east <= 180 and -90 < south < north < 90):
        raise ValueError('invalid bbox_wgs84')
    # Densify every geographic edge before projection; corner-only envelopes
    # lose the curved boundary of long parallels in Albers equal area.
    t = np.linspace(0, 1, 257)
    points = np.concatenate([
        np.column_stack((west + (east-west)*t, np.full_like(t, south))),
        np.column_stack((np.full_like(t, east), south + (north-south)*t)),
        np.column_stack((east - (east-west)*t, np.full_like(t, north))),
        np.column_stack((np.full_like(t, west), north - (north-south)*t)),
    ])
    x, y = Transformer.from_crs(4326, 5070, always_xy=True).transform(points[:, 0], points[:, 1])
    core = Polygon(np.column_stack((x, y)))
    xmin, ymin, xmax, ymax = core.bounds
    xmin, ymin, xmax, ymax = xmin-buffer_m, ymin-buffer_m, xmax+buffer_m, ymax+buffer_m
    xmin, ymin = math.floor(xmin/30)*30, math.floor(ymin/30)*30
    xmax, ymax = math.ceil(xmax/30)*30, math.ceil(ymax/30)*30
    transform = Affine(30, 0, xmin, 0, -30, ymax)
    shape = (int((ymax-ymin)/30), int((xmax-xmin)/30))
    core_mask = geometry_mask([mapping(core)], shape, transform, invert=True)
    return transform, shape, core_mask, (xmin, ymin, xmax, ymax)


class _Inputs:
    def __init__(self, manifest, directory, start):
        self.sources = manifest.get('sources')
        if not isinstance(self.sources, dict) or not self.sources:
            raise ValueError('sources must document every local input')
        self.paths = {}
        self.arrays = {}
        self.tables = {}
        for key, source in self.sources.items():
            if not isinstance(source, dict) or not _SOURCE_REQUIRED.issubset(source):
                raise ValueError(f'source {key} requires {sorted(_SOURCE_REQUIRED)}')
            for name in ('provider', 'version', 'license', 'crs', 'path'):
                if not isinstance(source[name], str) or not source[name].strip():
                    raise ValueError(f'source {key}.{name} must be explicit text')
            if not isinstance(source['units'], (str, dict)) or not source['units']:
                raise ValueError(f'source {key}.units must be explicit')
            if source['crs'] != 'none':
                CRS.from_user_input(source['crs'])
            _utc(source['retrieved_utc'], f'{key}.retrieved_utc')
            begin = _utc(source['valid_from_utc'], f'{key}.valid_from_utc')
            end = _utc(source['valid_until_utc'], f'{key}.valid_until_utc')
            if begin >= end:
                raise ValueError(f'source {key} has empty validity interval')
            digest = source['sha256']
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError(f'source {key} sha256 must be lowercase SHA-256')
            path = Path(source['path'])
            if not path.is_absolute():
                path = directory / path
            if not path.is_file():
                raise ValueError(f'source {key} is not a local file: {path}')
            with path.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != digest:
                raise ValueError(f'source {key} SHA-256 mismatch')
            self.paths[key] = path
        self.start = start

    def source(self, ref, begin, end):
        if not isinstance(ref, dict) or not {'source', 'units', 'scale', 'offset'}.issubset(ref):
            raise ValueError('field reference requires source, units, scale and offset')
        key = ref['source']
        if key not in self.sources:
            raise ValueError(f'unknown source {key}')
        source = self.sources[key]
        if (_utc(source['valid_from_utc'], key) > begin or
                _utc(source['valid_until_utc'], key) <= begin or
                _utc(source['valid_until_utc'], key) < end):
            raise ValueError(f'source {key} does not cover requested validity interval')
        variable = ref.get('variable')
        unit_key = variable if variable is not None else str(ref.get('band'))
        declared_units = source['units'].get(unit_key) if isinstance(source['units'], dict) else source['units']
        if declared_units != ref['units']:
            raise ValueError(f'source {key} units disagree with field reference')
        _finite(ref['scale'], 'scale')
        _finite(ref['offset'], 'offset')
        if ref['scale'] <= 0:
            raise ValueError('scale must be positive')
        return key, source

    def read(self, ref, begin, end):
        key, source = self.source(ref, begin, end)
        path = self.paths[key]
        if path.suffix.lower() == '.npz':
            if key not in self.arrays:
                with np.load(path, allow_pickle=False) as archive:
                    self.arrays[key] = {name: archive[name] for name in archive.files}
            variable = ref.get('variable')
            if variable not in self.arrays[key]:
                raise ValueError(f'source {key} missing NPZ variable {variable}')
            data = self.arrays[key][variable]
            if 'time_index' in ref:
                index = ref['time_index']
                if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(data):
                    raise ValueError('invalid NPZ time_index')
                # Acquisition time coordinates must agree with declared intervals.
                times = self.arrays[key].get('time_utc')
                if times is not None and _utc(str(times[index]), 'NPZ time_utc') != begin:
                    raise ValueError('NPZ time_utc disagrees with weather interval')
                data = data[index]
            if data.dtype.kind not in 'biuf':
                raise ValueError('input arrays must be numeric and non-executable')
            data = np.asarray(data, dtype=np.float64)
            valid = np.isfinite(data)
            if 'nodata' in source:
                valid &= data != source['nodata']
            transform = source.get('transform', source.get('native_transform'))
            return data, valid, source['crs'], Affine(*transform) if transform is not None else None
        if path.suffix.lower() == '.json':
            if key not in self.tables:
                self.tables[key] = _decode_json(path.read_bytes(), f'source {key}')
            value = self.tables[key].get(ref.get('variable'))
            if isinstance(value, list) and 'time_index' in ref:
                index = ref['time_index']
                if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(value):
                    raise ValueError('invalid JSON time_index')
                value = value[index]
            return np.asarray(_finite(value, 'JSON scalar')), np.asarray(True), source['crs'], None
        band = ref.get('band')
        if isinstance(band, bool) or not isinstance(band, int) or band < 1:
            raise ValueError('raster reference requires a positive band')
        with rasterio.open(path) as raster:
            if raster.crs is None or CRS(raster.crs) != CRS(source['crs']):
                raise ValueError(f'source {key} CRS disagrees with raster')
            if band > raster.count:
                raise ValueError(f'source {key} missing band {band}')
            data = raster.read(band, masked=True)
            valid = ~np.ma.getmaskarray(data) & np.isfinite(data.data)
            return np.asarray(data.data, dtype=np.float64), valid, raster.crs, raster.transform


def _units(ref, field):
    if not isinstance(ref, dict):
        raise ValueError(f'{field} requires an explicit source reference')
    # Conversion is explicit, and unsupported unit/scale combinations fail rather
    # than silently accepting dimensionally plausible but numerically wrong data.
    options = {
        'elevation': [('m', 1, 0)], 'fuel_model': [('code', 1, 0)],
        'canopy_cover': [('fraction', 1, 0), ('percent', .01, 0)],
        'canopy_height': [('m', 1, 0), ('decimeter', .1, 0)],
        'canopy_base_height': [('m', 1, 0), ('decimeter', .1, 0)],
        'canopy_bulk_density': [('kg/m3', 1, 0), ('hundredth kg/m3', .01, 0)],
        'asset_value': [('value', 1, 0), ('USD', 1, 0)],
        'u_wind': [('m/s', 1, 0)], 'v_wind': [('m/s', 1, 0)],
        'temperature_c': [('C', 1, 0), ('K', 1, -273.15)],
    }
    allowed = options.get(field, [('fraction', 1, 0), ('percent', .01, 0)])
    if not any(ref.get('units') == unit and ref.get('scale') == scale and ref.get('offset') == offset
               for unit, scale, offset in allowed):
        raise ValueError(f'{field} has unsupported units/scale/offset')


def _field(inputs, ref, field, begin, end, transform, shape):
    _units(ref, field)
    data, valid, crs, native_transform = inputs.read(ref, begin, end)
    if data.ndim == 0:
        if field not in MOISTURE_FIELDS:
            raise ValueError(f'{field} must have an explicit spatial raster')
        return np.full(shape, float(data)*ref['scale']+ref['offset']), np.full(shape, bool(valid))
    if data.ndim != 2 or native_transform is None:
        raise ValueError(f'{field} requires a two-dimensional georeferenced raster')
    method = Resampling.nearest if field == 'fuel_model' else Resampling.bilinear
    destination = np.zeros(shape, dtype=np.float64)
    support = np.zeros(shape, dtype=np.float64)
    kwargs = dict(src_transform=native_transform, src_crs=crs,
                  dst_transform=transform, dst_crs='EPSG:5070', resampling=method)
    if method == Resampling.bilinear:
        # An explicit invalid collar prevents GDAL from renormalizing partial
        # coverage-edge interpolation onto the last available source pixel.
        data = np.pad(data, 1, constant_values=0)
        valid = np.pad(valid, 1, constant_values=False)
        kwargs['src_transform'] = native_transform * Affine.translation(-1, -1)
    reproject(np.where(valid, data, 0), destination, **kwargs)
    # Never let nodata disappear through GDAL interpolation renormalization.
    # Every nonzero contributing weight must be valid.
    reproject(valid.astype(np.float64), support, **kwargs)
    return destination*ref['scale']+ref['offset'], support >= 1-1e-12


def _terrain(elevation, valid):
    # Central differences need all four neighbours; perimeter cells have no
    # defensible derivative and remain invalid rather than wrapping or filling.
    safe = np.zeros_like(valid)
    safe[1:-1, 1:-1] = (valid[1:-1, 1:-1] & valid[:-2, 1:-1] &
                        valid[2:, 1:-1] & valid[1:-1, :-2] & valid[1:-1, 2:])
    north_derivative, east_derivative = np.gradient(elevation, -30, 30)
    slope = np.hypot(east_derivative, north_derivative)
    aspect = np.degrees(np.arctan2(-east_derivative, -north_derivative)) % 360
    aspect[slope == 0] = 0
    return slope, aspect, safe


def _roads(inputs, spec, bounds, start, hindcast):
    if not isinstance(spec, dict) or spec.get('crs') != 'EPSG:5070' or spec.get('buffer_m') != 20000:
        raise ValueError('roads require EPSG:5070 and a 20000 m acquisition buffer')
    key = spec.get('source')
    if key not in inputs.sources:
        raise ValueError('roads require a hashed local source')
    source = inputs.sources[key]
    if CRS(source['crs']) != CRS(5070):
        raise ValueError('road source must use EPSG:5070')
    as_of = _utc(spec.get('as_of_utc'), 'roads.as_of_utc')
    if hindcast and as_of > start:
        raise ValueError('hindcast road snapshot must be pre-event')
    if not (_utc(source['valid_from_utc'], key) <= start < _utc(source['valid_until_utc'], key)):
        raise ValueError('road source does not cover scenario ignition')
    coverage = spec.get('coverage_bounds')
    if not isinstance(coverage, list) or len(coverage) != 4:
        raise ValueError('roads require documented coverage_bounds')
    coverage = [_finite(v, 'road coverage') for v in coverage]
    xmin, ymin, xmax, ymax = bounds
    if not (coverage[0] <= xmin-20000 and coverage[1] <= ymin-20000 and
            coverage[2] >= xmax+20000 and coverage[3] >= ymax+20000):
        raise ValueError('road coverage must extend 20 km beyond buffered landscape')
    return _decode_json(inputs.paths[key].read_bytes(), 'road graph')


def build(inputs_path: Path, out: Path) -> Scenario:
    """Validate local provenance, prepare conservative rasters and persist once."""
    inputs_path, out = Path(inputs_path), Path(out)
    if out.exists():
        raise FileExistsError(f'scenario destination already exists: {out}')
    raw = inputs_path.read_bytes()
    manifest = _decode_json(raw, 'INPUTS manifest')
    if manifest.get('schema_version') != 1 or manifest.get('mode') not in {'hindcast', 'synthetic'}:
        raise ValueError('INPUTS requires schema_version=1 and mode=hindcast or synthetic')
    start = _utc(manifest.get('start_utc'), 'start_utc')
    duration = manifest.get('duration_s')
    if isinstance(duration, bool) or not isinstance(duration, int) or duration <= 0:
        raise ValueError('duration_s must be a positive integer')
    bbox, buffer_m = manifest.get('bbox_wgs84'), _finite(manifest.get('buffer_m'), 'buffer_m')
    if buffer_m < 0:
        raise ValueError('buffer_m cannot be negative')
    hindcast = manifest['mode'] == 'hindcast'
    if hindcast and (bbox != list(PALISADES_BBOX) or buffer_m != 3000):
        raise ValueError('hindcast requires the Palisades bbox and 3000 m landscape buffer')
    transform, shape, core, bounds = _grid(bbox, buffer_m)
    inputs = _Inputs(manifest, inputs_path.parent, start)
    valid = np.ones(shape, dtype=bool)
    landscape = {}
    for field in LANDSCAPE_FIELDS:
        ref = manifest.get('landscape', {}).get(field)
        if ref is None:
            raise ValueError(f'missing landscape field {field}')
        landscape[field], field_valid = _field(inputs, ref, field, start, start, transform, shape)
        valid &= field_valid
    slope, aspect, terrain_valid = _terrain(landscape['elevation'], valid)
    valid &= terrain_valid
    landscape.update(slope=slope, aspect=aspect, core_mask=core)
    fuel = landscape['fuel_model']
    if not np.equal(fuel[valid], np.floor(fuel[valid])).all():
        raise ValueError('fuel_model contains fractional categorical codes')
    landscape['fuel_model'] = np.where(valid, fuel, 0).astype(np.int32)
    records = manifest.get('weather')
    if not isinstance(records, list) or not records:
        raise ValueError('weather must contain hourly validity records')
    weather = {field: [] for field in WEATHER_KEYS}
    intervals = []
    previous = start
    rows, cols = np.indices(shape)
    x = transform.c + (cols+.5)*30
    y = transform.f - (rows+.5)*30
    lon, lat = Transformer.from_crs(5070, 4326, always_xy=True).transform(x, y)
    convergence = np.radians(Proj('EPSG:5070').get_factors(lon, lat).meridian_convergence)
    for record in records:
        begin = _utc(record.get('start_utc'), 'weather.start_utc')
        end = _utc(record.get('end_utc'), 'weather.end_utc')
        if begin != previous or end-begin != timedelta(hours=1):
            raise ValueError('weather records must be contiguous hourly intervals beginning at ignition')
        values = {}
        for field in ('u_wind', 'v_wind', 'temperature_c', *MOISTURE_FIELDS):
            ref = record.get(field)
            if field in {'u_wind', 'v_wind'}:
                if inputs.sources.get(ref.get('source') if isinstance(ref, dict) else None, {}).get('vector_basis') != 'earth_east_north':
                    raise ValueError('wind requires declared earth_east_north vector basis')
            values[field], field_valid = _field(inputs, ref, field, begin, end, transform, shape)
            valid &= field_valid
        u, v = values.pop('u_wind'), values.pop('v_wind')
        grid_u = u*np.cos(convergence)-v*np.sin(convergence)
        grid_v = u*np.sin(convergence)+v*np.cos(convergence)
        values['wind_speed_m_s'] = np.hypot(grid_u, grid_v)
        values['upwind_direction_deg'] = np.degrees(np.arctan2(-grid_u, -grid_v)) % 360
        values['upwind_direction_deg'][values['wind_speed_m_s'] == 0] = 0
        for field in WEATHER_KEYS:
            weather[field].append(values[field])
        intervals.append({'start_utc': record['start_utc'], 'end_utc': record['end_utc']})
        previous = end
    if previous < start + timedelta(seconds=duration):
        raise ValueError('hourly weather does not cover scenario duration')
    if not np.any(valid & core):
        raise ValueError('no conservatively valid cells remain in the core study area')
    landscape['valid'] = valid
    roads = _roads(inputs, manifest.get('roads'), bounds, start, hindcast)
    provenance = dict(manifest.get('provenance', {}))
    moisture = provenance.get('moisture')
    if not isinstance(moisture, dict) or moisture.get('method') not in {'nfdrs4', 'explicit_priors'}:
        raise ValueError('provenance.moisture requires nfdrs4 or explicit_priors method')
    if moisture['method'] == 'explicit_priors' and (not moisture.get('intervals') or not moisture.get('rationale')):
        raise ValueError('explicit moisture priors require intervals and rationale')
    if moisture['method'] == 'explicit_priors':
        uncertainty = moisture.get('uncertainty_manifest', provenance.get('uncertainty_manifest', {}))
        if not isinstance(uncertainty, dict):
            raise ValueError('uncertainty_manifest must be an object')
        moisture_priors = uncertainty.get('moisture_priors', {})
        wind_priors = uncertainty.get('wind_priors', {})
        if not isinstance(moisture_priors, dict) or not isinstance(wind_priors, dict):
            raise ValueError('uncertainty prior collections must be objects')
        required = dict(moisture_priors)
        if hindcast:
            required.update({field: wind_priors.get(field)
                             for field in ('wind_speed_m_s', 'upwind_direction_deg')})
        for field in (*MOISTURE_FIELDS, *(('wind_speed_m_s', 'upwind_direction_deg') if hindcast else ())):
            prior = required.get(field)
            if not isinstance(prior, dict) or prior.get('distribution') != 'uniform' or not prior.get('provenance'):
                raise ValueError(f'explicit provenance-tagged uniform interval prior required: {field}')
            low, high = _finite(prior.get('low'), field), _finite(prior.get('high'), field)
            if low < 0 or high <= low or (field == 'upwind_direction_deg' and high > 360):
                raise ValueError(f'nondegenerate physical prior interval required: {field}')
            if field in MOISTURE_FIELDS:
                if any(((band[valid] < low).any() or (band[valid] > high).any())
                       for band in weather[field]):
                    raise ValueError(f'prepared {field} values lie outside declared prior interval')
        moisture = dict(moisture, moisture_source='assumed_prior')
        provenance['moisture'] = moisture
        provenance['uncertainty_manifest'] = uncertainty
    provenance.update(
        sources=inputs.sources,
        source_hashes={key: source['sha256'] for key, source in inputs.sources.items()},
        inputs_manifest_sha256=hashlib.sha256(raw).hexdigest(),
        weather_intervals=intervals, mode=manifest['mode'],
        input_fields={'landscape': manifest['landscape'], 'weather': records},
        historical_ready=False,
        historical_readiness_note='Preparation is not historical validation; explicit moisture priors are not spun-up estimates.'
        if moisture['method'] == 'explicit_priors' else 'Preparation alone does not establish historical validation.',
        preparation={'crs': 'EPSG:5070', 'cell_size_m': 30, 'bbox_wgs84': bbox,
                     'buffer_m': buffer_m, 'core_mask': 'cell-centre within densified WGS84 bbox',
                     'categorical_resampling': 'nearest', 'continuous_resampling': 'bilinear',
                     'nodata': 'all contributing weights valid; terrain requires four neighbours',
                     'slope_units': 'rise/run', 'aspect_units': 'degrees clockwise grid north',
                     'wind_basis': 'EPSG:5070 grid east/north; direction is wind-from',
                     'roads': manifest['roads']},
    )
    scenario = Scenario(
        name=manifest.get('name'), crs='EPSG:5070', transform=tuple(transform)[:6],
        start_utc=manifest['start_utc'], duration_s=duration, landscape=landscape,
        weather={field: np.stack(values) for field, values in weather.items()},
        weather_step_s=3600, roads=roads, resources=manifest.get('resources'),
        provenance=provenance, seed=manifest.get('seed', 0),
    )
    scenario.validate()
    save_scenario(scenario, out)
    return scenario


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    build(args.inputs, args.out)


if __name__ == '__main__':
    main()
