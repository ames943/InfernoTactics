"""Behavioral local preparation fixtures; all generated inputs are synthetic."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from pyproj import Proj, Transformer
import rasterio
from rasterio.transform import Affine

from src.world.contracts import WEATHER_KEYS
from src.world.prepare import (
    LANDSCAPE_FIELDS, PALISADES_BBOX, _Inputs, _field, _grid, _roads, build,
)
from src.world.scenario import load_scenario


def make_inputs(directory: Path, *, full_domain=False, hours=2) -> Path:
    """Write a usable, explicitly synthetic INPUTS.json for CLI/build smoke runs.

    full_domain=True exercises the exact Palisades bbox, 3 km landscape buffer,
    and declared road coverage extending a further 20 km. No measured incident
    inputs or scientific readiness are implied. The fixture intentionally has
    no resources; callers exercising operations must provide their own units.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    bbox = list(PALISADES_BBOX) if full_domain else [-118.53, 34.08, -118.525, 34.085]
    buffer_m = 3000 if full_domain else 120
    transform, shape, _, bounds = _grid(bbox, buffer_m)
    # Explicit coverage collar permits interpolation/terrain at target edges.
    source_transform = transform * Affine.translation(-2, -2)
    source_shape = (shape[0]+4, shape[1]+4)
    rows, cols = np.indices(source_shape)
    values = {
        'elevation': 100 + cols.astype(float)*3,
        'fuel_model': np.where(cols < source_shape[1]//2, 101, 102),
        'canopy_cover': np.full(source_shape, 20.),
        'canopy_height': np.full(source_shape, 100.),
        'canopy_base_height': np.full(source_shape, 20.),
        'canopy_bulk_density': np.full(source_shape, 10.),
        'asset_value': np.zeros(source_shape),
    }
    units = dict(zip(LANDSCAPE_FIELDS,
                     ['m', 'code', 'percent', 'decimeter', 'decimeter', 'hundredth kg/m3', 'value']))
    scales = dict(zip(LANDSCAPE_FIELDS, [1, 1, .01, .1, .1, .01, 1]))
    raster_path = directory / 'landscape.tif'
    with rasterio.open(raster_path, 'w', driver='GTiff', width=source_shape[1],
                       height=source_shape[0], count=7, dtype='float64',
                       crs='EPSG:5070', transform=source_transform, nodata=-9999) as raster:
        for index, field in enumerate(LANDSCAPE_FIELDS, 1):
            raster.write(values[field].astype(np.float64), index)
    from datetime import datetime, timedelta, timezone
    start = datetime(2025, 1, 7, 18, tzinfo=timezone.utc)
    iso = lambda dt: dt.isoformat().replace('+00:00', 'Z')
    times = [iso(start+timedelta(hours=i)) for i in range(hours)]
    end = iso(start+timedelta(hours=hours))
    weather_path = directory / 'weather.npz'
    np.savez_compressed(weather_path,
        u_wind=np.stack([np.full(source_shape, 2.+i) for i in range(hours)]),
        v_wind=np.stack([np.full(source_shape, -4.) for _ in range(hours)]),
        temperature_k=np.stack([np.full(source_shape, 293.15+i) for i in range(hours)]),
        time_utc=np.asarray(times))
    moisture_path = directory / 'moisture.json'
    moisture_path.write_text(json.dumps({field: .1 if 'dead' in field else .8
                                         for field in WEATHER_KEYS[3:]}))
    roads_path = directory / 'roads.json'
    roads_path.write_text(json.dumps({'nodes': [], 'edges': []}))

    def metadata(path, source_units, crs='EPSG:5070'):
        return {'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'provider': 'explicit synthetic fixture', 'version': 'fixture-1',
                'license': 'CC0 synthetic fixture', 'retrieved_utc': '2025-01-01T00:00:00Z',
                'crs': crs, 'units': source_units,
                'valid_from_utc': '2024-01-01T00:00:00Z', 'valid_until_utc': end}

    sources = {'land': metadata(raster_path, 'mixed'),
               'wx': metadata(weather_path, {'u_wind': 'm/s', 'v_wind': 'm/s', 'temperature_k': 'K'}),
               'moisture': metadata(moisture_path, 'fraction', 'none'),
               'roads': metadata(roads_path, 'm')}
    # Multiple original units are explicit per source variable/band, not hidden.
    sources['land']['units'] = {str(i): units[field] for i, field in enumerate(LANDSCAPE_FIELDS, 1)}
    sources['wx'].update(transform=list(source_transform)[:6], vector_basis='earth_east_north')
    landscape = {field: {'source': 'land', 'band': i, 'units': units[field],
                         'scale': scales[field], 'offset': 0}
                 for i, field in enumerate(LANDSCAPE_FIELDS, 1)}
    weather = []
    for i, time in enumerate(times):
        record = {'start_utc': time, 'end_utc': iso(start+timedelta(hours=i+1))}
        for field, variable, unit, offset in [('u_wind', 'u_wind', 'm/s', 0),
                ('v_wind', 'v_wind', 'm/s', 0), ('temperature_c', 'temperature_k', 'K', -273.15)]:
            record[field] = {'source': 'wx', 'variable': variable, 'time_index': i,
                             'units': unit, 'scale': 1, 'offset': offset}
        for field in WEATHER_KEYS[3:]:
            record[field] = {'source': 'moisture', 'variable': field,
                             'units': 'fraction', 'scale': 1, 'offset': 0}
        weather.append(record)
    xmin, ymin, xmax, ymax = bounds
    manifest = {'schema_version': 1, 'name': 'synthetic local preparation fixture',
                'mode': 'synthetic', 'start_utc': times[0], 'duration_s': hours*3600,
                'bbox_wgs84': bbox, 'buffer_m': buffer_m,
                'sources': sources, 'landscape': landscape, 'weather': weather,
                'roads': {'source': 'roads', 'crs': 'EPSG:5070', 'buffer_m': 20000,
                          'as_of_utc': '2025-01-01T00:00:00Z',
                          'coverage_bounds': [xmin-20000, ymin-20000, xmax+20000, ymax+20000]},
                'resources': [], 'seed': 4,
                'provenance': {'moisture': {'method': 'explicit_priors',
                    'intervals': [{'start_utc': times[0], 'end_utc': end}],
                    'rationale': 'Synthetic regression fixture only, not measured moisture.'}}}
    manifest['provenance']['moisture']['uncertainty_manifest'] = {
        'moisture_priors': {field: {'distribution': 'uniform',
            'low': .05 if 'dead' in field else .5,
            'high': .2 if 'dead' in field else 1.2,
            'provenance': 'Explicit synthetic interval, not calibrated.'}
            for field in WEATHER_KEYS[3:]}}
    inputs = directory / 'INPUTS.json'
    inputs.write_text(json.dumps(manifest))
    return inputs


def _change(inputs, change):
    manifest = json.loads(inputs.read_text())
    change(manifest)
    inputs.write_text(json.dumps(manifest))


def test_build_roundtrip_hourly_si_core_and_vectors(tmp_path):
    path = make_inputs(tmp_path / 'inputs')
    scenario = build(path, tmp_path / 'scenario')
    valid = scenario.landscape['valid']
    assert valid.any()
    assert scenario.crs == 'EPSG:5070'
    assert scenario.transform[0] == 30 and scenario.transform[4] == -30
    assert not scenario.landscape['core_mask'].all()
    assert np.allclose(scenario.landscape['canopy_cover'][valid], .2)
    assert np.allclose(scenario.landscape['canopy_height'][valid], 10)
    assert np.allclose(scenario.landscape['canopy_base_height'][valid], 2)
    assert np.allclose(scenario.landscape['canopy_bulk_density'][valid], .1)
    assert np.allclose(scenario.landscape['slope'][valid], .1)
    assert np.allclose(scenario.landscape['aspect'][valid], 270)
    assert set(np.unique(scenario.landscape['fuel_model'][valid])) == {101, 102}
    assert np.allclose(scenario.weather['temperature_c'][0][valid], 20)
    assert np.allclose(scenario.weather['temperature_c'][1][valid], 21)
    assert np.allclose(scenario.weather['wind_speed_m_s'][0][valid], np.sqrt(20))
    row, col = np.argwhere(valid)[len(np.argwhere(valid))//2]
    a, _, c, _, e, f = scenario.transform
    lon, lat = Transformer.from_crs(5070, 4326, always_xy=True).transform(c+(col+.5)*a, f+(row+.5)*e)
    gamma = np.radians(Proj('EPSG:5070').get_factors(lon, lat).meridian_convergence)
    u, v = 2*np.cos(gamma)+4*np.sin(gamma), 2*np.sin(gamma)-4*np.cos(gamma)
    assert scenario.weather['upwind_direction_deg'][0, row, col] == pytest.approx(np.degrees(np.arctan2(-u, -v)) % 360)
    assert scenario.provenance['historical_ready'] is False
    loaded = load_scenario(tmp_path / 'scenario')
    np.testing.assert_array_equal(loaded.landscape['valid'], valid)
    assert loaded.provenance['source_hashes'] == scenario.provenance['source_hashes']
    assert loaded.provenance['moisture']['moisture_source'] == 'assumed_prior'
    assert set(loaded.provenance['uncertainty_manifest']['moisture_priors']) == set(WEATHER_KEYS[3:])


def test_full_palisades_envelope_contains_densified_core_and_3km_buffer():
    transform, shape, core, bounds = _grid(PALISADES_BBOX, 3000)
    xmin, ymin, xmax, ymax = bounds
    t = np.linspace(0, 1, 1001)
    w, s, e, n = PALISADES_BBOX
    lon = np.r_[w+(e-w)*t, np.full_like(t, e), e-(e-w)*t, np.full_like(t, w)]
    lat = np.r_[np.full_like(t, s), s+(n-s)*t, np.full_like(t, n), n-(n-s)*t]
    x, y = Transformer.from_crs(4326, 5070, always_xy=True).transform(lon, lat)
    assert xmin <= x.min()-3000 and xmax >= x.max()+3000
    assert ymin <= y.min()-3000 and ymax >= y.max()+3000
    assert all(value % 30 == 0 for value in bounds)
    assert core.any() and not core.all() and shape == core.shape
    assert transform.a == 30


def test_bad_checksum_fails_before_output(tmp_path):
    path = make_inputs(tmp_path / 'inputs')
    with (path.parent / 'weather.npz').open('ab') as stream:
        stream.write(b'tampered')
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        build(path, tmp_path / 'scenario')
    assert not (tmp_path / 'scenario').exists()


@pytest.mark.parametrize('mutation,match', [
    (lambda m: m['sources']['land'].pop('version'), 'requires'),
    (lambda m: m['sources']['land'].pop('valid_from_utc'), 'requires'),
    (lambda m: m['landscape']['canopy_cover'].update(scale=1), 'unsupported units'),
    (lambda m: m['weather'][1].update(start_utc='2025-01-07T20:00:00Z'), 'contiguous'),
    (lambda m: m['weather'].pop(), 'cover scenario'),
    (lambda m: m['sources']['wx'].update(vector_basis='grid_relative'), 'vector basis'),
    (lambda m: m['roads'].update(buffer_m=1000), '20000'),
    (lambda m: m['roads'].update(coverage_bounds=[0, 0, 1, 1]), '20 km'),
    (lambda m: m['provenance'].pop('moisture'), 'moisture'),
    (lambda m: m.update(mode='hindcast'), 'Palisades'),
])
def test_invalid_manifest_rejected(tmp_path, mutation, match):
    path = make_inputs(tmp_path / 'inputs')
    _change(path, mutation)
    with pytest.raises(ValueError, match=match):
        build(path, tmp_path / 'scenario')


def test_nodata_propagates_weather_and_terrain_validity(tmp_path):
    path = make_inputs(tmp_path / 'inputs')
    raster_path = path.parent / 'landscape.tif'
    with rasterio.open(raster_path, 'r+') as raster:
        elevation = raster.read(1)
        elevation[10, 10] = -9999
        raster.write(elevation, 1)
    def update(manifest):
        manifest['sources']['land']['sha256'] = hashlib.sha256(raster_path.read_bytes()).hexdigest()
    _change(path, update)
    scenario = build(path, tmp_path / 'scenario')
    # The source collar shifts source [10,10] to destination [8,8].
    assert not scenario.landscape['valid'][8, 8]
    assert not scenario.landscape['valid'][8, 9]
    assert scenario.landscape['valid'][8, 10]


def test_bilinear_nodata_does_not_renormalize_and_categories_are_nearest():
    class Native:
        def read(self, *args):
            return (np.array([[101., 102.], [101., 102.]]),
                    np.array([[True, False], [True, True]]), 'EPSG:5070', Affine(30, 0, 0, 0, -30, 60))
    target = Affine(30, 0, 15, 0, -30, 45)
    ref = {'source': 'x', 'band': 1, 'units': 'm', 'scale': 1, 'offset': 0}
    _, valid = _field(Native(), ref, 'elevation', None, None, target, (1, 1))
    assert not valid[0, 0]
    ref['units'] = 'code'
    categorical, valid = _field(Native(), ref, 'fuel_model', None, None, target, (1, 1))
    assert valid[0, 0] and categorical[0, 0] in {101, 102}


def test_wind_components_not_direction_angles_are_interpolated(tmp_path):
    path = make_inputs(tmp_path / 'inputs')
    with np.load(path.parent / 'weather.npz', allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    data['u_wind'][:] = .1
    data['u_wind'][:, :, ::2] = -.1
    data['v_wind'][:] = -10
    np.savez_compressed(path.parent / 'weather.npz', **data)
    _change(path, lambda m: m['sources']['wx'].update(
        sha256=hashlib.sha256((path.parent / 'weather.npz').read_bytes()).hexdigest(),
        transform=[30, 0, m['sources']['wx']['transform'][2]+15,
                   0, -30, m['sources']['wx']['transform'][5]]))
    scenario = build(path, tmp_path / 'scenario')
    valid = scenario.landscape['valid']
    assert np.allclose(scenario.weather['wind_speed_m_s'][:, valid], 10)
    # Averaging 359/1 degree winds must remain north, not turn south.
    directions = scenario.weather['upwind_direction_deg'][:, valid]
    assert np.all((directions < 45) | (directions > 315))


def test_pre_event_road_hindcast_gate(tmp_path):
    from src.world.contracts import _utc
    path = make_inputs(tmp_path / 'inputs')
    manifest = json.loads(path.read_text())
    start = _utc(manifest['start_utc'], 'start')
    inputs = _Inputs(manifest, path.parent, start)
    _, _, _, bounds = _grid(manifest['bbox_wgs84'], manifest['buffer_m'])
    manifest['roads']['as_of_utc'] = '2025-01-08T00:00:00Z'
    with pytest.raises(ValueError, match='pre-event'):
        _roads(inputs, manifest['roads'], bounds, start, True)


def test_existing_destination_is_not_overwritten(tmp_path):
    path = make_inputs(tmp_path / 'inputs')
    output = tmp_path / 'scenario'
    output.mkdir()
    (output / 'user-file').write_text('preserve')
    with pytest.raises(FileExistsError):
        build(path, output)
    assert (output / 'user-file').read_text() == 'preserve'


@pytest.mark.parametrize('change,match', [
    (lambda p: p.pop('foliar_moisture'), 'foliar_moisture'),
    (lambda p: p['fuel_moisture_dead_1hr'].update(low=.1, high=.1), 'nondegenerate'),
    (lambda p: p['fuel_moisture_dead_1hr'].update(low=-.1), 'physical'),
    (lambda p: p['fuel_moisture_dead_1hr'].update(low=.12), 'outside'),
    (lambda p: p['fuel_moisture_dead_1hr'].pop('provenance'), 'provenance-tagged'),
])
def test_explicit_prior_ranges_are_required_and_values_stay_within_them(tmp_path, change, match):
    path = make_inputs(tmp_path / 'inputs')
    _change(path, lambda m: change(m['provenance']['moisture']['uncertainty_manifest']['moisture_priors']))
    with pytest.raises(ValueError, match=match):
        build(path, tmp_path / 'scenario')


def test_hindcast_priors_require_wind_uncertainty(tmp_path, monkeypatch):
    # Override only the study geometry constant for this gate's small fixture;
    # production hindcasts still require the actual fixed Palisades domain.
    import src.world.prepare as prepare
    path = make_inputs(tmp_path / 'inputs')
    manifest = json.loads(path.read_text())
    monkeypatch.setattr(prepare, 'PALISADES_BBOX', tuple(manifest['bbox_wgs84']))
    _change(path, lambda m: m.update(mode='hindcast', buffer_m=3000))
    # Rebuild the small fixture with expanded geometry/coverage sources without
    # allocating a full historical-domain test merely to exercise the gate.
    _, _, _, bounds = _grid(manifest['bbox_wgs84'], 3000)
    _change(path, lambda m: m['roads'].update(coverage_bounds=[
        bounds[0]-20000, bounds[1]-20000, bounds[2]+20000, bounds[3]+20000]))
    with pytest.raises(ValueError, match='wind_speed_m_s'):
        build(path, tmp_path / 'missing-wind')
    def add_wind(m):
        m['provenance']['moisture']['uncertainty_manifest']['wind_priors'] = {
            'wind_speed_m_s': {'distribution': 'uniform', 'low': 0, 'high': 15,
                               'provenance': 'Explicit illustrative forecast error interval.'},
            'upwind_direction_deg': {'distribution': 'uniform', 'low': 0, 'high': 360,
                                    'provenance': 'Explicit illustrative full-circle uncertainty.'}}
    _change(path, add_wind)
    scenario = build(path, tmp_path / 'with-wind')
    assert scenario.provenance['historical_ready'] is False
    assert set(scenario.provenance['uncertainty_manifest']['wind_priors']) == {
        'wind_speed_m_s', 'upwind_direction_deg'}
