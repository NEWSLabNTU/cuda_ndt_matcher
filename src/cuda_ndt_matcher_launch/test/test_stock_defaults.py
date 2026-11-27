"""
cuda_ndt carries Autoware's defaults, so it drops in for ndt_scan_matcher.

Every value this package ships that Autoware's ndt_scan_matcher (or its input
chain) also has must equal the installed Autoware's: in the matcher's
parameter file, in the three preprocessing files, and in the matcher's own
compiled-in defaults (io/params.rs), which apply to a key a file leaves out.
Keys only cuda_ndt has (batch, divergence, ...) are not checked.

A deployment's tuning belongs in the deployment's own files, passed in as
launch arguments, never here.
"""

from pathlib import Path
import re

import pytest
import yaml

PKG = Path(__file__).parents[1]
CRATE = PKG.parents[0] / 'cuda_ndt_matcher' / 'src' / 'io' / 'params.rs'
PRE = 'pointcloud_preprocessor/'
# ours -> path under Autoware's ndt_scan_matcher config directory
FILES = {
    'cuda_scan_matcher.param.yaml': 'ndt_scan_matcher.param.yaml',
    PRE + 'crop_box_filter_measurement_range.param.yaml':
        PRE + 'crop_box_filter_measurement_range.param.yaml',
    PRE + 'voxel_grid_filter.param.yaml': PRE + 'voxel_grid_filter.param.yaml',
    PRE + 'random_downsample_filter.param.yaml': PRE + 'random_downsample_filter.param.yaml',
}


def _stock_dir():
    share = pytest.importorskip('ament_index_python.packages')
    # Autoware 2.1.0 and later, then 1.5.0's autoware_launch.
    for package, sub in (('autoware_localization_config', 'config'),
                         ('autoware_launch', 'config/localization')):
        try:
            return Path(share.get_package_share_directory(package)) / sub / 'ndt_scan_matcher'
        except share.PackageNotFoundError:
            continue
    pytest.skip('no Autoware localization config installed')


def _leaves(node, prefix=''):
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            out.update(_leaves(value, f'{prefix}.{key}' if prefix else str(key)))
        return out
    return {prefix: node}


def _params(path):
    return {k.split('ros__parameters.', 1)[1]: v
            for k, v in _leaves(yaml.safe_load(path.read_text())).items()}


@pytest.mark.parametrize('ours', sorted(FILES))
def test_file_values_are_stock(ours):
    stock = _params(_stock_dir() / FILES[ours])
    mine = _params(PKG / 'config' / ours)
    differ = {k: (stock[k], mine[k]) for k in mine if k in stock and mine[k] != stock[k]}
    assert not differ, f'not Autoware defaults (stock, ours): {differ}'


def test_compiled_defaults_are_stock():
    stock = _params(_stock_dir() / 'ndt_scan_matcher.param.yaml')
    source = CRATE.read_text()
    found = re.findall(
        r'declare_parameter(?:::<[^>]*>)?\(\s*"([^"]+)",?\s*\)\s*\.default\(([^)]*\)?)\)',
        source)
    assert found, 'no declarations parsed from params.rs'
    differ = {}
    for name, raw in found:
        if name not in stock:
            continue
        text = re.sub(r'Arc::from\((.*)\)$', r'\1', raw.replace('.into()', '').strip())
        try:
            value = yaml.safe_load(text)
        except yaml.YAMLError:
            continue
        if value != stock[name]:
            differ[name] = (stock[name], value)
    assert not differ, f'compiled defaults differ (stock, ours): {differ}'


def test_no_line_search():
    # Not a key Autoware's file has, so test_file_values_are_stock cannot see
    # it, but it is Autoware's behaviour: its NDT takes the full Newton step.
    mine = _params(PKG / 'config' / 'cuda_scan_matcher.param.yaml')
    assert mine.get('ndt.use_line_search', False) is False
