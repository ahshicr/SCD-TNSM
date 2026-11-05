from pathlib import Path
import pytest
import reproduce

def test_source_snapshot_import_priority(tmp_path):
    source=tmp_path/'frozen'
    source.mkdir()
    (source/'__init__.py').write_text('SNAPSHOT_MARKER = 731\n')
    package=reproduce.package_sources(tmp_path/'selected',source)
    # The cwd still contains the current src package. Safe-path import must
    # select the frozen package explicitly, including in a child interpreter.
    reproduce.run(['-c','import src; assert src.SNAPSHOT_MARKER == 731'],package)

def test_snapshot_rejects_version_mixing(tmp_path):
    source=tmp_path/'frozen'
    source.mkdir()
    (source/'__init__.py').write_text('VALUE = 1\n')
    destination=tmp_path/'selected'
    reproduce.package_sources(destination,source)
    (source/'__init__.py').write_text('VALUE = 2\n')
    with pytest.raises(ValueError,match='mix source'):
        reproduce.package_sources(destination,source)

def test_snapshot_rejects_removed_module(tmp_path):
    source=tmp_path/'frozen'
    source.mkdir()
    (source/'__init__.py').write_text('VALUE = 1\n')
    destination=tmp_path/'selected'
    reproduce.package_sources(destination,source)
    (destination/'src/stale.py').write_text('STALE = True\n')
    with pytest.raises(ValueError,match='stale modules'):
        reproduce.package_sources(destination,source)

def test_every_physical_condition_has_matched_constraints():
    from src.run_r2_physical import design
    trials=design()
    assert len(trials)==140
    assert {r['block'] for r in trials}==set(range(10))
    for block in range(10):
        rows=[r for r in trials if r['block']==block]
        assert len(rows)==14
        for condition in {(r['scenario'],r['workload']) for r in rows}:
            cell=[r for r in rows if (r['scenario'],r['workload'])==condition]
            common=[{k:v for k,v in r.items() if k not in ['ordinal','method']} for r in cell]
            assert all(r==common[0] for r in common)
            assert {r['method'] for r in cell}>={'scd','dpp_matched','compute_only'}
            assert common[0]['constraint_mode']=='matched'
            assert common[0]['jobs']==2880 and common[0]['drain_s']==15
