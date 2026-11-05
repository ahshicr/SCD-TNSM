import hashlib
import json
import pytest
from verify_reconstructed_traces import verify


def fixture(tmp_path):
    reference=tmp_path/'reference'
    generated=tmp_path/'generated'
    for directory in [reference/'trace',generated/'trace']:
        directory.mkdir(parents=True)
    array=generated/'trace/release_s.npy'
    array.write_bytes(b'fixed-array-bytes')
    metadata=dict(source=dict(path='old/input.npz',sha256='source-sha'),
        sampling=dict(retained_requests=1), arrays=dict(release_s=dict(
            file='release_s.npy',shape=[1],dtype='float64',sha256=hashlib.sha256(array.read_bytes()).hexdigest())))
    (reference/'trace/manifest.json').write_text(json.dumps(metadata))
    metadata['source']['path']='new/input.npz'
    (generated/'trace/manifest.json').write_text(json.dumps(metadata))
    return reference,generated,array


def test_reconstruction_allows_only_source_location_change(tmp_path):
    reference,generated,array=fixture(tmp_path)
    result=verify(generated,reference)
    assert result['complete'] and result['traces']==1 and result['required_traces']==1


def test_reconstruction_rejects_changed_array(tmp_path):
    reference,generated,array=fixture(tmp_path)
    array.write_bytes(b'changed-array-bytes')
    with pytest.raises(ValueError,match='array bytes'):
        verify(generated,reference)


def test_self_consistent_new_trace_does_not_replace_archived_evidence(tmp_path):
    reference,generated,array=fixture(tmp_path)
    array.write_bytes(b'changed-array-bytes')
    path=generated/'trace/manifest.json'
    metadata=json.loads(path.read_text())
    metadata['arrays']['release_s']['sha256']=hashlib.sha256(array.read_bytes()).hexdigest()
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='archived scientific metadata'):
        verify(generated,reference)
