import hashlib
import pytest
from reanalyse_original_hardware import public_collection_entry


def test_redacted_runtime_record_requires_original_and_public_size_and_hash(tmp_path):
    path=tmp_path/'runtime_info.json'
    old=b'{"hostname":"private-host","cpu_count":4}'
    new=b'{"hostname":"deployment-host","cpu_count":4}'
    path.write_bytes(new)
    entry=dict(local=path.name,bytes=len(old),sha256=hashlib.sha256(old).hexdigest())
    record=dict(source_bytes=len(old),public_bytes=len(new),original_sha256=entry['sha256'],
                public_sha256=hashlib.sha256(new).hexdigest(),
                transformation='deployment_redaction_or_portable_path_serialization')
    updated=public_collection_entry(path,entry,record)
    assert updated['bytes']==len(new) and updated['sha256']==record['public_sha256']
    assert entry['sha256']==hashlib.sha256(old).hexdigest()
    for key in ['source_bytes','public_bytes']:
        with pytest.raises(AssertionError):
            public_collection_entry(path,entry,{**record,key:record[key]+1})
    for key in ['original_sha256','public_sha256']:
        with pytest.raises(AssertionError):
            public_collection_entry(path,entry,{**record,key:'0'*64})
    path.write_bytes(new.replace(b'4',b'5'))
    with pytest.raises(AssertionError):
        public_collection_entry(path,entry,record)


def test_unchanged_collected_bytes_need_no_redaction_bridge(tmp_path):
    path=tmp_path/'daemon.log'
    path.write_text('runtime ready\n')
    entry=dict(local=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert public_collection_entry(path,entry,None)==entry
