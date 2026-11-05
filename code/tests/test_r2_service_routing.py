"""Service identities must select both the right payload and parity reference."""
import io
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from src.run_physical_trials import select_service_batch
from src.calibrate_r2_services import load_services

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import board_inference_worker as worker

def frame(metadata,raw=b''):
    header=json.dumps(metadata,separators=(',',':')).encode()
    return struct.pack('!I',len(header))+header+raw

def test_mixed_real_models_receive_their_own_inputs_and_references():
    model,examples=load_services(ROOT/'results/revision/physical/portable_model_20260905',ROOT/'results/r2/trained_services_20261105')
    incoming=frame(dict(type='init',model=model,temperature_limit_c=65.))
    expected={}
    for identity,service in enumerate(['forest','mlp','cnn','forest','cnn','mlp']):
        job=dict(id=identity,service_id=service,batch=4,feature_start=38+identity)
        features,reference=select_service_batch(dict(services=examples),job)
        raw=struct.pack('<264f',*features.reshape(-1))
        incoming+=frame(dict(type='infer',id=identity,service_id=service,batch=4,payload_bytes=len(raw)),raw)
        expected[identity]=reference
    incoming+=frame(dict(type='stop'))
    output=io.BytesIO()
    with patch.object(worker.sys,'stdin',SimpleNamespace(buffer=io.BytesIO(incoming))),patch.object(worker.sys,'stdout',SimpleNamespace(buffer=output)),patch.object(worker,'telemetry',return_value=dict(cpu_c=40.)),patch.object(worker,'cpu_temperature',return_value=40.):
        worker.main()
    output.seek(0)
    found=[]
    while size:=output.read(4):
        result=json.loads(output.read(struct.unpack('!I',size)[0]))
        if result['type']=='result':
            assert result['status']=='ok'
            np.testing.assert_allclose(result['predictions'],expected[result['id']],rtol=1e-4,atol=1e-4)
            found.append(result['id'])
    assert found==list(range(6))
