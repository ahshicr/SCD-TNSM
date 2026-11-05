"""Fixed-seed training and full held-out export parity for two added services."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from src.r2_portable_models import predict, validate

SEED = 20261105

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    x, y = load_digits(return_X_y=True)
    x = (x/16).astype(np.float32)
    train, test = train_test_split(np.arange(len(y)), test_size=.25, random_state=SEED, stratify=y)
    protocol = dict(seed=SEED, dataset='scikit-learn load_digits, 1797 UCI test-subset images',
        dataset_source='https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html',
        original_source='https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits',
        source_license='CC BY 4.0',
        interpretation='A newly declared stratified split of the 1797-image subset, not the original UCI train/test evaluation',
        training_indices=train.tolist(), heldout_indices=test.tolist(),
        data_sha256=hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest(),
        forest=dict(trees=64, max_depth=12, min_samples_leaf=1),
        cnn=dict(conv='1 to 8 channels, 3x3 padding 1', activation='ReLU', pool='2x2 average',
                 classifier='128 to 10 linear', epochs=80, optimizer='Adam', learning_rate=.003, batch_size=128),
        evaluation='Final fixed-epoch model, no held-out tuning or early stopping',
        sklearn_version=sklearn.__version__, torch_version=torch.__version__, started_utc=datetime.now(timezone.utc).isoformat())
    (a.out/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    forest = RandomForestClassifier(n_estimators=64,max_depth=12,random_state=SEED,n_jobs=2).fit(x[train],y[train])
    trees = []
    for estimator in forest.estimators_:
        t = estimator.tree_
        value = t.value[:,0,:]
        value = value/value.sum(axis=1,keepdims=True)
        trees.append(dict(left=t.children_left.tolist(), right=t.children_right.tolist(),
            feature=t.feature.tolist(), threshold=t.threshold.tolist(), value=value.tolist()))
    forest_model = dict(kind='forest', trees=trees, input_features=66, used_features=64, classes=list(range(10)))
    cnn = nn.Sequential(nn.Conv2d(1,8,3,padding=1),nn.ReLU(),nn.AvgPool2d(2),nn.Flatten(),nn.Linear(128,10))
    optimizer = torch.optim.Adam(cnn.parameters(),lr=.003)
    tx = torch.from_numpy(x[train].reshape(-1,1,8,8))
    ty = torch.from_numpy(y[train]).long()
    losses = []
    cnn.train()
    for epoch in range(80):
        order = torch.randperm(len(train))
        total = 0.
        for indices in order.split(128):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(cnn(tx[indices]),ty[indices])
            loss.backward()
            optimizer.step()
            total += float(loss.detach())*len(indices)
        losses.append(total/len(train))
    cnn.eval()
    with torch.no_grad():
        expected_cnn = cnn(torch.from_numpy(x[test].reshape(-1,1,8,8))).argmax(dim=1).numpy()
    cnn_model = dict(kind='cnn',conv_weight=cnn[0].weight.detach().numpy()[:,0].tolist(),
        conv_bias=cnn[0].bias.detach().tolist(),fc_weight=cnn[4].weight.detach().tolist(),fc_bias=cnn[4].bias.detach().tolist())
    features = np.pad(x[test],((0,0),(0,2)))
    results = []
    for name, model, expected in [('forest',forest_model,forest.predict(x[test])),('cnn',cnn_model,expected_cnn)]:
        validate(model)
        actual = np.asarray([predict(model,row) for row in features])
        if not np.array_equal(actual, expected):
            raise AssertionError(name+' export parity failed')
        directory = a.out/name
        directory.mkdir()
        (directory/'model.json').write_text(json.dumps(model,separators=(',',':')),encoding='utf-8')
        np.savez_compressed(directory/'validation_inputs.npz',features=features,expected=expected,labels=y[test],indices=test)
        results.append(dict(service=name,heldout_examples=len(test),correct=int((expected==y[test]).sum()),
            accuracy=float((expected==y[test]).mean()),export_parity_cases=int(len(test)),
            model_sha256=hashlib.sha256((directory/'model.json').read_bytes()).hexdigest()))
    (a.out/'training_losses.json').write_text(json.dumps(losses),encoding='utf-8')
    (a.out/'evaluation.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2))

if __name__ == '__main__':
    main()
