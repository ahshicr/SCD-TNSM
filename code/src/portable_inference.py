"""Standard-library inference for the actual trained demand MLP.

This is a reference CPU implementation, not a SHA/hash proxy or an NPU/GPU
benchmark. No package installation is needed on legacy ARM boards.
"""
import math


def validate_model(model):
    if model.get('kind') == 'collection':
        if set(model['services']) != {'mlp', 'cnn', 'forest'}:
            raise ValueError('Expected the three declared trained services')
        for service in model['services'].values():
            validate_model(service)
        return
    if model.get('kind') in ('cnn', 'forest'):
        try:
            from r2_portable_models import validate
        except ModuleNotFoundError:
            from .r2_portable_models import validate
        validate(model)
        return
    widths = [66, 256, 128, 64, 1]
    if len(model['layers']) != 4:
        raise ValueError('Expected four trained linear layers')
    for i, layer in enumerate(model['layers']):
        if len(layer['weight']) != widths[i+1] or len(layer['bias']) != widths[i+1]:
            raise ValueError('Model layer output shape mismatch')
        if any(len(row) != widths[i] for row in layer['weight']):
            raise ValueError('Model layer input shape mismatch')
    if len(model['x_mean']) != 66 or len(model['x_std']) != 66:
        raise ValueError('Normalization shape mismatch')
    if any(x <= 0 or not math.isfinite(x) for x in model['x_std']):
        raise ValueError('Invalid feature scale')


def predict_one(model, features):
    if model.get('kind') in ('cnn', 'forest'):
        try:
            from r2_portable_models import predict
        except ModuleNotFoundError:
            from .r2_portable_models import predict
        return predict(model, features)
    if len(features) != 66:
        raise ValueError('Expected 66 features')
    x = [(float(a)-b)/c for a, b, c in zip(features, model['x_mean'], model['x_std'])]
    for i, layer in enumerate(model['layers']):
        y = [bias+sum(a*b for a, b in zip(row, x))
             for row, bias in zip(layer['weight'], layer['bias'])]
        if i < 3:
            x = [.5*z*(1.+math.erf(z/math.sqrt(2.))) for z in y]
        else:
            x = y
    # Match training target: log1p(sum of next four minutes' pressure).
    return max(0., math.expm1(x[0]*model['y_std']+model['y_mean']))/model['horizon']
