"""Actual trained CNN and forest inference using the Python standard library."""
import math

def validate(model):
    if model['kind'] == 'forest':
        if len(model['trees']) != 64:
            raise ValueError('Expected 64 trained trees')
        for t in model['trees']:
            if len({len(t[k]) for k in ['left', 'right', 'feature', 'threshold', 'value']}) != 1:
                raise ValueError('Malformed tree')
    elif model['kind'] == 'cnn':
        if len(model['conv_weight']) != 8 or len(model['fc_weight']) != 10:
            raise ValueError('Malformed trained CNN')
    else:
        raise ValueError('Unknown model')

def predict(model, features):
    if len(features) != 66:
        raise ValueError('Expected 64 image pixels and two zero padding fields')
    if model['kind'] == 'forest':
        probabilities = [0.] * 10
        for tree in model['trees']:
            node = 0
            while tree['left'][node] != -1:
                node = tree['left'][node] if features[tree['feature'][node]] <= tree['threshold'][node] else tree['right'][node]
            for c, v in enumerate(tree['value'][node]):
                probabilities[c] += v
        return float(max(range(10), key=lambda c: probabilities[c]))
    activations = []
    for k in range(8):
        conv = [0.] * 64
        weights = model['conv_weight'][k]
        for y in range(8):
            for x in range(8):
                value = model['conv_bias'][k]
                for dy in range(3):
                    yy = y+dy-1
                    if yy < 0 or yy >= 8:
                        continue
                    for dx in range(3):
                        xx = x+dx-1
                        if 0 <= xx < 8:
                            value += features[yy*8+xx]*weights[dy][dx]
                conv[y*8+x] = max(0., value)
        for y in range(0,8,2):
            for x in range(0,8,2):
                activations.append((conv[y*8+x]+conv[y*8+x+1]+conv[(y+1)*8+x]+conv[(y+1)*8+x+1])*.25)
    logits = [b+sum(w*v for w,v in zip(row, activations))
              for b,row in zip(model['fc_bias'], model['fc_weight'])]
    return float(max(range(10), key=lambda c: logits[c]))
