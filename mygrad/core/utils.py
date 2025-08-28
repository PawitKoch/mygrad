import numpy as np

def unbroadcast(grad, target_shape):
        g = grad
        while g.ndim > len(target_shape):
            g = g.sum(axis=0)
        
        for i, (gs, ts) in enumerate(zip(g.shape, target_shape)):
            if ts == 1 and gs != 1:
                g = g.sum(axis=i, keepdims=True)
        return g