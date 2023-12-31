import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{} is not supported.".format(type(data)))
        self.data = data
        self.grad = None
        self.creator = None
    def set_creator(self, func):
        self.creator = func
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        funcs: list[Function] = [self.creator]
        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                if x.creator is not None:
                    funcs.append(x.creator)
    """勾配のリセット"""
    def cleargrad(self):
        self.grad = None
    


class Function:
    def __call__(self, *inputs: Variable) -> Variable:
        """numpyのスカラーを配列に変換"""
        def as_array(x):
            if np.isscalar(x):
                return np.array(x)
            return x
        
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]
    def forward(self, *xs):
        raise NotImplementedError
    def backward(self, *gys):
        raise NotImplementedError