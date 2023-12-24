import unittest
from type import *
class TestFunctionMethods(unittest.TestCase):
    def test_auto_back_propagation(self):
        class Square(Function):
            def forward(self, x):
                y = x ** 2
                return y
            def backward(self, gy):
                x = self.inputs[0].data
                gx = 2 * x * gy
                return gx
        class Exp(Function):
            def forward(self, x):
                y = np.exp(x)
                return y
            def backward(self, gy):
                x = self.inputs[0].data
                gx = np.exp(x) * gy
                return gx
        """forward"""
        x = Variable(np.array(0.5))
        A = Square()
        B = Exp()
        C = Square()
        a = A(x)
        b = B(a)
        y = C(b)
        """backward"""
        y.backward()
        # 3.29744...
        self.assertTrue(abs(x.grad-3.29) <= 0.01)
    def test_variable_creator(self):
        class Square(Function):
            def forward(self, x):
                y = x ** 2
                return y
            def backward(self, gy):
                x = self.inputs[0].data
                gx = 2 * x * gy
                return gx
        class Exp(Function):
            def forward(self, x):
                y = np.exp(x)
                return y
            def backward(self, gy):
                x = self.inputs[0].data
                gx = np.exp(x) * gy
                return gx
        """forward"""
        x = Variable(np.array(0.5))
        A = Square()
        B = Exp()
        C = Square()
        a = A(x)
        b = B(a)
        y = C(b)
        """test"""
        self.assertTrue(y.creator == C)
        self.assertTrue(y.creator.inputs[0] == b)
        self.assertTrue(y.creator.inputs[0].creator == B)
        self.assertTrue(y.creator.inputs[0].creator.inputs[0] == a)
        self.assertTrue(y.creator.inputs[0].creator.inputs[0].creator == A)
        self.assertTrue(y.creator.inputs[0].creator.inputs[0].creator.inputs[0] == x)
    def test_grad(self):
        class Add(Function):
            def forward(self, x1, x2):
                y = x1 + x2
                return y
            def backward(self, gy):
                x = self.inputs[0].data
                return gy, gy
        x = Variable(np.array(3.0))
        def add(x0, x1):
            return Add()(x0, x1)
        y = add(x, x)
        y.backward()
        self.assertEqual(x.grad, 2.0)
        x.cleargrad()
        y = add(add(x, x), x)
        y.backward()
        self.assertEqual(x.grad, 3.0)

if __name__ == "__main__":
    unittest.main()