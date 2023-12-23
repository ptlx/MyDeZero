import unittest
from type import *
class TestFunctionMethods(unittest.TestCase):
    def test_auto_back_propagation(self):
        class Square(Function):
            def forward(self, x):
                y = x ** 2
                return y
            def backward(self, gy):
                x = self.input.data
                gx = 2 * x * gy
                return gx
        class Exp(Function):
            def forward(self, x):
                y = np.exp(x)
                return y
            def backward(self, gy):
                x = self.input.data
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
                x = self.input.data
                gx = 2 * x * gy
                return gx
        class Exp(Function):
            def forward(self, x):
                y = np.exp(x)
                return y
            def backward(self, gy):
                x = self.input.data
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
        self.assertTrue(y.creator.input == b)
        self.assertTrue(y.creator.input.creator == B)
        self.assertTrue(y.creator.input.creator.input == a)
        self.assertTrue(y.creator.input.creator.input.creator == A)
        self.assertTrue(y.creator.input.creator.input.creator.input == x)
if __name__ == "__main__":
    unittest.main()