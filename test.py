from scipy.integrate import quad
import numpy as np
from collections import Callable
import matplotlib.pyplot as plt

"""
This module implements solving ODE:
(-au')' + bu' + c = f with weak solutions
It demands function on U=(a,b)
u(a)=0
u'(b)+nu(b)=m
"""


def generic_func(x, a, b, c):
    if x <= a or x >= c: return 0
    if a < x <= b: return (x - a) / (b - a)
    if b < x < c: return (c - x) / (c - b)


def generic_deriative(x, a, b, c):
    if x <= a or x >= c: return 0
    if a < x <= b: return 1 / (b - a)
    if b < x < c: return 1 / (c - b)


class ChainFunction(Callable):
    def __init__(self, function, predecessor, *args):
        self.function = function
        self.predecessor = predecessor
        self.args = args

    def __call__(self, x):
        return self.function(x, *self.args) * self.predecessor(x)

    def __mul__(self, other):
        return ChainFunction(self, other)


class Equation:
    def __init__(self, i_range, numb, cauchy, u_0, *args):
        """

        :param i_range: range of the analized problem
        :param numb: number of aproximating functions
        :param cauchy: cauchy condition on right boundary of range, m,k, where condition is  a(l)u'(l) + m(l)u'(l) = k
        :param u_0: dirichlet condition on right boundary of range
        :param args: equation, f , c, b, a
        :return:
        """
        if len(args) != 4: raise AttributeError('Unimplemented type of equation\n')
        self.deg = len(args)
        self.args = args
        self.solved = False
        self.range = i_range
        self.numb = numb - 1
        self.cauchy = cauchy
        self.u_0 = u_0

    def special_function(self, x):
        return self.u_0 * (self.range[1] - x) / (self.range[1] - self.range[0])

    def generic_function(self, x, ind):
        w = (self.range[1] - self.range[0]) / self.numb
        v = self.range[0] + ind * w
        return generic_func(x, v, v + w, v + 2 * w)

    def generic_deriative(self, x, ind):
        w = (self.range[1] - self.range[0]) / self.numb
        v = self.range[0] + ind * w
        return generic_deriative(x, v, v + w, v + 2 * w)

    def get_combinations(self):
        """

        :return: Matrix of functions representing multiplication of functions e_i,e_j (or their deriatives)
        """
        u_functions = [self.generic_function, self.generic_deriative, self.generic_deriative]
        v_functions = [self.generic_function, self.generic_function, self.generic_deriative]
        return [
            [
                [ChainFunction(u_functions[z], ChainFunction(v_functions[z], lambda x: 1, j), i) * self.args[z+1] for z in
                 range(0, 3)]
                for j in range(0, self.numb)
                ]
            for i in range(0, self.numb)
            ]

    def get_a_combinations(self, function):
        """

        :return: [-u_0' * v' , -u_0' * v, -u_0 * v]
        """
        u_functions = [self.generic_function, self.generic_deriative, self.generic_deriative]
        v_functions = [self.generic_function, self.generic_function, self.generic_deriative]
        # f(x)*v(x)
        r_functions = [[ChainFunction(self.args[0], lambda x: 1) * ChainFunction(self.generic_function, lambda x: 1, i)]
                       for i in range(0, self.numb)]
        # u = d * w, below: a*d'*v',f*v, b*d'*v, c*d*v -it can be simply integrated by sum()
        # function to integrate: fe_i - c*d*v - b*d'*v - a*d'*v'
        for i in range(0, self.numb):
            r_functions[i].extend(
                [ChainFunction(u_functions[z], ChainFunction(v_functions[z], lambda x: -1, i), i) * self.args[z + 1]
                 for z in range(0, 3)])
        return r_functions

    def first_constant(self, i, j):
        return self.cauchy[0] * self.generic_function(self.range[1], i) * self.generic_function(self.range[1], j)

    def second_constant(self, i):
        return self.cauchy[0] * self.special_function(self.range[1]) * self.generic_function(self.range[1], i)

    def third_constant(self, i):
        return self.cauchy[1] * self.generic_function(self.range[1], i)

    @property
    def solution(self):
        if self.solved: return self.output
        self.solved = True
        functions = self.get_combinations()
        b_functions = self.get_a_combinations(self.args[0])
        self.matrix = [
            [sum([quad(functions[i][j][z], *self.range)[0] for z in range(0, 3)]) + self.first_constant(i, j)
             for j in range(0, self.numb)]
            for i in range(0, self.numb)]
        self.fmatrix = [sum([quad(func, *self.range)[0] for func in b_functions[i]]) - self.second_constant(i) +
                        self.third_constant(i) for i in range(0, self.numb)]
        return_array = np.linalg.solve(self.matrix, self.fmatrix)
        self.output = [1.] * (self.numb + 1)
        for i in range(1, self.numb + 1):
            self.output[i] = return_array[i - 1]

        return self.output

    def plot_func(self, function):
        w = (self.range[1] - self.range[0]) / self.numb
        print (w)
        plt.plot([x * w for x in range(0, self.numb + 1)], [function(v * w) for v in range(0, self.numb + 1)])
        plt.ylabel('Numbers')
        plt.show()

    def plot_solution(self):
        w = (self.range[1] - self.range[0]) / self.numb
        functions = [ChainFunction(self.special_function,lambda x: 1)]
        functions.extend([ChainFunction(self.generic_function,lambda x: self.output[i],i) for i in range (1,self.numb+1)])
        plt.plot([x * w for x in range(0, self.numb + 1)], [sum([functions[i](x * w) for i in range(0,self.numb+1)]) for x in range(0, self.numb + 1)])
        plt.show()

if __name__ == '__main__':
    w = Equation((0,4), 20, (0,20), 3, lambda x: 1, lambda x: 0, lambda x: 0, lambda x: 1/100)
    print (w.solution)
    w.plot_solution()