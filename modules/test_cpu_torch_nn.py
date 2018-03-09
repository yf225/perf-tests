import sys
import timeit
import torch
import argparse
import numpy
import subprocess

# You should import the common methods from another file!

# measure(test_name='torch.numel',
#         stmt='''
# torch.numel(t)
# ''',
#         setup='''
# import torch
# t = torch.ones(1, 1)
# ''',
#         number=1000,
#         repeat=200)

# TODO:


'''
Forward:

>>> m = nn.Conv1d(16, 33, 3, stride=2)
>>> input = autograd.Variable(torch.randn(20, 16, 50))
>>> output = m(input)
'''

'''
Backward:

>>> inputs = torch.autograd.Variable(torch.randn(2, 3, 5, 5), requires_grad=True)
>>> conv1 = torch.nn.Conv2d(3, 3, 3)
>>> out1 = conv1(inputs)
>>> y = torch.randn(out1.size())
>>> out1.backward(y)
'''

# TODO: all existing nn layers