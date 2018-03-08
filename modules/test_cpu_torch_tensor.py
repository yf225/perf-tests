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

# Tensor indexing: x[0]

# Tensor slicing: x[1:3]

# Tensor math: +-*/

# create tensor with the following types of arguments:
# class torch.Tensor
# class torch.Tensor(*sizes)
# class torch.Tensor(size)
# class torch.Tensor(sequence)
# class torch.Tensor(ndarray)
# class torch.Tensor(tensor)
# class torch.Tensor(storage)

# Tensor clone: t.clone()

