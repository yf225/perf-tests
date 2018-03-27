import sys
import timeit
import torch
import argparse
import numpy
import subprocess

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

# Common operations?