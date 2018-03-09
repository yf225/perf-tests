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
# Tensor indexing: index(m)
# index_add_(dim, index, tensor) → Tensor
# index_copy_(dim, index, tensor) → Tensor
# index_fill_(dim, index, val) → Tensor

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
# new(*args, **kwargs)

# Tensor clone: t.clone()

# t.byte()
# t.char()
# t.double()
# t.float()
# t.half()
# t.int()
# t.long()
# t.short()

# t.contiguous(), with t being contiguous
# t.contiguous(), with t being non-contiguous
# t.is_contiguous()

# t.cauchy_
# geometric_(p, *, generator=None) → Tensor
# exponential_(lambd=1, *, generator=None) → Tensor
# log_normal_(mean=1, std=2, *, generator=None)
# normal_(mean=0, std=1, *, generator=None)
# random_(from=0, to=None, *, generator=None)
# uniform_(from=0, to=1) → Tensor


# copy_(src, async=False, broadcast=True) → Tensor

# t.cuda() for CPU test / t.cpu() for GPU test
# is_cuda

# t.data_ptr()

# t.dim()

# t.element_size() 

# expand(*sizes) → Tensor

# fill_(value) → Tensor

# is_signed()

# map_(tensor, callable)

# masked_scatter_(mask, source)
# masked_fill_(mask, value)

# narrow(dimension, start, length) → Tensor

# numpy() → ndarray

# permute(*dims)

# put_(indices, tensor, accumulate=False) → Tensor

# repeat(*sizes)

# resize_(*sizes)

# scatter_(dim, index, src) → Tensor

# select(dim, index) → Tensor or number

# set_(source=None, storage_offset=0, size=None, stride=None)

# share_memory_()

# size()

# storage()
# storage_offset() 

# stride(dim) → tuple or int

# sub(value, other) → Tensor

# tolist()

# type(new_type=None, async=False)

# type_as(tensor)

# unfold(dim, size, step) → Tensor

# view(*args) → Tensor

# zero_() 

# torch.ByteTensor.all() → bool
# torch.ByteTensor.any() → bool