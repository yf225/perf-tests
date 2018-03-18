from common import PerfTestCase

tc = PerfTestCase()

tc.measure(test_name='torch.tensor[]',
        stmt='''
t[0]
''',
        setup='''
import torch
t = torch.ones(3)
''',
        number=1000,
        repeat=20)

tc.measure(test_name='torch.tensor.index',
        stmt='''
t.index(0)
''',
        setup='''
import torch
t = torch.ones(3)
''',
        number=1000,
        repeat=20)

tc.measure(test_name='torch.tensor.index_add_',
        stmt='''
x.index_add_(0, index, t)
''',
        setup='''
import torch
x = torch.Tensor(5, 3).fill_(1)
t = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
index = torch.LongTensor([0, 4, 2])
''',
        number=1000,
        repeat=20)

tc.measure(test_name='torch.tensor.index_copy_',
        stmt='''
x.index_copy_(0, index, t)
''',
        setup='''
import torch
x = torch.zeros(5, 3)
t = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
index = torch.LongTensor([0, 4, 2])
''',
        number=1000,
        repeat=20)

tc.measure(test_name='torch.tensor.index_fill_',
        stmt='''
x.index_fill_(1, index, -1)
''',
        setup='''
import torch
x = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
index = torch.LongTensor([0, 2])
''',
        number=1000,
        repeat=20)

tc.measure(test_name='torch.tensor[:]',
        stmt='''
x[1:3]
''',
        setup='''
import torch
x = torch.ones(5)
''',
        number=1000,
        repeat=20)

# TODO:

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