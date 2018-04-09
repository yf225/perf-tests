from common import PerfTestCase

tc = PerfTestCase()

tc.measure(test_name='torch.Tensor[]',
        stmt='''
t[0]
''',
        setup='''
import torch
t = torch.ones(3)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor[]_multidim',
        stmt='''
t[0]
''',
        setup='''
import torch
t = torch.randn(10, 10, 10)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor[tensor]',
        stmt='''
x[idx]
''',
        setup='''
import torch
idx = torch.tensor(0, dtype=torch.int64)
x = torch.arange(9).reshape(3, 3)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.index',
        stmt='''
t.index((index,))
''',
        setup='''
import torch
t = torch.ones(3)
index = torch.ones(3).byte()
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.index_add_',
        stmt='''
x.index_add_(0, index, t)
''',
        setup='''
import torch
x = torch.Tensor(5, 3).fill_(1)
t = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
index = torch.LongTensor([0, 4, 2])
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.index_copy_',
        stmt='''
x.index_copy_(0, index, t)
''',
        setup='''
import torch
x = torch.zeros(5, 3)
t = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
index = torch.LongTensor([0, 4, 2])
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.index_fill_',
        stmt='''
x.index_fill_(1, index, -1)
''',
        setup='''
import torch
x = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
index = torch.LongTensor([0, 2])
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor[:]',
        stmt='''
x[1:3]
''',
        setup='''
import torch
x = torch.ones(5)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor+',
        stmt='''
x + y
''',
        setup='''
import torch
x = torch.randn(5)
y = torch.randn(5)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor-',
        stmt='''
x - y
''',
        setup='''
import torch
x = torch.randn(5)
y = torch.randn(5)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor*',
        stmt='''
x * y
''',
        setup='''
import torch
x = torch.randn(5)
y = torch.randn(5)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor/',
        stmt='''
x / y
''',
        setup='''
import torch
x = torch.randn(5)
y = torch.ones(5)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor(*sizes)',
        stmt='''
torch.FloatTensor(5)
''',
        setup='''
import torch
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor(size)',
        stmt='''
torch.FloatTensor(x_size)
''',
        setup='''
import torch
x = torch.randn(5)
x_size = x.size()
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor(sequence)',
        stmt='''
torch.FloatTensor(x)
''',
        setup='''
import torch
x = [[1, 2, 3], [4, 5, 6]]
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor(ndarray)',
        stmt='''
torch.FloatTensor(x)
''',
        setup='''
import torch
import numpy
x = numpy.random.randn(5)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor(tensor)',
        stmt='''
torch.FloatTensor(x)
''',
        setup='''
import torch
x = torch.randn(5)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor(storage)',
        stmt='''
torch.FloatTensor(x_storage)
''',
        setup='''
import torch
x = torch.randn(5)
x_storage = x.storage()
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.clone',
        stmt='''
x.clone()
''',
        setup='''
import torch
x = torch.randn(5)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.byte',
        stmt='''
x.byte()
''',
        setup='''
import torch
x = torch.ones(5)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.char',
        stmt='''
x.char()
''',
        setup='''
import torch
x = torch.ones(5)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.double',
        stmt='''
x.double()
''',
        setup='''
import torch
x = torch.ones(5)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.float',
        stmt='''
x.float()
''',
        setup='''
import torch
x = torch.ones(5)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.half',
        stmt='''
x.half()
''',
        setup='''
import torch
x = torch.ones(5)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.int',
        stmt='''
x.int()
''',
        setup='''
import torch
x = torch.ones(5)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.long',
        stmt='''
x.long()
''',
        setup='''
import torch
x = torch.ones(5)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.short',
        stmt='''
x.short()
''',
        setup='''
import torch
x = torch.ones(5)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.contiguous#case1',
        stmt='''
x.contiguous()
''',
        setup='''
import torch
x = torch.randn(4, 5)
assert x.is_contiguous()
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.contiguous#case2',
        stmt='''
x_t.contiguous()
''',
        setup='''
import torch
x = torch.randn(4, 5)
x_t = x.transpose(0, 1)
assert not x_t.is_contiguous()
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.is_contiguous',
        stmt='''
x_t.is_contiguous()
''',
        setup='''
import torch
x = torch.randn(4, 5)
x_t = x.transpose(0, 1)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.cauchy_',
        stmt='''
x.cauchy_()
''',
        setup='''
import torch
x = torch.randn(5)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.geometric_',
        stmt='''
x.geometric_(0.5)
''',
        setup='''
import torch
x = torch.randn(5)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.exponential_',
        stmt='''
x.exponential_()
''',
        setup='''
import torch
x = torch.randn(5)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.log_normal_',
        stmt='''
x.log_normal_()
''',
        setup='''
import torch
x = torch.randn(5)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.normal_',
        stmt='''
x.normal_()
''',
        setup='''
import torch
x = torch.randn(5)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.random_',
        stmt='''
x.random_()
''',
        setup='''
import torch
x = torch.randn(5)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.uniform_',
        stmt='''
x.uniform_()
''',
        setup='''
import torch
x = torch.randn(5)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.uniform_',
        stmt='''
x.uniform_()
''',
        setup='''
import torch
x = torch.randn(5)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.copy_',
        stmt='''
y.copy_(x)
''',
        setup='''
import torch
x = torch.randn(5)
y = torch.randn(5)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.expand',
        stmt='''
x.expand(4, 5)
''',
        setup='''
import torch
x = torch.randn(1, 5)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.fill_',
        stmt='''
x.fill_(1)
''',
        setup='''
import torch
x = torch.randn(5)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.zero_',
        stmt='''
x.zero_()
''',
        setup='''
import torch
x = torch.randn(5)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.map_',
        stmt='''
y.map_(x, callable)
''',
        setup='''
import torch
def callable(a, b):
    return a + b
x = torch.ones(5)
y = torch.ones(5)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.masked_scatter_',
        stmt='''
y.masked_scatter_(mask, x)
''',
        setup='''
import torch
x = torch.ones(5)
y = torch.zeros(5)
mask = torch.zeros(5).byte()
mask[1] = 1
mask[3] = 1
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.masked_fill_',
        stmt='''
x.masked_fill_(mask, 1)
''',
        setup='''
import torch
x = torch.zeros(5)
mask = torch.zeros(5).byte()
mask[1] = 1
mask[3] = 1
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.narrow',
        stmt='''
x.narrow(0, 0, 2)
''',
        setup='''
import torch
x = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.numpy',
        stmt='''
x.numpy()
''',
        setup='''
import torch
x = torch.randn(5)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.permute',
        stmt='''
y.permute(2, 1, 0)
''',
        setup='''
import torch
y = torch.randn(3, 4, 5)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.put_',
        stmt='''
src.put_(indices, tensor)
''',
        setup='''
import torch
src = torch.Tensor([[4, 3, 5], [6, 7, 8]])
indices = torch.LongTensor([1, 3])
tensor = torch.Tensor([9, 10])
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.repeat',
        stmt='''
x.repeat(4, 2)
''',
        setup='''
import torch
x = torch.Tensor([1, 2, 3])
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.resize_',
        stmt='''
x.resize_(2, 2)
''',
        setup='''
import torch
x = torch.Tensor([[1, 2], [3, 4], [5, 6]])
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.scatter_',
        stmt='''
y.scatter_(0, index, x)
''',
        setup='''
import torch
x = torch.rand(2, 5)
y = torch.zeros(3, 5)
index = torch.LongTensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]])
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.select',
        stmt='''
x.select(0, 1)
''',
        setup='''
import torch
x = torch.randn(4, 5)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.size',
        stmt='''
x.size()
''',
        setup='''
import torch
x = torch.randn(4, 5)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.storage',
        stmt='''
x.storage()
''',
        setup='''
import torch
x = torch.randn(4, 5)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.sub',
        stmt='''
x.sub(value, other)
''',
        setup='''
import torch
value = 2.0
other = torch.randn(5)
x = torch.randn(5)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.tolist',
        stmt='''
x.tolist()
''',
        setup='''
import torch
x = torch.randn(5)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.unfold',
        stmt='''
x.unfold(0, 2, 1)
''',
        setup='''
import torch
x = torch.arange(1, 8)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.view',
        stmt='''
x.view(16)
''',
        setup='''
import torch
x = torch.randn(4, 4)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.Tensor.view',
        stmt='''
x.view(16)
''',
        setup='''
import torch
x = torch.randn(4, 4)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.ByteTensor.all',
        stmt='''
x.all()
''',
        setup='''
import torch
x = torch.ones(100).byte()
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.ByteTensor.any',
        stmt='''
x.any()
''',
        setup='''
import torch
x = torch.zeros(100).byte()
''',
        number=500,
        repeat=20)
