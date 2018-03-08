import sys
import timeit
import torch
import argparse
import numpy
import subprocess
import json
import time

MAX_TRIAL = 10
COOLDOWN_PERIOD = 10

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--compare', dest='compare_data_file_path', action='store',
                    help='perf test data to compare with')
parser.add_argument('--update', dest='update_data_file_path', action='store',
                    help='perf test data to update')
args = parser.parse_args()

should_compare = False
should_update = False

if args.compare_data_file_path:
    should_compare = True
    with open(args.compare_data_file_path) as compare_data_file:
        compare_data = json.load(compare_data_file)

if args.update_data_file_path:
    should_update = True
    with open(args.update_data_file_path) as update_data_file:
        update_data = json.load(update_data_file)

def measure(test_name, stmt, setup, number, repeat, trial=0):
    if only_test_name and not test_name == only_test_name:
        return
    print('Testing: {} ...'.format(test_name))
    trial += 1

    runtimes = []
    for i in range(repeat):
        runtimes += [timeit.timeit(stmt=stmt, setup=setup, number=number)]
    sample_mean = numpy.mean(runtimes)
    sample_sigma = numpy.std(runtimes)
    print("sample mean: ", sample_mean)
    print("sample sigma: ", sample_sigma)
    
    if should_compare:
        if test_name in compare_data:
            baseline_mean = compare_data[test_name]['mean']
            baseline_sigma = compare_data[test_name]['sigma']
        else:
            baseline_mean = sys.maxsize
            baseline_sigma = 0.01
        z_value = (sample_mean - baseline_mean) / baseline_sigma
        print("z-value: {}".format(z_value))
        if z_value >= 3:
            if trial == MAX_TRIAL:
                raise Exception('''\n
z-value >= 3 in all {} trials, there is perf regression.\n
'''.format(trial))
            else:
                print("z-value >= 3, doing another trial in {} seconds.".format(COOLDOWN_PERIOD))
                time.sleep(COOLDOWN_PERIOD)
                measure(test_name, stmt, setup, number, repeat, trial)
        else:
            print("z-value < 3, no perf regression detected.")

    if should_update:
        if not test_name in update_data:
            update_data[test_name] = {}
        update_data[test_name]['mean'] = sample_mean
        update_data[test_name]['sigma'] = max(sample_sigma, sample_mean * 0.1) # Allow a larger margin
        with open(args.update_data_file_path, 'w') as update_data_file:
            json.dump(update_data, update_data_file, indent=4)

only_test_name = 'torch.numel'

measure(test_name='torch.numel',
        stmt='''
torch.numel(t)
''',
        setup='''
import torch
t = torch.ones(1, 1)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.eye',
        stmt='''
torch.eye(3)
''',
        setup='''
import torch
''',
        number=1000,
        repeat=100)

measure(test_name='torch.from_numpy',
        stmt='''
torch.from_numpy(a)
''',
        setup='''
import torch
import numpy
a = numpy.array([1, 2, 3])
''',
        number=1000,
        repeat=100)

measure(test_name='torch.linspace',
        stmt='''
torch.linspace(start=-10, end=10, steps=5)
''',
        setup='''
import torch
''',
        number=1000,
        repeat=100)

measure(test_name='torch.logspace',
        stmt='''
torch.logspace(start=0.1, end=1.0, steps=5)
''',
        setup='''
import torch
''',
        number=1000,
        repeat=100)

measure(test_name='torch.ones',
        stmt='''
torch.ones(2, 3)
''',
        setup='''
import torch
''',
        number=1000,
        repeat=100)

measure(test_name='torch.ones_like',
        stmt='''
torch.ones_like(input)
''',
        setup='''
import torch
input = torch.FloatTensor(2, 3)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.arange',
        stmt='''
torch.arange(1, 2.5, 0.5)
''',
        setup='''
import torch
''',
        number=1000,
        repeat=100)

measure(test_name='torch.zeros',
        stmt='''
torch.zeros(2, 3)
''',
        setup='''
import torch
''',
        number=1000,
        repeat=100)

measure(test_name='torch.zeros_like',
        stmt='''
torch.zeros_like(input)
''',
        setup='''
import torch
input = torch.FloatTensor(2, 3)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.cat',
        stmt='''
torch.cat((x, x, x), 0)
''',
        setup='''
import torch
x = torch.randn(2, 3)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.chunk',
        stmt='''
torch.chunk(t, 3, 0)
''',
        setup='''
import torch
t = torch.randn(3, 2)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.gather',
        stmt='''
torch.gather(t, 1, torch.LongTensor([[0,0],[1,0]]))
''',
        setup='''
import torch
t = torch.Tensor([[1,2],[3,4]])
''',
        number=1000,
        repeat=100)

measure(test_name='torch.index_select',
        stmt='''
torch.index_select(x, 0, indices)
''',
        setup='''
import torch
x = torch.randn(3, 4)
indices = torch.LongTensor([0, 2])
''',
        number=1000,
        repeat=100)

measure(test_name='torch.masked_select',
        stmt='''
torch.masked_select(x, mask)
''',
        setup='''
import torch
x = torch.randn(3, 4)
mask = x.ge(0.5)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.nonzero',
        stmt='''
torch.nonzero(x)
''',
        setup='''
import torch
x = torch.Tensor([[0.6, 0.0, 0.0, 0.0],
                [0.0, 0.4, 0.0, 0.0],
                [0.0, 0.0, 1.2, 0.0],
                [0.0, 0.0, 0.0,-0.4]])
''',
        number=1000,
        repeat=100)

measure(test_name='torch.split',
        stmt='''
torch.split(t, 2, 0)
''',
        setup='''
import torch
t = torch.randn(4, 2)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.squeeze',
        stmt='''
torch.squeeze(x)
''',
        setup='''
import torch
x = torch.zeros(2,1,2,1,2)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.stack',
        stmt='''
torch.stack([x, y], 0)
''',
        setup='''
import torch
x = torch.randn(2, 2)
y = torch.randn(2, 2)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.t',
        stmt='''
torch.t(x)
''',
        setup='''
import torch
x = torch.randn(2, 3)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.take',
        stmt='''
torch.take(src, indices)
''',
        setup='''
import torch
src = torch.Tensor([[4, 3, 5], [6, 7, 8]])
indices = torch.LongTensor([0, 2, 5])
''',
        number=1000,
        repeat=100)

measure(test_name='torch.transpose',
        stmt='''
torch.transpose(x, 0, 1)
''',
        setup='''
import torch
x = torch.randn(2, 3)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.unbind',
        stmt='''
torch.unbind(x, 0)
''',
        setup='''
import torch
x = torch.randn(2, 3)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.unsqueeze',
        stmt='''
torch.unsqueeze(x, 1)
''',
        setup='''
import torch
x = torch.Tensor([1, 2, 3, 4])
''',
        number=1000,
        repeat=100)

measure(test_name='torch.bernoulli',
        stmt='''
torch.bernoulli(a)
''',
        setup='''
import torch
a = torch.Tensor(3, 3).uniform_(0, 1)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.multinomial',
        stmt='''
torch.multinomial(weights, 4)
''',
        setup='''
import torch
weights = torch.Tensor([0, 10, 3, 0])
''',
        number=1000,
        repeat=100)

measure(test_name='torch.normal',
        stmt='''
torch.normal(mean=means_range, std=std)
''',
        setup='''
import torch
means_range = torch.randn(5).data
std = 1.0
''',
        number=1000,
        repeat=100)

measure(test_name='torch.rand',
        stmt='''
torch.rand(2, 3)
''',
        setup='''
import torch
''',
        number=1000,
        repeat=100)

measure(test_name='torch.randn',
        stmt='''
torch.randn(2, 3)
''',
        setup='''
import torch
''',
        number=1000,
        repeat=100)

measure(test_name='torch.randperm',
        stmt='''
torch.randperm(100)
''',
        setup='''
import torch
''',
        number=1000,
        repeat=100)

measure(test_name='torch.abs',
        stmt='''
torch.abs(x)
''',
        setup='''
import torch
x = torch.FloatTensor([-1, -2, 3])
''',
        number=1000,
        repeat=100)

measure(test_name='torch.acos',
        stmt='''
torch.acos(a)
''',
        setup='''
import torch
a = torch.randn(4)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.add',
        stmt='''
torch.add(a, 10, b)
''',
        setup='''
import torch
a = torch.randn(4)
b = torch.randn(4)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.addcdiv',
        stmt='''
torch.addcdiv(t, 0.1, t1, t2)
''',
        setup='''
import torch
t = torch.randn(6)
t1 = torch.randn(6)
t2 = torch.randn(6)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.addcmul',
        stmt='''
torch.addcmul(t, 0.1, t1, t2)
''',
        setup='''
import torch
t = torch.randn(6)
t1 = torch.randn(6)
t2 = torch.randn(6)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.asin',
        stmt='''
torch.asin(a)
''',
        setup='''
import torch
a = torch.randn(4)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.atan',
        stmt='''
torch.atan(a)
''',
        setup='''
import torch
a = torch.randn(4)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.atan2',
        stmt='''
torch.atan2(a, b)
''',
        setup='''
import torch
a = torch.randn(4)
b = torch.randn(4)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.ceil',
        stmt='''
torch.ceil(a)
''',
        setup='''
import torch
a = torch.randn(4)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.clamp',
        stmt='''
torch.clamp(a, min=-0.5, max=0.5)
''',
        setup='''
import torch
a = torch.randn(4)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.cos',
        stmt='''
torch.cos(a)
''',
        setup='''
import torch
a = torch.randn(4)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.cosh',
        stmt='''
torch.cosh(a)
''',
        setup='''
import torch
a = torch.randn(4)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.div',
        stmt='''
torch.div(a, b)
''',
        setup='''
import torch
a = torch.randn(16)
b = torch.randn(16)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.erf',
        stmt='''
torch.erf(a)
''',
        setup='''
import torch
a = torch.Tensor([0, -1., 10.])
''',
        number=1000,
        repeat=100)

measure(test_name='torch.erfinv',
        stmt='''
torch.erfinv(a)
''',
        setup='''
import torch
a = torch.Tensor([0, 0.5, -1.])
''',
        number=1000,
        repeat=100)

measure(test_name='torch.exp',
        stmt='''
torch.exp(a)
''',
        setup='''
import torch
import math
a = torch.Tensor([0, math.log(2)])
''',
        number=1000,
        repeat=100)

measure(test_name='torch.floor',
        stmt='''
torch.floor(a)
''',
        setup='''
import torch
a = torch.randn(4)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.fmod',
        stmt='''
torch.fmod(a, 1.5)
''',
        setup='''
import torch
a = torch.Tensor([1, 2, 3, 4, 5])
''',
        number=1000,
        repeat=100)

measure(test_name='torch.frac',
        stmt='''
torch.frac(a)
''',
        setup='''
import torch
a = torch.Tensor([1, 2.5, -3.2])
''',
        number=1000,
        repeat=100)

measure(test_name='torch.lerp',
        stmt='''
torch.lerp(start, end, 0.5)
''',
        setup='''
import torch
start = torch.arange(1, 5)
end = torch.Tensor(4).fill_(10)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.log',
        stmt='''
torch.log(a)
''',
        setup='''
import torch
a = torch.randn(5)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.log1p',
        stmt='''
torch.log1p(a)
''',
        setup='''
import torch
a = torch.randn(5)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.mul',
        stmt='''
torch.mul(a, b)
''',
        setup='''
import torch
a = torch.randn(16)
b = torch.randn(16)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.neg',
        stmt='''
torch.neg(a)
''',
        setup='''
import torch
a = torch.randn(5)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.pow',
        stmt='''
torch.pow(a, exp)
''',
        setup='''
import torch
exp = torch.arange(1, 5)
a = torch.arange(1, 5)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.reciprocal',
        stmt='''
torch.reciprocal(a)
''',
        setup='''
import torch
a = torch.randn(4)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.remainder',
        stmt='''
torch.remainder(a, 1.5)
''',
        setup='''
import torch
a = torch.Tensor([1, 2, 3, 4, 5])
''',
        number=1000,
        repeat=100)

measure(test_name='torch.round',
        stmt='''
torch.round(a)
''',
        setup='''
import torch
a = torch.randn(4)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.rsqrt',
        stmt='''
torch.rsqrt(a)
''',
        setup='''
import torch
a = torch.randn(4)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.sigmoid',
        stmt='''
torch.sigmoid(a)
''',
        setup='''
import torch
a = torch.randn(4)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.sign',
        stmt='''
torch.sign(a)
''',
        setup='''
import torch
a = torch.randn(4)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.sin',
        stmt='''
torch.sin(a)
''',
        setup='''
import torch
a = torch.randn(4)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.sinh',
        stmt='''
torch.sinh(a)
''',
        setup='''
import torch
a = torch.randn(4)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.sqrt',
        stmt='''
torch.sqrt(a)
''',
        setup='''
import torch
a = torch.randn(4)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.tan',
        stmt='''
torch.tan(a)
''',
        setup='''
import torch
a = torch.randn(4)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.tanh',
        stmt='''
torch.tanh(a)
''',
        setup='''
import torch
a = torch.randn(4)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.trunc',
        stmt='''
torch.trunc(a)
''',
        setup='''
import torch
a = torch.randn(4)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.cumprod',
        stmt='''
torch.cumprod(a, dim=0)
''',
        setup='''
import torch
a = torch.randn(10)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.cumsum',
        stmt='''
torch.cumsum(a, dim=0)
''',
        setup='''
import torch
a = torch.randn(10)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.dist',
        stmt='''
torch.dist(x, y, 3.5)
''',
        setup='''
import torch
x = torch.randn(4)
y = torch.randn(4)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.mean',
        stmt='''
torch.mean(a, 1)
''',
        setup='''
import torch
a = torch.randn(4, 4)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.median',
        stmt='''
torch.median(a, 1)
''',
        setup='''
import torch
a = torch.randn(4, 4)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.mode',
        stmt='''
torch.mode(a, 1)
''',
        setup='''
import torch
a = torch.randn(4, 4)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.norm',
        stmt='''
torch.norm(a, 2, 1)
''',
        setup='''
import torch
a = torch.randn(4, 2)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.prod',
        stmt='''
torch.prod(a, 1)
''',
        setup='''
import torch
a = torch.randn(4, 2)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.std',
        stmt='''
torch.std(a)
''',
        setup='''
import torch
a = torch.randn(1, 3)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.sum',
        stmt='''
torch.sum(a, 1)
''',
        setup='''
import torch
a = torch.randn(4, 4)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.var',
        stmt='''
torch.var(a, 1)
''',
        setup='''
import torch
a = torch.randn(4, 4)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.eq',
        stmt='''
torch.eq(a, b)
''',
        setup='''
import torch
a = torch.Tensor([[1, 2], [3, 4]])
b = torch.Tensor([[1, 1], [4, 4]])
''',
        number=1000,
        repeat=100)

measure(test_name='torch.equal',
        stmt='''
torch.equal(a, b)
''',
        setup='''
import torch
a = torch.Tensor([[1, 2], [3, 4]])
b = torch.Tensor([[1, 1], [4, 4]])
''',
        number=1000,
        repeat=100)

measure(test_name='torch.ge',
        stmt='''
torch.ge(a, b)
''',
        setup='''
import torch
a = torch.Tensor([[1, 2], [3, 4]])
b = torch.Tensor([[1, 1], [4, 4]])
''',
        number=1000,
        repeat=100)

measure(test_name='torch.gt',
        stmt='''
torch.gt(a, b)
''',
        setup='''
import torch
a = torch.Tensor([[1, 2], [3, 4]])
b = torch.Tensor([[1, 1], [4, 4]])
''',
        number=1000,
        repeat=100)

measure(test_name='torch.kthvalue',
        stmt='''
torch.kthvalue(x, 4)
''',
        setup='''
import torch
x = torch.arange(1, 6)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.le',
        stmt='''
torch.le(a, b)
''',
        setup='''
import torch
a = torch.Tensor([[1, 2], [3, 4]])
b = torch.Tensor([[1, 1], [4, 4]])
''',
        number=1000,
        repeat=100)

measure(test_name='torch.lt',
        stmt='''
torch.lt(a, b)
''',
        setup='''
import torch
a = torch.Tensor([[1, 2], [3, 4]])
b = torch.Tensor([[1, 1], [4, 4]])
''',
        number=1000,
        repeat=100)

measure(test_name='torch.max',
        stmt='''
torch.max(a)
''',
        setup='''
import torch
a = torch.randn(4, 4)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.min',
        stmt='''
torch.min(a)
''',
        setup='''
import torch
a = torch.randn(4, 4)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.ne',
        stmt='''
torch.ne(a, b)
''',
        setup='''
import torch
a = torch.Tensor([[1, 2], [3, 4]])
b = torch.Tensor([[1, 1], [4, 4]])
''',
        number=1000,
        repeat=100)

measure(test_name='torch.sort',
        stmt='''
sorted, indices = torch.sort(x)
''',
        setup='''
import torch
x = torch.randn(3, 4)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.topk',
        stmt='''
torch.topk(x, 3)
''',
        setup='''
import torch
x = torch.arange(1, 6)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.cross',
        stmt='''
torch.cross(a, b)
''',
        setup='''
import torch
a = torch.randn(4, 3)
b = torch.randn(4, 3)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.diag',
        stmt='''
torch.diag(a)
''',
        setup='''
import torch
a = torch.randn(3)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.histc',
        stmt='''
torch.histc(a, bins=4, min=0, max=3)
''',
        setup='''
import torch
a = torch.FloatTensor([1, 2, 1])
''',
        number=1000,
        repeat=100)

measure(test_name='torch.renorm',
        stmt='''
torch.renorm(x, 1, 0, 5)
''',
        setup='''
import torch
x = torch.ones(3, 3)
x[1].fill_(2)
x[2].fill_(3)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.trace',
        stmt='''
torch.trace(x)
''',
        setup='''
import torch
x = torch.arange(1, 10).view(3, 3)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.tril',
        stmt='''
torch.tril(a)
''',
        setup='''
import torch
a = torch.randn(3,3)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.triu',
        stmt='''
torch.triu(a)
''',
        setup='''
import torch
a = torch.randn(3,3)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.addbmm',
        stmt='''
torch.addbmm(M, batch1, batch2)
''',
        setup='''
import torch
M = torch.randn(3, 5)
batch1 = torch.randn(10, 3, 4)
batch2 = torch.randn(10, 4, 5)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.addmm',
        stmt='''
torch.addmm(M, mat1, mat2)
''',
        setup='''
import torch
M = torch.randn(2, 3)
mat1 = torch.randn(2, 3)
mat2 = torch.randn(3, 3)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.addmv',
        stmt='''
torch.addmv(M, mat, vec)
''',
        setup='''
import torch
M = torch.randn(2)
mat = torch.randn(2, 3)
vec = torch.randn(3)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.addr',
        stmt='''
torch.addr(M, vec1, vec2)
''',
        setup='''
import torch
vec1 = torch.arange(1, 4)
vec2 = torch.arange(1, 3)
M = torch.zeros(3, 2)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.baddbmm',
        stmt='''
torch.baddbmm(M, batch1, batch2)
''',
        setup='''
import torch
M = torch.randn(10, 3, 5)
batch1 = torch.randn(10, 3, 4)
batch2 = torch.randn(10, 4, 5)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.bmm',
        stmt='''
res = torch.bmm(batch1, batch2)
''',
        setup='''
import torch
batch1 = torch.randn(10, 3, 4)
batch2 = torch.randn(10, 4, 5)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.btrifact',
        stmt='''
A_LU = A.btrifact()
''',
        setup='''
import torch
A = torch.randn(2, 3, 3)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.btrisolve',
        stmt='''
x = b.btrisolve(*A_LU)
''',
        setup='''
import torch
A = torch.randn(2, 3, 3)
b = torch.randn(2, 3)
A_LU = torch.btrifact(A)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.dot',
        stmt='''
torch.dot(a, b)
''',
        setup='''
import torch
a = torch.Tensor([2, 3])
b = torch.Tensor([2, 1])
''',
        number=1000,
        repeat=100)

measure(test_name='torch.gels',
        stmt='''
torch.gels(B, A)
''',
        setup='''
import torch
A = torch.Tensor([[1, 1, 1],
                [2, 3, 4],
                [3, 5, 2],
                [4, 2, 5],
                [5, 4, 3]])
B = torch.Tensor([[-10, -3],
                [ 12, 14],
                [ 14, 12],
                [ 16, 16],
                [ 18, 16]])
''',
        number=1000,
        repeat=100)

measure(test_name='torch.ger',
        stmt='''
torch.ger(v1, v2)
''',
        setup='''
import torch
v1 = torch.arange(1, 5)
v2 = torch.arange(1, 4)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.gesv',
        stmt='''
torch.gesv(B, A)
''',
        setup='''
import torch
A = torch.Tensor([[6.80, -2.11,  5.66,  5.97,  8.23],
                    [-6.05, -3.30,  5.36, -4.44,  1.08],
                    [-0.45,  2.58, -2.70,  0.27,  9.04],
                    [8.32,  2.71,  4.35,  -7.17,  2.14],
                    [-9.67, -5.14, -7.26,  6.08, -6.87]]).t()
B = torch.Tensor([[4.02,  6.19, -8.22, -7.57, -3.03],
                    [-1.56,  4.00, -8.67,  1.75,  2.86],
                    [9.81, -4.09, -4.57, -8.61,  8.99]]).t()
''',
        number=1000,
        repeat=100)

measure(test_name='torch.inverse',
        stmt='''
torch.inverse(x)
''',
        setup='''
import torch
x = torch.rand(10, 10)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.matmul',
        stmt='''
torch.matmul(mat1, mat2)
''',
        setup='''
import torch
mat1 = torch.randn(2, 3)
mat2 = torch.randn(3, 3)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.mm',
        stmt='''
torch.mm(mat1, mat2)
''',
        setup='''
import torch
mat1 = torch.randn(2, 3)
mat2 = torch.randn(3, 3)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.mv',
        stmt='''
torch.mv(mat, vec)
''',
        setup='''
import torch
mat = torch.randn(2, 3)
vec = torch.randn(3)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.potrf',
        stmt='''
torch.potrf(a)
''',
        setup='''
import torch
a = torch.Tensor([[5.4417, -2.5280, 1.3643],
                [-2.5280, 2.9689, -2.1368],
                [1.3643, -2.1368, 4.6116]])
''',
        number=1000,
        repeat=100)

measure(test_name='torch.potri',
        stmt='''
torch.potri(u)
''',
        setup='''
import torch
a = torch.Tensor([[5.4417, -2.5280, 1.3643],
                [-2.5280, 2.9689, -2.1368],
                [1.3643, -2.1368, 4.6116]])
u = torch.potrf(a)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.potrs',
        stmt='''
torch.potrs(b,u)
''',
        setup='''
import torch
a = torch.Tensor([[5.4417, -2.5280, 1.3643],
                [-2.5280, 2.9689, -2.1368],
                [1.3643, -2.1368, 4.6116]])
u = torch.potrf(a)
b = torch.randn(3,2)
''',
        number=1000,
        repeat=100)

measure(test_name='torch.pstrf',
        stmt='''
torch.pstrf(a)
''',
        setup='''
import torch
a = torch.Tensor([[5.4417, -2.5280, 1.3643],
                [-2.5280, 2.9689, -2.1368],
                [1.3643, -2.1368, 4.6116]])
''',
        number=1000,
        repeat=100)

measure(test_name='torch.qr',
        stmt='''
torch.qr(a)
''',
        setup='''
import torch
a = torch.Tensor([[12, -51, 4], [6, 167, -68], [-4, 24, -41]])
''',
        number=1000,
        repeat=100)

measure(test_name='torch.svd',
        stmt='''
torch.svd(a)
''',
        setup='''
import torch
a = torch.Tensor([[8.79,  6.11, -9.15,  9.57, -3.49,  9.84],
                    [9.93,  6.91, -7.93,  1.64,  4.02,  0.15],
                    [9.83,  5.04,  4.86,  8.83,  9.80, -8.99],
                    [5.45, -0.27,  4.85,  0.74, 10.00, -6.02],
                    [3.16,  7.98,  3.01,  5.80,  4.27, -5.31]]).t()
''',
        number=1000,
        repeat=100)

measure(test_name='torch.symeig',
        stmt='''
torch.symeig(a, eigenvectors=True)
''',
        setup='''
import torch
a = torch.Tensor([[ 1.96,  0.00,  0.00,  0.00,  0.00],
                   [-6.49,  3.80,  0.00,  0.00,  0.00],
                   [-0.47, -6.39,  4.17,  0.00,  0.00],
                   [-7.20,  1.50, -1.51,  5.70,  0.00],
                   [-0.65, -6.34,  2.67,  1.80, -7.10]]).t()
''',
        number=1000,
        repeat=100)

measure(test_name='torch.trtrs',
        stmt='''
torch.trtrs(b, A)
''',
        setup='''
import torch
A = torch.randn(2,2).triu()
b = torch.randn(2,3)
''',
        number=1000,
        repeat=100)