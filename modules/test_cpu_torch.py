import sys
import timeit
import torch
import argparse
import numpy
import subprocess

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

# torch.is_tensor

# subprocess.run(['time python -c "import torch; t = torch.ones(10000, 10000).fill_(100); torch.is_tensor(t)"'], stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')
# subprocess.run(['time python -c "import torch"'], stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')

# avg_runtimes = []
# for k in range(20):
#     runtimes = []
#     for i in range(20):
#         timer = timeit.Timer(stmt='torch.is_tensor(t)', setup='import torch; t = torch.ones(1, 1)')
#         runtime = timer.timeit(number=1000)
#         runtimes += [runtime]
#     avg_runtimes += [numpy.mean(runtimes)]
# print(numpy.mean(avg_runtimes))
# print(numpy.std(avg_runtimes))

from time import process_time, perf_counter

# Setup
t = torch.ones(1, 1)

# start_time = perf_counter()

# for i in range(10000):
#     torch.is_tensor(t)

# elapsed_time = perf_counter() - start_time

# print(elapsed_time)

# print("")

runtimes = []
for k in range(200):
    start_time = process_time()
    for i in range(1000):
        torch.is_tensor(t)
    elapsed_time = process_time() - start_time
    runtimes += [elapsed_time]
print(numpy.mean(runtimes))
print(numpy.std(runtimes))
