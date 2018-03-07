import sys
import timeit
import torch
import argparse
import numpy
import subprocess

MAX_TRIAL = 3

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
    print('Testing: {} ...'.format(test_name))
    runtimes = []
    for i in range(repeat):
        runtimes += [timeit.timeit(stmt=stmt, setup=setup, number=number)]
    sample_mean = numpy.mean(runtimes)
    sample_sigma = numpy.std(runtimes)
    print("sample mean: ", sample_mean)
    print("sample sigma: ", sample_sigma)

    trial += 1
    if should_compare:
        baseline_mean = compare_data[test_name]['mean']
        baseline_sigma = compare_data[test_name]['sigma']
        z_value = (sample_mean - baseline_mean) / baseline_sigma
        if z_value >= 3:
            if trial == MAX_TRIAL:
                raise Exception('''\n
z-value >= 3 in all {} trials, there is perf regression.\n
'''.format(trial))
            else:
                measure(test_name, stmt, setup, number, repeat, trial)
        else:
            print("z-value < 3, no perf regression detected.")

    if should_update:
        update_data[test_name]['mean'] = sample_mean
        update_data[test_name]['sigma'] = sample_sigma
        with open(args.update_data_file_path, 'w') as update_data_file:
            json.dump(update_data, update_data_file, indent=4)

measure(test_name='torch.numel',
        stmt='torch.numel(t)',
        setup='import torch; t = torch.ones(1, 1)',
        number=1000,
        repeat=200)
