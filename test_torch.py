import sys
import timeit
import torch
import argparse
import numpy

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
runtime = timeit.timeit(stmt='torch.is_tensor(torch.ones(1, 1))', setup='import torch', number=20)
print(runtime)