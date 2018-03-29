import sys
import timeit
import argparse
import numpy
import subprocess
import json
import time
from time import process_time

MAX_TRIAL = 5
COOLDOWN_PERIOD = 10 # seconds
Z_VALUE_BOUND = 3

class PerfTestCase():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Process some integers.')
        self.parser.add_argument('--compare', dest='compare_data_file_path', action='store',
                                 help='perf test data to compare with')
        self.parser.add_argument('--update', dest='update_data_file_path', action='store',
                                 help='perf test data to update')

        # yf225: couldn't figure out how to let Bash pass variables without quotes, so have to use this workaround
        args_str = ''
        if len(sys.argv) > 2:
            args_str = ' '.join(str(x) for x in sys.argv[1:])
        elif len(sys.argv) == 2:
            args_str = sys.argv[1]
        args_list = None
        if args_str != '':
            args_list = args_str.split(' ')
        self.args = self.parser.parse_args(args_list)

        self.should_compare = False
        self.should_update = False
        self.only_test_name = None

        if self.args.compare_data_file_path:
            self.should_compare = True
            with open(self.args.compare_data_file_path) as compare_data_file:
                self.compare_data = json.load(compare_data_file)

        if self.args.update_data_file_path:
            self.should_update = True
            with open(self.args.update_data_file_path) as update_data_file:
                self.update_data = json.load(update_data_file)

    def measure(self, test_name, stmt, setup, number, repeat, trial=0):
        if self.only_test_name and not test_name == self.only_test_name:
            return
        print('Testing: {} ...'.format(test_name))
        trial += 1

        runtimes = []

        # Measure using timeit
        # for i in range(repeat):
        #     runtimes += [timeit.timeit(stmt=stmt, setup=setup, number=number)]

        # Measure using time.process_time()
        for i in range(repeat):
            exec(setup)
            start_time = process_time()
            for i in range(number):
                exec(stmt)
            elapsed_time = process_time() - start_time
            runtimes += [elapsed_time]

        sample_mean = numpy.mean(runtimes)
        sample_sigma = numpy.std(runtimes)
        print("sample mean: ", sample_mean)
        print("sample sigma: ", sample_sigma)
        
        if self.should_compare:
            if test_name in self.compare_data:
                baseline_mean = self.compare_data[test_name]['mean']
                baseline_sigma = self.compare_data[test_name]['sigma']
            else:
                baseline_mean = sys.maxsize
                baseline_sigma = 0.01
            z_value = (sample_mean - baseline_mean) / baseline_sigma
            print("z-value: {}".format(z_value))
            if z_value >= Z_VALUE_BOUND:
                if trial == MAX_TRIAL:
                    raise Exception('''\n
z-value >= {} in all {} trials, there is perf regression.\n
'''.format(Z_VALUE_BOUND, trial))
                else:
                    print("z-value >= {}, doing another trial in {} seconds.".format(Z_VALUE_BOUND, COOLDOWN_PERIOD))
                    time.sleep(COOLDOWN_PERIOD)
                    self.measure(test_name, stmt, setup, number, repeat, trial)
            else:
                print("z-value < {}, no perf regression detected.".format(Z_VALUE_BOUND))

        if self.should_update:
            if not test_name in self.update_data:
                self.update_data[test_name] = {}
            self.update_data[test_name]['mean'] = sample_mean
            self.update_data[test_name]['sigma'] = max(sample_sigma, sample_mean * 0.1) # Allow a larger margin
            with open(self.args.update_data_file_path, 'w') as update_data_file:
                json.dump(self.update_data, update_data_file, indent=4)
