#!/usr/bin/env python

"""
Plot parsed log

Author: Yaroslav Ganin
Date: 2014-11-24
"""

import argparse
import csv
import pylab as pl

def plot_output_list(output_list, fields, plot_range, prefix, csv_path):
    iters = [float(d['NumIters']) for d in output_list]

    if plot_range[0]:
        try:
            start = next(i for i, x in enumerate(iters) if x >= plot_range[0])
        except StopIteration:
            start = 0
    else:
        start = 0

    if plot_range[1]:
        try:
            end = next(i for i, x in enumerate(iters) if x >= plot_range[1])
        except StopIteration:
            end = None
    else:
        end = None

    for f in fields:
        y = [float(d[f]) if d[f] else float('nan') for d in output_list]
        pl.plot(iters[start : end], y[start : end], label='%s %s' % (prefix, f))
    pl.savefig(csv_path + "_losses.png")	
    

def read_csv(input_filename):
    """
    Read a CSV file
    """

    output_list = []
    with open(input_filename, 'r') as f:
        dict_reader = csv.DictReader(f)
        for row in dict_reader:
            output_list += [row]

    return output_list

def parse_args():
    description = ('Plot CSV files produced by parse_log.py ')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('csv_path', nargs='+',
                        help='CSV file path')
    parser.add_argument('-p', '--prefix', default=[], 
                        help='Prefix to append to legend entry', 
                        action='append')
    parser.add_argument('-f', '--fields', default=[],
                        help='Comma-separated list of fields to plot', 
                        action='append')
    parser.add_argument('-r', '--range', default=':',
                        help='Range of iterations to display')
	
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    output_lists = []
    fields = []
    prefixes = []

    for i, csv_path in enumerate(args.csv_path):
        output_lists += [read_csv(csv_path)]
        
        if i < len(args.fields):
            fields += [args.fields[i].split(',')]
        else:
            fields += [[]]

        if i < len(args.prefix):
            prefixes += [args.prefix[i]]
        else:
            prefixes += ['']

    plot_range = [int(x) if x else None for x in args.range.split(':')]

    pl.figure()

    for i, output_list in enumerate(output_lists):
        plot_output_list(output_list, fields[i], plot_range, prefixes[i], csv_path)

    pl.legend()
    pl.show()

if __name__ == '__main__':
    main()
    
