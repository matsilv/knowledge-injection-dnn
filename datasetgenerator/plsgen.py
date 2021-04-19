#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
An efficient PLS generator.

The code handles the instances generation via CP search. It is based on a
classical PLS model (AllDifferent constraints with GAC propagation), and the
or-tools solver. We used randomized search to find solutions starting from
an empty square: this process always finds a solution with 0 backtracks. When
this happers:
- The solver checkes whether the solution had been previously generated
- If this is not the case, the solution is printed to stdout
- If the desired number of solutions has been generated, the solver stops
- Otherwise, a restart is triggered

Dependencies:

- The code is written for python 3
- The code requires the or-tools, solver, which can be installed with:
    pip3 install py3-ortools
'''

from ortools.constraint_solver import pywrapcp as pycp
import argparse
import numpy as np

import common


#reload(common)


class StoreDecisionBuilder(pycp.PyDecisionBuilder):
    def __init__(self, X, res, num):
        pycp.PyDecisionBuilder.__init__(self)
        self.res = res
        self.X = X
        self.num = num

    def Next(self, slv):
        # Check whether the solution is new
        sol = str([x.Value() for x in self.X.values()])
        if sol not in self.res:
            self.res.add(sol)
            print(common.format_pls(X, n, args.format))
        # Trigger a fail if the generation process is not over
        if len(self.res) < self.num:
            slv.Fail()
        else:
            return None

########################################################################################################################


if __name__ == '__main__':
    # Build a command line parser
    parser = argparse.ArgumentParser(description='Generate PLS instances')
    # Configure the parser
    parser.add_argument('-o', '--order', type=int, required=True,
            help='Order of the PLS to be generated')
    parser.add_argument('-n', '--number', type=int, default=1,
            help='Number of PLS instances to be generated')
    parser.add_argument('-s', '--seed', type=int, default=100,
            help='Seed for the Random Number Generator')
    parser.add_argument('-f', '--format',
            choices=['friendly', 'csv', 'bin'], default='friendly',
            help='Format for displaying the instances. "friendly" will use '+
            'the most natural format (a matrix of integers); "csv" will ' +
            'flatten the square in a vectore (by rows); "bin" will do the ' +
            'same using a one-hot encoding (by number and then row).')
    parser.add_argument('-b', '--bias',
            choices=['none', 'fwd', 'bwd'], default='none',
            help='Introduces a bias in the instance generation. Choosing ' +
            '"fwd" will assign variables in a fixed order, which _should_ ' +
            'introduce a subtle bias while still having a non-zero ' +
            'generation probability for all PLSs. Chosing "bwd" will use ' +
            'the reversed ordering')

    # Parse command line options
    args = parser.parse_args()

    # Extract the most frequently used options
    n = args.order
    # Build a solver
    slv = pycp.Solver('PLS Instance generator')
    # Build the variables
    X = {(i,j):slv.IntVar(1, n, 'x_{%d,%d}' % (i,j))
            for i in range(n) for j in range(n)}
    # Post the constraitns
    for i in range(n):
        slv.Add(slv.AllDifferent([X[i,j] for j in range(n)]))
        slv.Add(slv.AllDifferent([X[j,i] for j in range(n)]))

    # Configure search
    allvars = [X[i,j] for i in range(n) for j in range(n)]

    if args.bias == 'none':
        db = slv.Phase(allvars, slv.CHOOSE_RANDOM, slv.ASSIGN_RANDOM_VALUE)
    elif args.bias == 'fwd':
        db = slv.Phase(allvars, slv.CHOOSE_FIRST_UNBOUND, 
                slv.ASSIGN_RANDOM_VALUE)
    elif args.bias == 'bwd':
        db = slv.Phase(allvars[::-1], slv.CHOOSE_FIRST_UNBOUND,
                slv.ASSIGN_RANDOM_VALUE)
    # Use a custom decision builder to store a solution and trigger a fail
    res = set()
    storedb = StoreDecisionBuilder(X, res, args.number)
    db = slv.Compose([db, storedb])
    # Restart whenever a solution is found
    monitors = [slv.ConstantRestart(1)]

    # Generate the instances
    slv.ReSeed(args.seed)
    slv.Solve(db, monitors)

