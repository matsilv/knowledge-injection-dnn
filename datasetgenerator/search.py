#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ortools.constraint_solver import pywrapcp as pycp
import sys


class SnailLexDecisionBuilder(pycp.PyDecisionBuilder):
    def __init__(self, X):
        pycp.PyDecisionBuilder.__init__(self)
        self.X = X

    def Next(self, slv):
        # Choose the variable to branch upon
        var = None
        for x in self.X:
            if not x.Bound():
                var = x
                break
        # If all variables are bound, this DB job is done
        if var is None:
            return None
        # Choose the branching value
        val = var.Min()
        # Open a choice point
        return slv.AssignVariableValue(var, val)


class SnailMinSizeDecisionBuilder(pycp.PyDecisionBuilder):
    def __init__(self, X):
        pycp.PyDecisionBuilder.__init__(self)
        self.X = X

    def Next(self, slv):
        # Choose the variable to branch upon
        var = None
        score = None
        for x in self.X:
            if not x.Bound() and (score is None or x.Size() < score):
                var = x
                score = x.Size()
        # If all variables are bound, this DB job is done
        if var is None:
            return None
        # Choose the branching value
        val = var.Min()
        # Open a choice point
        return slv.AssignVariableValue(var, val)


class SubPDecisionBuilder(pycp.PyDecisionBuilder):
    def __init__(self, X, K, bmark, stats, frm=None, failcap=0):
        pycp.PyDecisionBuilder.__init__(self)
        self.X = X
        self.bmark = bmark
        self.K = K
        self.stats = stats
        self.frm = frm
        self.over_cap = 0
        self.all_fails = 0
        self.failcap = failcap

    def Next(self, slv):
        # Read the minimum value in the K variable
        k = self.K.Value()
        # View statistics about the previous attempt
        if k > 0:
            # NOTE: The number of fails should be corrected to take into account those due to the fake optimization
            #  process
            fails = slv.Failures() - self.stats['base_fails'] - 1
            self.all_fails += fails
            if fails >= self.failcap:
              self.over_cap += 1
            time = (slv.WallTime() - self.stats['base_time']) / 1000.0
            # print('Attempt: %d | fails: %d, time: %.3f | all fails: %d | over cap: %d' % (k, fails, time, self.all_fails, self.over_cap))
            
        # Update statistics
        self.stats['base_fails'] = slv.Failures()
        self.stats['base_time'] = slv.WallTime()
        self.stats['all_fails'] = self.all_fails
        self.stats['overcap'] = self.over_cap

        # Force all assignments for the k-th instace
        for i in self.bmark[k]:
            x = self.X[i]
            self.bmark[k][i] = int(self.bmark[k][i])
            x.SetValue(self.bmark[k][i])
        # Print the current instance
        if self.frm is not None:
            #sys.stdout.write(self.frm.format(self.X) + '/')
            sys.stdout.flush()
            # bmk = self.bmark[k]
            # prb = [bmk[i]+1 if i in bmk else 0 for i in range(len(self.X))]
            # sys.stdout.write(','.join(str(v) for v in prb) + '/')
            # sys.stdout.flush()
        return None


class StoreDecisionBuilder(pycp.PyDecisionBuilder):
    def __init__(self, X, frm, stats, verbose):
        pycp.PyDecisionBuilder.__init__(self)
        # self.res = res
        self.X = X
        self.frm = frm
        self.stats = stats
        self.verbose = verbose

    def Next(self, slv):
        # Check whether the solution is new
        #sol = [x.Value() for x in self.X]
        #self.res.append(sol)
        if self.verbose:
            sys.stdout.write(self.frm.format(self.X))
            sys.stdout.write("\n")
        '''else:
            sys.stdout.write('T')'''
        return None



