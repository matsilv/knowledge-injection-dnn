#!/usr/bin/env python
# -*- coding: utf-8 -*-

def onehot(val, N, extra=False):
    vals = [0] * N
    if extra: vals.append(0)
    vals[val-1] = 1
    return ','.join('%d' % v for v in vals)


def format_pls(X, n, frm):
    s = ''
    V = {(i,j):x.Value() if x.Bound() else 0 for (i,j), x in X.items()}
    if frm == 'friendly':
        s += '\n'
        for i in range(n):
            s += ' '.join('%3d' % V[i,j] for j in range(n)) + '\n'
    elif frm == 'csv':
        for i in range(n):
            s += ','.join('%d' % V[i,j] for j in range(n))
            if i < n-1: s += ','
    elif frm == 'bin':
        for i in range(n):
            s += ','.join(onehot(V[i,j], n) for j in range(n))
            if i < n-1: s += ','
    return s
