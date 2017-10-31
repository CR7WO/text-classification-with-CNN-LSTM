import os
import numpy as np


def preprocess(fname, redo=False):
    new_file = fname[:fname.index('.')] + '.preprocess'
    if not redo and os.path.exists(new_file):
        return open(new_file, 'r')
    available = 'abcdefghijklmnopqrstuvwxyz0123456789 \'\"'
    with open(fname) as f:
        lines = f.readlines()
        lines = [_.strip() for _ in lines]
        for idx, line in enumerate(lines):
            _ = ''
            for c in line.lower():
                if c in available:
                    _ += c
                else:
                    _ += ' '
            _ = _.replace('\'', ' \'')
            while '  ' in _:
                _ = _.replace('  ', ' ')
            lines[idx] = _
    with open(new_file, 'w') as f:
        f.write('\n'.join(lines))
    return preprocess(fname)


if __name__ == '__main__':
    preprocess('/hdd/data/author/train.in', True)
    preprocess('/hdd/data/author/test.in', True)
