'''Original author: Divyanshi Sirvastava'''

import numpy as np


def file_line_generator(filename):
    with open(filename) as f:
        while True:
            for line in f:
                yield line
            f.seek(0)


def one_hot(seqs):
    fd = {'A':[1,0,0,0], 'T':[0,1,0,0], 'G':[0,0,1,0], 'C':[0,0,0,1], 'N':[0,0,0,0]}
    onehot = np.array([fd[base] for seq in seqs for base in seq.upper()])
    onehot_reshape = np.reshape(onehot,(-1,len(seqs[0]),4))
    return onehot_reshape


def process(line_buffer, use_binding_labels):
    split_buffer = [line.split() for line in line_buffer]

    binding_labels = np.array([int(line[-1]) for line in split_buffer])
    binding_labels = np.reshape(binding_labels, (binding_labels.shape[0], 1))
    sequences = one_hot([line[-2] for line in split_buffer])

    chromsize = len(split_buffer[0]) - 5
    if chromsize > 0:
        accs = np.array([int(num) for line in split_buffer for num in line[3:3+chromsize]])
        accs = np.reshape(accs, (accs.shape[0] / chromsize, chromsize))

        if use_binding_labels:
            return [sequences, accs], binding_labels
        return [sequences, accs]

    if use_binding_labels:
        return sequences, binding_labels
    return sequences


def get_generator(filename, batchsize, use_binding_labels):
    file_gen = file_line_generator(filename)
    while True:
        line_buffer = [next(file_gen) for _ in range(batchsize)]
        yield process(line_buffer, use_binding_labels)

