"""
memory.py
prioritized replay memory structure

"""

import numpy
import random


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = numpy.zeros(capacity, dtype=object)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def find(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return idx, self.tree[idx], dataIdx

    def get(self, dataIdx):
        return self.data[dataIdx % self.capacity]


class Memory:
    e = 0.01
    a = 0.5

    def __init__(self, capacity, n_step, discount_r):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.n_step = n_step
        self.discount_r = discount_r

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def sample(self, n, priority_weight):
        batch = []
        weights = []
        segment = self.tree.total() / n
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            while True:
                s = random.uniform(a, b)
                (idx, p, data_index) = self.tree.find(s)
                if (self.tree.write - idx) % self.capacity > self.n_step:
                    break
            n_step_data = []
            for nn in range(self.n_step):
                n_step_data.append(self.tree.get(data_index + nn))

            state = n_step_data[0][0]

            reward = 0.
            for nn in range(self.n_step):
                reward = self.discount_r * reward + n_step_data[nn][1]

            action = n_step_data[0][2]
            next_state = n_step_data[self.n_step-1][3]
            done = n_step_data[self.n_step-1][4]

            data = (state, reward, action, next_state, done)
            batch.append((idx, data))
            w = ((p / self.tree.total()) * self.capacity) ** - priority_weight
            weights.append(w)
        weights = weights / (max(weights) + 1e-6)

        return batch, weights

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)