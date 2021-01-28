import numpy as np
import tensorflow as tf

class ReplayBuffer:
    """
    A buffer for storing trajectories experienced by a agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, capacity=10000):

        self.capacity=capacity
        self.buffer = []

    def reformat(self, indices):
        # Reformat a list of Transition tuples for training.
        # indices: list<int>
        names = ['s', 'a', 'r', 's_next', 'done']
        data = {n:[] for n in names}
        for i in indices:
            transition = self.buffer[i]
            for i in range(len(names)):
                data[names[i]].append(transition[i])
        return data

    def add(self, record):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        self.buffer += record

        while self.capacity and self.size > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):

        assert len(self.buffer) >= batch_size
        idxs = np.random.choice(range(len(self.buffer)), size=batch_size, replace=False)
        # samples = [self.buffer[i] for i in idxs]
        return self.reformat(idxs)

    def pop(self, batch_size):

        l = min(self.size, batch_size)
        batch = self.reformat(range(i))
        self.buffer = self.buffer[l:]
        return batch

    @property
    def size(self):
        return len(self.buffer)

