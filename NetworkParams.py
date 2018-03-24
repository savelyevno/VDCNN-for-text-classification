import numpy as np


class NetworkParams:
    def __init__(self,
                 names,
                 domains,
                 is_log,
                 is_random,
                 sample_count):
        self.names = names
        self.samples = []

        for i in range(sample_count):
            values = []

            for j in range(len(names)):
                a, b = domains[j]

                if is_random[j]:
                    val = np.random.uniform(a, b)
                else:
                    val = a + i * (b - a) / (sample_count - 1)

                if is_log[j]:
                    val = 10 ** val

                values.append(val)

            self.samples.append(values)

        self.last_sample_ind = -1

    def get_next(self):
        res = {}

        if self.last_sample_ind + 1 == len(self.samples):
            raise Exception('All params have been already retrieved')

        values = self.samples[self.last_sample_ind + 1]
        for i in range(len(self.names)):
            res[self.names[i]] = values[i]

        self.last_sample_ind += 1

        return res
