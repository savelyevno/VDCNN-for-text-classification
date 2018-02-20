import numpy as np


class NetworkParams:
    def __init__(self,
                 names,
                 log_domains):
        self.names = names
        self.log_domains = log_domains

        assert len(self.names) == len(self.log_domains)

        self.param_cnt = len(self.names)
        self.values = []

        self.sample()

    def sample(self):
        self.values = []
        for i in range(self.param_cnt):
            a, b = self.log_domains[i]
            self.values.append(10**np.random.uniform(a, b))

    def __str__(self):
        res = '('

        for i in range(self.param_cnt):
            res += self.names[i] + ': ' + str(self.values[i]) + ';'
        res = res[:-1] + ')'

        return res
