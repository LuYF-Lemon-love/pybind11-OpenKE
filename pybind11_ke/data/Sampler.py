import torch
import numpy as np
from collections import defaultdict as ddict
import random
from pybind11_ke.data.TradSampler import BaseSampler
        
class BernSampler(BaseSampler):
    """Using bernoulli distribution to select whether to replace the head entity or tail entity.
    
    Attributes:
        lef_mean: Record the mean of head entity
        rig_mean: Record the mean of tail entity
    """
    def __init__(self, args):
        super().__init__(args)
        self.lef_mean, self.rig_mean = self.calc_bern()
    def __normal_batch(self, h, r, t, neg_size):
        """Generate replace head/tail list according to Bernoulli distribution.
        
        Args:
            h: The head of triples.
            r: The relation of triples.
            t: The tail of triples.
            neg_size: The number of negative samples corresponding to each triple

        Returns:
             numpy.array: replace head list and replace tail list.
        """
        neg_size_h = 0
        neg_size_t = 0
        prob = self.rig_mean[r] / (self.rig_mean[r] + self.lef_mean[r])
        for i in range(neg_size):
            if random.random() > prob:
                neg_size_h += 1
            else:
                neg_size_t += 1

        res = []

        neg_list_h = []
        neg_cur_size = 0
        while neg_cur_size < neg_size_h:
            neg_tmp_h = self.corrupt_head(t, r, num_max=(neg_size_h - neg_cur_size) * 2)
            neg_list_h.append(neg_tmp_h)
            neg_cur_size += len(neg_tmp_h)
        if neg_list_h != []:
            neg_list_h = np.concatenate(neg_list_h)
        
        for hh in neg_list_h[:neg_size_h]:
            res.append((hh, r, t))
        
        neg_list_t = []
        neg_cur_size = 0
        while neg_cur_size < neg_size_t:
            neg_tmp_t = self.corrupt_tail(h, r, num_max=(neg_size_t - neg_cur_size) * 2)
            neg_list_t.append(neg_tmp_t)
            neg_cur_size += len(neg_tmp_t)
        if neg_list_t != []:
            neg_list_t = np.concatenate(neg_list_t)
        
        for tt in neg_list_t[:neg_size_t]:
            res.append((h, r, tt))

        return res

    def sampling(self, data):
        """Using bernoulli distribution to select whether to replace the head entity or tail entity.
    
        Args:
            data: The triples used to be sampled.

        Returns:
            batch_data: The training data.
        """
        batch_data = {}
        neg_ent_sample = []

        batch_data['mode'] = 'bern'
        for h, r, t in data:
            neg_ent = self.__normal_batch(h, r, t, self.args.num_neg)
            neg_ent_sample += neg_ent
        
        batch_data["positive_sample"] = torch.LongTensor(np.array(data))
        batch_data["negative_sample"] = torch.LongTensor(np.array(neg_ent_sample))

        return batch_data
    
    def calc_bern(self):
        """Calculating the lef_mean and rig_mean.
        
        Returns:
            lef_mean: Record the mean of head entity.
            rig_mean: Record the mean of tail entity.
        """
        h_of_r = ddict(set)
        t_of_r = ddict(set)
        freqRel = ddict(float)
        lef_mean = ddict(float)
        rig_mean = ddict(float)
        for h, r, t in self.train_triples:
            freqRel[r] += 1.0
            h_of_r[r].add(h)
            t_of_r[r].add(t)
        for r in h_of_r:
            lef_mean[r] = freqRel[r] / len(h_of_r[r])
            rig_mean[r] = freqRel[r] / len(t_of_r[r])
        return lef_mean, rig_mean

    @staticmethod
    def sampling_keys():
        return ['positive_sample', 'negative_sample', 'mode']