"""
========================================================
Author: Sevan Harootonian
Affiliation: Princeton University
========================================================
"""

from msdm.core.distributions import DictDistribution

class CustomDictDistribution(DictDistribution):
    def factor(self, func):
        new_dist = {}
        for ele, prob in self.items():
            likelihood_prob = func(ele)
            if likelihood_prob == 0:
                continue
            new_dist[ele] = new_dist.get(ele, 0)
            new_dist[ele] += prob*likelihood_prob
        norm = sum(new_dist.values())
        new_ele, new_probs = [], []
        for e, p in new_dist.items():
            new_ele.append(e)
            new_probs.append(p/norm)
        new_probs[-1] = 1 - sum(new_probs[:-1])
        new_dist = dict(zip(new_ele, new_probs))
        return CustomDictDistribution(new_dist)
    
    def condition(self, predicate):
        dist = {e: p for e, p in self.items() if predicate(e)}
        norm = sum(dist.values())
        for e, p in dist.items():
            dist[e] = p/norm
        return CustomDictDistribution(dist)
    
    @classmethod
    def uniform(cls, elements):
        prob = 1/len(elements)
        return CustomDictDistribution({e: prob for e in elements})
