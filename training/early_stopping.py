# -*- coding: utf-8 -*-

class EarlyStopping(object):
    
    def __init__(self, direction='minimize', n_iter_no_change=10):
        self.best = None
        self.direction = direction
        self.is_better = None
        self.iter_no_change = 0
        self.n_iter_no_change = n_iter_no_change
        self.init_direction()
        pass
    
    
    def step(self, metrics):
        
        if self.best is None:
            self.best = metrics
            return False
        
        if self.evaluate(metrics, self.best):
            self.iter_no_change = 0
            self.best = metrics
            # print('Improved')
        else:
            self.iter_no_change += 1
            # print('No improvement : ' % (self.iter_no_change))
            
        if self.iter_no_change >= self.n_iter_no_change:
            return True
            
    def init_direction(self):

        # Pickle doesn't support anonymous function        
        # if self.direction == 'minimize':
        #     self.is_better = lambda a, best: a < best    
        # if self.direction == 'maximize':
        #     self.is_better = lambda a, best: a > best
        
        if self.direction == 'maximize':
            self.evaluate = self.is_greater 
        if self.direction == 'minimize':
            self.evaluate = self.is_smaller 
        
    
    def is_greater(self, metrics, best):
        return metrics > best
    
    def is_smaller(self, metrics, best):
        return metrics < best