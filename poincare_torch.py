# -*- coding: utf-8 -*-

import torch, random, numpy as np 
from torch.autograd import Variable
from utils import pplot, PoincareBase

class PoincareTorch(PoincareBase):
	eps = 1e-6
	def __init__(self,num_iter=10,num_negs=10,lr1=0.2,lr2=0.01,dp='data/mammal_subtree.tsv'): # dim=2
		super(PoincareTorch,self).__init__(num_iter,num_negs,lr1,lr2,dp)
		self.pembs = torch.Tensor(len(self.pdict),self.dim)
		torch.nn.init.uniform(self.pembs,a=-0.001,b=0.001)
	def proj(self,x):
		norm = x.norm(p=2,dim=1).unsqueeze(1)
		norm[norm<1] = 1; norm[norm>=1] += self.eps
		return x.div(norm)
	def acosh(self,x):
		return torch.log(x+torch.sqrt(x**2-1))
	def dists(self,u,v):
		uu, uv, vv = u.norm(dim=1)**2, u.mm(v.t()), v.norm(dim=1)**2
		alpha, beta = (1-uu).clamp(min=self.eps), (1-vv).clamp(min=self.eps)
		gamma = (1+2*(uu+vv-2*uv)/alpha/beta).clamp(min=1+self.eps)
		return self.acosh(gamma)
	def train(self): # LEFT SAMPLING
		for epoch in xrange(self.num_iter):
			print epoch; random.shuffle(self.pdata)
			r = 1.*epoch/self.num_iter; lr = (1-r)*self.lr1+r*self.lr2
			for w1,w2 in self.pdata:
				i1,i2 = self.pdict[w1], self.pdict[w2]
				u = Variable(self.pembs[i1].unsqueeze(0),requires_grad=True)
				v = Variable(self.pembs[i2].unsqueeze(0),requires_grad=True)
				sp = torch.from_numpy(np.random.randint(0,len(self.pdict),size=(self.num_negs,)))
				negs = Variable(self.pembs[sp],requires_grad=True)
				loss = -torch.log(torch.exp(-self.dists(u,v))/torch.exp(-self.dists(u,negs)).sum())
				loss.backward()
				self.pembs[sp] -= lr*(((1-negs.norm(dim=1)**2)**2)/4.).data.unsqueeze(1)*negs.grad.data
				self.pembs[i1] -= lr*(((1-u.norm()**2)**2)/4.).data*u.grad.data
				self.pembs[i2] -= lr*(((1-v.norm()**2)**2)/4.).data*v.grad.data
				self.pembs = self.proj(self.pembs)
		pplot(self.pdict,self.pembs,'mammal_torch')

if __name__ == '__main__':
	PoincareTorch(num_iter=100).train()
