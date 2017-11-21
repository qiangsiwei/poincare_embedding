# -*- coding: utf-8 -*-

import random, numpy as np 
import tensorflow as tf
from utils import pplot, PoincareBase

# caution! may not yield correct result.

class PoincareTensor(PoincareBase):
	eps = 1e-6
	def __init__(self,num_iter=10,num_negs=10,lr1=0.2,lr2=0.001,dp='data/mammal_subtree.tsv'): # dim=2
		super(PoincareTensor,self).__init__(num_iter,num_negs,lr1,lr2,dp)
	def proj(self,x):
		return tf.clip_by_norm(x,1.-self.eps,axes=1)
	def dists(self,u,v):
		uu, uv, vv = tf.norm(u)**2, tf.matmul(u,tf.transpose(v)), tf.norm(v)**2
		alpha, beta = tf.maximum(1-uu,self.eps), tf.maximum(1-vv,self.eps)
		gamma = tf.maximum(1+2*(uu+vv-2*uv)/alpha/beta,1+self.eps)
		return tf.acosh(gamma)
	def train(self): # LEFT SAMPLING
		ld = len(self.pdata); lp = range(len(self.pdict))
		graph = tf.Graph()
		with graph.as_default():
			step = tf.Variable(0,trainable=False)
			pembs = tf.Variable(tf.random_uniform([len(self.pdict),self.dim],minval=-0.001,maxval=0.001))
			n1 = tf.placeholder(tf.int32,shape=(1,),name='n1')
			n2 = tf.placeholder(tf.int32,shape=(1,),name='n2')
			sp = tf.placeholder(tf.int32,shape=(self.num_negs,),name='sp')
			u,v,negs = map(lambda x:tf.nn.embedding_lookup(pembs,x),[n1,n2,sp])
			loss = -tf.log(tf.exp(-self.dists(u,v))/tf.reduce_sum(tf.exp(-self.dists(u,negs))))
			learning_rate = tf.train.polynomial_decay(self.lr1,step,self.num_iter*ld,self.lr2)
			optimizer = tf.train.GradientDescentOptimizer(learning_rate)
			grad_vars = optimizer.compute_gradients(loss)
			rescaled = [(g*(1.-tf.reshape(tf.norm(v,axis=1),(-1,1))**2)**2/4.,v) for g,v in grad_vars]
			trainstep = optimizer.apply_gradients(rescaled,global_step=step); pembs = self.proj(pembs)
			init = tf.global_variables_initializer()
		with tf.Session(graph=graph) as session:
			init.run()
			for epoch in xrange(self.num_iter):
				print epoch; random.shuffle(self.pdata)
				for w1,w2 in self.pdata:
					i1,i2 = self.pdict[w1], self.pdict[w2]
					_,self.pembs = session.run([trainstep,pembs],feed_dict=\
					{n1:[i1],n2:[i2],sp:[random.choice(lp) for _ in range(self.num_negs)]})
		pplot(self.pdict,self.pembs,'mammal_tensor')

if __name__ == '__main__':
	PoincareTensor(num_iter=100).train()
