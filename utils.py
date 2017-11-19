# -*- coding: utf-8 -*-

import operator
import matplotlib.pyplot as plt
from collections import defaultdict
from nltk.corpus import wordnet as wn
plt.style.use('ggplot')

def gen_data(network=defaultdict(set)):
	words, target = wn.words(), wn.synset('mammal.n.01')
	targets = set(open('data/targets.txt').read().split('\n'))
	nouns = {noun for word in words for noun in wn.synsets(word,pos='n') if noun.name() in targets}
	for noun in nouns:
		for path in noun.hypernym_paths():
			if not target in path: continue
			for i in range(path.index(target),len(path)-1):
				if not path[i].name() in targets: continue
				network[noun.name()].add(path[i].name())
	with open('data/mammal_subtree.tsv','w') as out:
		for key,vals in network.iteritems():
			for val in vals: out.write(key+'\t'+val+'\n')

def pplot(pdict,pembs,name='mammal'):
	fig = plt.figure(figsize=(10,10)); ax = plt.gca(); ax.cla()
	ax.set_xlim((-1.1,1.1)); ax.set_ylim((-1.1,1.1))
	ax.add_artist(plt.Circle((0,0),1.,color='black',fill=False))
	for w,i in pdict.iteritems():
		c0,c1 = pembs[i]
		ax.plot(c0,c1,'o',color='y')
		ax.text(c0+.01,c1+.01,w,color='b')
	fig.savefig('data/'+name+'.png',dpi=fig.dpi); # plt.show()

class PoincareBase(object):
	def __init__(self,num_iter=10,num_negs=10,lr1=0.2,lr2=0.01,dp='data/mammal_subtree.tsv'): # dim=2
		self.dim = 2
		self.num_iter = num_iter
		self.num_negs = num_negs
		self.lr1, self.lr2 = lr1, lr2
		self.pdata = map(lambda l:l.split('\t'),filter(None,open(dp).read().split('\n')))
		self.pdict = {w:i for i,w in enumerate(set(reduce(operator.add,self.pdata)))}
	def dists(self,u,v): pass
	def train(self): pass

if __name__ == '__main__':
	gen_data()
