from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import TanhLayer, SigmoidLayer, LinearLayer, FeedForwardNetwork, FullConnection, RecurrentNetwork
from pybrain.datasets import SupervisedDataSet
import csv
import numpy as np


class AnnPybrain(object):
	def __init__(self):
		self.test = ""
		pass

	def data_loader_csv(self):
		csvfile = open('/Users/ziqipeng/Dropbox/bci/x/data/openbci/csv/testing_1419553974297.csv', 'rb')
		csv_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
		count = 0
		for row in csv_reader:
			print type(row)
			print row
			# TO USE...
			# if count <= 250:
			# 	continue
			# else:
			# 	pass
			count += 1
			break

	def net(self):
		net = buildNetwork(2, 3, 1, bias=True, hiddenclass=TanhLayer)
		net.activate([2, 1])
		trainer = BackpropTrainer(net, ds)
		trainer.train()

	def net_feedforward(self, inNum, data):
		n = FeedForwardNetwork()
		inLayer = LinearLayer(inNum)
		hiddenLayer = SigmoidLayer(3)
		hiddenLayer1 = SigmoidLayer(3)
		hiddenLayer2 = SigmoidLayer(3)
		outLayer = LinearLayer(1)
		n.addInputModule(inLayer)
		n.addModule(hiddenLayer)
		n.addModule(hiddenLayer1)
		n.addModule(hiddenLayer2)
		n.addOutputModule(outLayer)
		in_to_hidden = FullConnection(inLayer, hiddenLayer)
		h_to_h1 = FullConnection(hiddenLayer, hiddenLayer1)
		h1_to_h2 = FullConnection(hiddenLayer1, hiddenLayer2)
		hidden_to_out = FullConnection(hiddenLayer, outLayer)
		n.addConnection(in_to_hidden)
		n.addConnection(h_to_h1)
		n.addConnection(h1_to_h2)
		n.addConnection(hidden_to_out)
		n.sortModules()
		trainer = BackpropTrainer(n, data)
		trainer.trainUntilConvergence()
		return trainer
		# print n.activate([1, 2])
		# print n

	def net_recurrent(self):
		n = RecurrentNetwork()
		n.addInputModule(LinearLayer(2, name='in'))
		n.addModule(SigmoidLayer(3, name='hidden'))
		n.addOutputModule(LinearLayer(1, name='out'))
		n.addConnection(FullConnection(n['in'], n['hidden'], name='c1'))
		n.addConnection(FullConnection(n['hidden'], n['out'], name='c2'))
		n.addRecurrentConnection(FullConnection(n['hidden']))
		# n.sortModules()
		print n.activate((2, 2))
		# n.reset()

	def create_dataset_train(self):
		ds = SupervisedDataSet(2, 1)
		x0 = [0, 0]
		print type(x0)
		ds.addSample(np.asarray(x0), (0,))
		ds.addSample((0, 1), np.asarray([1]))
		ds.addSample((1, 0), (1,))
		ds.addSample((1, 1), (0,))
		len(ds)
		print type(ds)
		for inpt, target in ds:
			print inpt, target
		net = buildNetwork(2, 3, 1, bias=True, hiddenclass=TanhLayer)
		trainer = BackpropTrainer(net, ds)
		trainer.trainUntilConvergence()

        # print trainer.activate([1, 2])
        # print n

if __name__ == "__main__":
	nn = AnnPybrain()
	nn.create_dataset_train()
	# nn.data_loader_csv()
	# nn.net_feedforward()