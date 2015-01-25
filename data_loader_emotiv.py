print(__doc__)
from mne import Epochs, pick_types
from mne.io import concatenate_raws
from mne.datasets import eegbci
from mne.io.edf import read_raw_edf
from mne.decoding import CSP
from mne.layouts import read_layout
from mne.fiff import pick_types
from mne.io import Raw
from mne.io import read_raw_edf
import mne.io.edf.edf as edfmodule
from mne.event import find_events
from scipy import io
import os, mne, unicodedata
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join


class DataLoaderEmotiv(object):

	def __init__(self):
		print self.__class__.__name__
		self.tmin, self.tmax = -1, 4
		self.event_id = dict(first=1, second=2)
		self.cur_path = os.path.dirname(os.path.realpath(__file__))
		# print self.cur_path
		num = len((self.cur_path).split('/'))
		# print num
		self.valid_path = '/'.join((self.cur_path).split('/')[:num-2]) + '/data/video_emotiv_valid/'
		self.edf_path = '/'.join((self.cur_path).split('/')[:num-2]) + '/data/video_emotiv/'
		self.rt_path = '/'.join((self.cur_path).split('/')[:num-2]) + '/data/emotiv/rt/'
		self.edf_path1 = '/'.join((self.cur_path).split('/')[:num-2]) + '/data/video2_emotiv/'
		# print self.edf_path
	
	# def load_realtime(self):
	# 	self.rt_path = self.path + '../emotiv/rt/'

	def new_data(self):
		import sys
		os.chdir('/Users/ziqipeng/Dropbox/bci/x/data/video_emotiv')
		import subprocess, commands
		print "path\n", os.path.dirname(os.path.realpath(__file__))
		cur_file = self.realtime_dir()
		# cur_file = 'test_f.txt'
		cmd = "tail -f %s" % cur_file
		print "cmd\n", cmd
		# output = commands.getoutput('ls -l')
		# print output
		# os.system(cmd)
		from subprocess import Popen, PIPE
		# cmd_list = cmd.split()
		# proc = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, shell=True)
		# (out, err) = proc.communicate()
		status, output = commands.getstatusoutput(cmd)
		# out = subprocess.check_output(cmd, shell=True)
		print "out: ", output

		# outout2 = p.stdout.read()
		# p.stdin.write(input)
		# output1 = commands.getoutput(cmd)
		print "+++++++++"
		# print output1
		# proc = subprocess.Popen('tail -f %s' % cur_file, shell=True, stdout=subprocess.PIPE, )
		print "+++++++++"
		# output = proc.communicate()[0]
		print "=========="
		# print output

	def build_dic(self):
		self.test_dir()
		self.f_dic = {}
		self.f_list = []
		for f in self.edffiles:
			if "trim_video" in str(f):
				self.f_list.append(str(f))
		# print "+++++++++++++++++++++++++++"
		print self.f_list
		print len(self.f_list)
		c = 0
		for e in self.f_list:
			for i in range(0, 7):
				if ("trim_video%d-08" % i) in e:
					joint_path = self.edf_path + e
					# print "{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{"
					# print joint_path
					# print "}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"
					video_value = self.edf_loader(joint_path)
					self.f_dic.update({e: video_value})
		# for i in self.f_list:
		# 	if "trim_video10" in i:
		# 		continue
		# 	joint_path = self.edf_path + i
		# 	print "{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{"
		# 	print joint_path
		# 	print "}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"
		# 	video_value = self.edf_loader(joint_path)
		# 	self.f_dic.update({i: video_value})
		# 	c += 1
		# 	if c == 3:
		# 		break
		print self.f_dic.keys()
		return self.f_dic

	def build_dic1(self):
		self.test_dir1()
		self.f_dic1 = {}
		self.f_list1 = []
		for f in self.edffiles1:
			self.f_list1.append(str(f))
		print self.f_list1
		for e in self.f_list1:
			joint_path = self.edf_path1 + e
			video_dic = self.edf_loader(joint_path)
			self.f_dic1.update({e: video_dic})
		print len(self.f_dic1.keys())
		print self.f_dic1
		return self.f_dic1

	def rt_read(self):
		pass

	def realtime_loader(self):
		latest_file = self.realtime_dir()

	def realtime_data(self):
		self.rt_dic = {}
		# for i in 
		rt_data = self.edf_loader(self.realtime_dir())
		self.rt_dic.update({self.latest_f: rt_data})
		return self.rt_dic

	def realtime_dir(self):
		self.rtfiles = [f for f in listdir(self.edf_path) if isfile(join(self.edf_path, f))]
		max_time = [0]*6
		print self.rtfiles
		self.rtfiles.remove('.DS_Store')
		for f in self.rtfiles:
			cur_time = (f.split('-')[2]).split('.')[:6]
			print cur_time
			cur_time = map(lambda x: int(x), cur_time)
			if cur_time[2] > max_time[2]:
				max_time = cur_time
			elif (cur_time[2] == max_time[2]) and (cur_time[1] > max_time[1]):
				max_time = cur_time
			elif (cur_time[2] == max_time[2]) and (cur_time[1] == max_time[1]) and (cur_time[0] > max_time[0]):
				max_time = cur_time
			elif (cur_time[2] == max_time[2]) and (cur_time[1] == max_time[1]) and (cur_time[0] == max_time[0]) and (cur_time[3] > max_time[3]):
				max_time = cur_time
			elif (cur_time[2] == max_time[2]) and (cur_time[1] == max_time[1]) and (cur_time[0] == max_time[0]) and (cur_time[3] == max_time[3]) and (cur_time[4] > max_time[4]):
				max_time = cur_time
			elif (cur_time[2] == max_time[2]) and (cur_time[1] == max_time[1]) and (cur_time[0] == max_time[0]) and (cur_time[3] == max_time[3]) and (cur_time[4] == max_time[4]) and (cur_time[5] > max_time[5]):
				max_time = cur_time
		max_time = map(lambda x: str(x), max_time)
		for i in range(len(max_time)):
			if (len(max_time[i]) < 2) and (i != 2):
				max_time[i] = '0'*(2 - len(max_time[i])) + max_time[i]
			elif (len(max_time[i]) < 2) and (i == 2):
				max_time[i] = '0'*(4 - len(max_time[i])) + max_time[i]
		max_t = '.'.join(max_time)
		print max_t
		for f in self.rtfiles:
			if max_t in f:
				self.latest_f = f
				# self.latest_path = self.rt_path + f
				print "ffffffffffffffffff=%s" % f
				self.test_latest_path = self.edf_path + f
				print self.test_latest_path
				# return f
				return self.test_latest_path
		# print self.rtfiles

	def validation_loader(self):
		self.vfs = [f for f in listdir(self.valid_path) if isfile(join(self.valid_path, f))]
		vfs_dic = {}
		for f in self.vfs:
			f_path = join(self.valid_path, f)
			vf_dic = self.edf_loader(f_path)
			vfs_dic.update({f: vf_dic})
		print vfs_dic
		return vfs_dic

	def test_dir1(self):
		self.edffiles1 = [f for f in listdir(self.edf_path1) if isfile(join(self.edf_path1, f))]

	def test_dir(self):
		self.edffiles = [f for f in listdir(self.edf_path) if isfile(join(self.edf_path, f))]
		# print self.edffiles

	# def load_test_data():
	# 	self.edf_path = '/'.join((self.cur_path).split('/')[:num-2]) + '/data/video_emotiv/'
	# 	joint_path = self.edf_path + e
	# 	self.edf_loader()

	def edf_loader(self, single_file):
		raw = read_raw_edf(single_file, preload=True)
		# print "raw: " 
		# print raw
		# print "number of channels: %d, type: %s, raw.ch_names: " % (len(raw.ch_names), type(raw.ch_names))
		# print raw.ch_names
		start, stop = raw.time_as_index([0, 150])
		ch_start = 2
		ch_end = 16
		data, times = raw[ch_start:ch_end, start:stop]
		# print data.shape
		# print type(data)
		# print raw.ch_names[2:16]
		print "TTTTTTTTTTTTTTTTTTTTTTT"
		print "data\n", data
		print np.shape(data)
		print "TTTTTTTTTTTTTTTTTTTTTTT"
		cha_names = []
		for n in raw.ch_names:
			str_n = str(n)
			# str_n = str(unicodedata.normalize('NFKD', n).encode('ascii', 'ignore'))
			cha_names.append(str_n)
		# print type(raw.ch_names[0])
		# print "cha_names: "
		# print cha_names
		data_dict = {i[0]: i[1:] for i in zip(cha_names[2:16], data)}
		# print "data: "
		# print data
		# print "data_dict['AF3']: "
		# print len(data_dict['AF3'][0])
		d_dict = {}
		for k, v in data_dict.iteritems():
			d_dict[k] = v[0]
		# print "d_dict['AF3']:"
		# print len(d_dict['AF3'])
		raw.info['ch_names'] = [chn.strip('.') for chn in raw.info['ch_names']]
		raw.filter(2., 30., method='iir')
		events = find_events(raw, shortest_event=0, stim_channel='STI 014')
		print "events: "
		print events
		# picks = pick_types(raw.info, meg=False, eeg=True, stim=False, exclude='bads')
		"""
		epochs = Epochs(raw, events, self.event_id, self.tmin, self.tmax, proj=True, picks=picks, baseline=None, preload=True, add_eeg_ref=False)
		epoch_train = epochs.crop(tmin=1., tmax=2., copy=True)
		labels = epochs.event[:, -1] - 2
		"""
		# raw.info['ch_names'] = [chn.strip('.') for chn in raw.info['ch_names']]
		
		# epochs = Epochs()
		# print "picks: "
		# print picks
		# print raw[picks]
		# print type(raw)
		# print raw.info
		# print d_dict
		# print d_dict.keys()
		return d_dict
		# print type(raw_py[0]), len(raw_py[0]), raw_py[0]
		# print type(raw_py[1]), len(raw_py[1]), raw_py[1]

if __name__ == "__main__":
	dle = DataLoaderEmotiv()
	dle.build_dic1()
	# dle.new_data()
	# dle.realtime_data()
	# dle.validation_loader()
	# dle.build_dic()
	# print dle.f_dic.keys()
	# dle.edf_loader()
	# dle.test_dir()
	# for f in dle.edffiles:
	# 	if "trim_video0" in f:
	# 		print "========"
	# 		print f
	# all_dic = {}
	




