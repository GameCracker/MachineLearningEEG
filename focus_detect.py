print(__doc__)
import sys, mne, datetime, time
import numpy as np
import pylab as pl
from math import *
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

sys.path.insert(0, '/Users/ziqipeng/Dropbox/bci/x/backend/emotiv')
import data_loader_emotiv as dle


def rt_ab_detector():
	# 5 second base realtime 
	for i in range(0, 180):
		time.sleep(4)
		cur_state = alpha_beta_compare()
		if cur_state == 1:
			print "MEDITATION -> TURN LEFT"
		else:
			print "ATTENTION -> KEEP GOING"

def alpha_beta_compare():
	dl = dle.DataLoaderEmotiv()
	cur_data = dl.realtime_data()
	# print cur_data
	rt_data = cur_data.values()
	rt_dic = rt_data[0]
	last5sec = 128*5
	x = range(0, last5sec)
	a_mean_add = b_mean_add = 0.0
	count = 0
	for k, v in rt_dic.iteritems():
		# if x_exists = 'x' in locals() or 'x' in globals()
		n = len(v)
		sp = n - last5sec
		ch_cur = v[sp:]
		# print type(ch_cur)
		# print "len(ch_cur)=%d" % len(ch_cur)
		print "ch_cur\n", ch_cur
		alpha = mne.filter.band_pass_filter(x=ch_cur, Fs=128, Fp1=7.5, Fp2=14)
		# a_pl = zip(x, alpha)
		a_mean = np.mean(alpha, dtype=np.float64)
		a_mean_add += a_mean
		a_max, a_min = np.max(alpha), np.min(alpha)
		# print "a_mean=%.8f, a_max=%.8f, a_min=%.8f" % (a_mean, a_max, a_min)
		# print "len(alpha)=%d" % len(alpha)
		# print alpha
		beta = mne.filter.band_pass_filter(x=ch_cur, Fs=128, Fp1=14, Fp2=32)
		# b_pl = zip(x, beta)
		b_mean = np.mean(beta, dtype=np.float64)
		b_mean_add += b_mean
		b_max, b_min = np.max(beta), np.min(beta)
		# print "b_mean=%.8f, b_max=%.8f, b_min=%.8f" % (b_mean, b_max, b_min)
		testFig = pl.figure(1)
		# blue
		pl.plot(alpha)
		# green
		pl.plot(beta)
		pl.show()
		# pl.show()
		count += 1
		# break

	if a_mean_add > b_mean_add:
		# meditation
		print "meditation"
		return 1
	else:
		# attention
		print "attention"
		return 0

# alpha_beta_compare()
rt_ab_detector()

# print type(rt_data)
# print rt_dic