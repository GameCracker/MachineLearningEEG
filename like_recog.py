import sys
import os
import mne
import re
import math
import scipy
import pylab
import inspect
import subprocess
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# from scipy.fftpack import fft, ifft
import scipy.fftpack
from numpy.fft import fft, fftshift
from scipy import signal, pi
from scipy.signal import butter, lfilter
from spectrum import *
from sklearn import svm
from mpl_toolkits.mplot3d import proj3d
from sklearn import datasets
from sklearn.externals import joblib
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tests.helpers import gradientCheck
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from sklearn import hmm


sys.path.insert(0, '/Users/ziqipeng/Dropbox/bci/x/backend/emotiv')
# sys.path.append('../backend/emotiv')
import data_loader_emotiv as dle


class LikeRecog(object):
    def __init__(self):
        self.el = dle.DataLoaderEmotiv()
        # self.data_dic = self.el.build_dic()
        self.data_dic = self.el.build_dic1()
        print self.data_dic.keys()
        # self.rt_dic = self.el.realtime_data()
        # self.dic = self.el.edf_loader()
        self.data = []
        self.label = []
        t = 0
        # for ck, cv in self.data_dic.iteritems():
        # if t == 0:
        # 		self.length = len(cv)
        # 	self.data.append(cv)
        # 	self.label.append(ck)
        # 	t += 1
        # self.terms = math.floor(self.length/(128/4))
        # self.train_terms = math.floor(self.length/((60 * 2 * 128) + (12 * 128)))
        # like for 0, dislike for 2, neutral for 1
        self.tag_dic = {'video0': 2,
                        'video1': 0,
                        'video2': 0,
                        'video3': 0,
                        'video4': 1,
                        'video5': 0,
                        'video6': 1,
                        'video7': 0,
                        'video8': 1,
                        'video9': 1,
                        'video10': 2,
                        'video11': 2,
                        'video13': 0,
                        'video14': 2,
                        'video15': 0,
                        'video16': 2,
                        'video17': 0,
                        'video18': 2,
                        'video19': 2,
                        'video20': 1,
                        'video21': 2,
                        'video22': 0,
                        'video23': 0,
                        'video24': 2,
                        'video25': 1}
        self.feature_list = ['F3_theta', 'T8_gamma', 'F7_gamma', 'FC5_alpha', 'FC5_delta', 'F3_delta', 'T8_theta',
                             'O2_theta', 'AF3_alpha', 'O1_theta', 'AF3_beta', 'AF3_gamma', 'T8_beta', 'F8_theta',
                             'F7_beta', 'FC6_gamma', 'F3_beta', 'F7_theta', 'F7_delta', 'O2_beta', 'AF4_delta',
                             'T8_alpha', 'F4_delta', 'P8_delta', 'O1_alpha', 'P8_gamma', 'FC6_delta', 'O2_delta',
                             'F8_beta', 'P8_beta', 'T7_delta', 'P7_alpha', 'T7_theta', 'P7_gamma', 'AF4_beta',
                             'P8_theta', 'F7_alpha', 'O1_beta', 'F3_gamma', 'FC5_theta', 'F4_theta', 'AF4_theta',
                             'P7_delta', 'FC6_beta', 'T7_gamma', 'F4_beta', 'AF4_alpha', 'F4_gamma', 'O1_gamma',
                             'AF3_delta', 'FC6_alpha', 'F8_gamma', 'O2_alpha', 'FC6_theta', 'T8_delta', 'F8_delta',
                             'P7_theta', 'F4_alpha', 'O2_gamma', 'F8_alpha', 'F3_alpha', 'P8_alpha', 'P7_beta',
                             'AF3_theta', 'O1_delta', 'FC5_beta', 'AF4_gamma', 'T7_beta', 'T7_alpha', 'FC5_gamma']
        self.feature_idx = range(70)
        self.feature_dict = dict(zip(self.feature_list, self.feature_idx))
        self.command = []

    # self.feature_idx =

    def web_command(self):
        pass

    # command to a webserver
    # transfer 0/1/2 command to certain callback function

    def average_secwid_train(self):
        print "Load video based EEG data and feature selection..."
        self.all_tags = []
        self.tags_flat = []
        self.all_features = []
        self.all_features_flat = []
        self.pre_average_fs = []
        self.video_labels = []
        self.all_features_average = []
        for k, v in self.data_dic.iteritems():
            video_pre_average = []
            video_average = []
            one_video_data = []
            one_video_ch = []
            for tag in self.tag_dic:
                if tag in k:
                    tag_label = self.tag_dic[tag]
                    # labe - like for 1, other for 0
                    if tag_label == 0:
                        label = 1
                    else:
                        label = 0
            video_label = label
            self.video_labels.append(video_label)
            for ch, data in v.iteritems():
                one_video_ch.append(ch)
                one_video_data.append(data)
                video_terms = int(math.floor(len(data) / 64))
            start = 0
            for i in range(video_terms):
                wid_sec = []
                for one_ch_data in one_video_data:
                    wid_sec_ch = one_ch_data[start:(start + 128)]
                    wid_sec.append(wid_sec_ch)
                start += 64
                if np.shape(wid_sec)[1] == 128:
                    sec_psd_dic = self.feature_psd(wid_sec, one_video_ch)
                    sec_psd_list = []
                    new_sec_dic = {}
                    track_key = []
                    # THOSE KEYS IN SEC_PSD_DIC ARE INTEGER, the order is right
                    for k, v in sec_psd_dic.iteritems():
                        track_key.append(k)
                        new_k = self.feature_dict[k]
                        new_sec_dic[new_k] = v
                    key_list = new_sec_dic.keys()
                    key_list.sort()
                    for key in key_list:
                        sec_psd_list.append((new_sec_dic[key]).tolist())
                    sec_flat = (np.array(sec_psd_list)).flatten()
                    video_pre_average.append(sec_flat)
                    self.all_features.extend(sec_psd_list)
                    self.all_features_flat.append(sec_flat)
                    self.test_data = self.all_features[0]
                    self.test_data1 = self.all_features[1]
                    tags = [label] * 70
                    self.all_tags.append(tags)
                    self.tags_flat.append(label)
            print "==================================="
            pre_average_ary = np.array(video_pre_average)
            print "pre_average_ary\n", pre_average_ary
            print "np.shape(pre_average_ary)\n", np.shape(pre_average_ary)
            pre_average_ary1 = pre_average_ary.T
            for t in pre_average_ary1:
                video_average.append(np.mean(t))
            self.all_features_average.append(video_average)
            print "np.shape(video_average)\n", np.shape(video_average)
            print "***********************************"

    def split_secwid_train(self):
        print "Load video based EEG data and feature selection..."
        self.all_tags = []
        self.tags_flat = []
        self.all_features = []
        self.all_features_flat = []
        for k, v in self.data_dic.iteritems():
            one_video_data = []
            one_video_ch = []
            for tag in self.tag_dic:
                if tag in k:
                    tag_label = self.tag_dic[tag]
                    # labe - like for 1, other for 0
                    if tag_label == 0:
                        label = 1
                    else:
                        label = 0
            for ch, data in v.iteritems():
                one_video_ch.append(ch)
                one_video_data.append(data)
                video_terms = int(math.floor(len(data) / 64))
            # print type(data)
            # print len(data)
            start = 0
            # tags = [label]*128
            # print "video_term: %d" % video_terms
            if_draw = 0
            for i in range(video_terms):
                wid_sec = []
                for one_ch_data in one_video_data:
                    wid_sec_ch = one_ch_data[start:(start + 128)]
                    wid_sec.append(wid_sec_ch)
                # print "i = %d" % i
                start += 64
                # print np.shape(wid_sec)
                # print len(tags)
                if np.shape(wid_sec)[1] == 128:
                    # print "hiiiiiiiiiiii"
                    # norm_sec_dict = self.normalize_train(wid_sec, one_video_ch, video_terms)
                    # print "{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{"
                    # print np.shape(wid_sec)
                    # # print wid_sec
                    # print "{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{"
                    # print one_video_ch
                    # print "]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]"
                    if if_draw == 1:
                        sec_psd_dic = self.feature_psd_plot(wid_sec, one_video_ch, 1)
                        if_draw = 0
                    else:
                        sec_psd_dic = self.feature_psd(wid_sec, one_video_ch)
                    # print sec_psd_dic.keys()
                    sec_psd_list = []
                    # print "&&&&&&&&&&&&&&&&&&&&&&&"
                    # print sec_psd_dic.keys()
                    # print len(sec_psd_dic.keys())
                    # print "&&&&&&&&&&&&&&&&&&&&&&&"
                    new_sec_dic = {}
                    track_key = []
                    # THOSE KEYS IN SEC_PSD_DIC ARE INTEGER, the order is right
                    for k, v in sec_psd_dic.iteritems():
                        track_key.append(k)
                        new_k = self.feature_dict[k]
                        new_sec_dic[new_k] = v
                    key_list = new_sec_dic.keys()
                    key_list.sort()
                    # if count == 0:
                    for key in key_list:
                        # all_features.append(new_sec_dic[key])
                        # print "!!!!!!!!!!!!!!!!!!!!!"
                        # print type(new_sec_dic[key])
                        # print np.shape(new_sec_dic[key])
                        sec_psd_list.append((new_sec_dic[key]).tolist())
                    sec_flat = (np.array(sec_psd_list)).flatten()
                    print "***********************************"
                    # print key_list
                    # print track_key
                    print np.shape(sec_flat)
                    # print sec_psd_list
                    print "***********************************"
                    self.all_features.extend(sec_psd_list)
                    self.all_features_flat.append(sec_flat)
                    # print "++++++++++++++++++++++++++++++++++"
                    # print np.shape(self.all_features)
                    # print self.all_features
                    self.test_data = self.all_features[0]
                    # self.tl = 1
                    self.test_data1 = self.all_features[1]
                    tags = [label] * 70
                    self.all_tags.append(tags)
                    self.tags_flat.append(label)
                    # print np.shape(self.test_data)
                    print "==================================="
                # dic_ir = iter(sorted(sec_psd_dic.iteritems()))
                # print "++++++++++++++++++++++++++++++++++"
                # print len(dic_ir.next())
                # print dic_ir.next()[0]
                # print "++++++++++++++++++++++++++++++++++"
                # self.trim_data = self.data[(128*10):(128*10 + 128*60)]

    def split_secwid_train_pca(self):
        print "Load video based EEG data and feature selection..."
        self.all_tags = []
        self.tags_flat = []
        self.all_features = []
        self.all_features_flat = []
        for k, v in self.data_dic.iteritems():
            one_video_data = []
            one_video_ch = []
            for tag in self.tag_dic:
                if tag in k:
                    tag_label = self.tag_dic[tag]
                    # labe - like for 1, other for 0
                    if tag_label == 0:
                        label = 1
                    else:
                        label = 0
            for ch, data in v.iteritems():
                one_video_ch.append(ch)
                one_video_data.append(data)
                video_terms = int(math.floor(len(data) / 64))
            start = 0
            for i in range(video_terms):
                wid_sec = []
                for one_ch_data in one_video_data:
                    wid_sec_ch = one_ch_data[start:(start + 128)]
                    wid_sec.append(wid_sec_ch)
                start += 64
                if np.shape(wid_sec)[1] == 128:
                    sec_psd_dic = self.feature_psd(wid_sec, one_video_ch)
                    sec_psd_list = []
                    new_sec_dic = {}
                    track_key = []
                    # THOSE KEYS IN SEC_PSD_DIC ARE INTEGER, the order is right
                    for k, v in sec_psd_dic.iteritems():
                        track_key.append(k)
                        new_k = self.feature_dict[k]
                        new_sec_dic[new_k] = v
                    key_list = new_sec_dic.keys()
                    key_list.sort()
                    for key in key_list:
                        sec_psd_list.append((new_sec_dic[key]).tolist())
                    sec_flat = (np.array(sec_psd_list)).flatten()
                    print "***********************************"
                    print np.shape(sec_flat)
                    print "***********************************"
                    self.all_features.extend(sec_psd_list)
                    self.all_features_flat.append(sec_flat)
                    self.test_data = self.all_features[0]
                    self.test_data1 = self.all_features[1]
                    tags = [label] * 70
                    self.all_tags.append(tags)
                    self.tags_flat.append(label)
                    print "==================================="

    def run_train(self, if_avg, cross_valid):
        if cross_valid == 1:
            self.split_secwid_train()
            print "Training classifier and validation..."
            self.cross_validation('', 1)
        else:
            if (if_avg == 0):
                self.split_secwid_train()
                print "Training classifier..."
                # self.svm()
                self.ann()
            else:
                self.average_secwid_train()
                print "Training classifier(avg)..."
                self.svm_average()

    def realtime_window(self):
        sys.path.insert(0, '/Users/ziqipeng/Dropbox/bci/x/backend/emotiv')
        self.rt_dic = self.el.realtime_data()
        for k, v in self.rt_dic.iteritems():
            pass

    def train_set(self):
        self.tterms = self.length / (128 * (60 + 12))

    def svm_average(self):
        print "++++++++++++++++++++++++++++++++++"
        print np.shape(self.video_labels)
        print np.shape(self.all_features_average)
        print "++++++++++++++++++++++++++++++++++"
        print np.shape(self.all_features_average)
        xs = np.asarray(self.all_features_average)
        ys = np.asarray(self.video_labels)
        clf_rbf = svm.NuSVC()
        clf_linear = svm.NuSVC(kernel='linear')
        clf_poly = svm.NuSVC(kernel='poly')
        clf_sigmoid = svm.NuSVC(kernel='sigmoid')
        # clf_precomputed = svm.NuSVC(kernel='precomputed')
        clf_rbf.fit(xs, ys)
        clf_linear.fit(xs, ys)
        clf_poly.fit(xs, ys)
        clf_sigmoid.fit(xs, ys)
        # clf_precomputed.fit(xs, ys)
        tscore_rbf = clf_rbf.fit(xs, ys).score(xs, ys)
        tscore_linear = clf_linear.fit(xs, ys).score(xs, ys)
        tscore_poly = clf_poly.fit(xs, ys).score(xs, ys)
        tscore_sigmoid = clf_sigmoid.fit(xs, ys).score(xs, ys)
        # tscore_precomputed = clf_percomputed.fit(xs, ys).score(xs, ys)
        print 'rbf=%.8f\nlinear=%.8f\npoly=%.8f\nsigmoid=%.8f' % (
            tscore_rbf, tscore_linear, tscore_poly, tscore_sigmoid)
        joblib.dump(clf_rbf, 'models/avg_svm_rbf.pkl')
        joblib.dump(clf_linear, 'models/avg_svm_linear.pkl')
        joblib.dump(clf_poly, 'models/avg_svm_poly.pkl')
        joblib.dump(clf_sigmoid, 'models/avg_svm_sigmoid.pkl')
        # joblib.dump(clf_precomputed, 'models/svm_precomputed.pkl')
        print "Done with classifier\nPredict..."

    # self.rt_test2()

    def train(self):
        self.svm()
        self.ann()
        pass

    # opt = 0, load saved data, opt = 1, do training on new data
    def acquire_module_data(self, opt, name='testing'):
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d,%H:%M:%S')
        if opt == 1:
            np.savetxt('/Users/ziqipeng/Dropbox/bci/x/binary_recognizer/tmp_data/x_matrix_%s_%s.txt' % (name, st),
                       self.all_features_flat)
            np.savetxt('/Users/ziqipeng/Dropbox/bci/x/binary_recognizer/tmp_data/y_matrix_%s_%s.txt' % (name, st),
                       self.tags_flat)
        saved_x = np.loadtxt('/Users/ziqipeng/Dropbox/bci/x/binary_recognizer/tmp_data/x_matrix.txt')
        saved_y = np.loadtxt('/Users/ziqipeng/Dropbox/bci/x/binary_recognizer/tmp_data/y_matrix.txt')
        print "saved_x.shape :", saved_x.shape
        print "saved_y.shape :", saved_y.shape
        # testing, load old data
        if opt == 0:
            xs = np.asarray(saved_x)
            ys = np.asarray(saved_y)
        elif opt == 1:
            # new training
            xs = np.asarray(self.all_features_flat)
            ys = np.asarray(self.tags_flat)
        else:
            pass
        xy_dic = {'xs': xs, 'ys': ys}
        return xy_dic

    def ann(self, opt):
        ann_path = '/Users/ziqipeng/Dropbox/bci/x/binary_recognizer/machine_learning/ann_pybrain'
        chdir = Chdir(ann_path)
        sys.path.insert(0, ann_path)
        # print "cur path:", os.getcwd()
        import ann_pybrain as annpb
        from pybrain.datasets import SupervisedDataSet

        ann = annpb.AnnPybrain()
        xy_dic = self.acquire_module_data(opt)
        xs = xy_dic['xs']
        ys = xy_dic['ys']
        # if opt == 1:
        # np.savetxt('/Users/ziqipeng/Dropbox/bci/x/binary_recognizer/tmp_data/x_matrix.txt', self.all_features_flat)
        # 	np.savetxt('/Users/ziqipeng/Dropbox/bci/x/binary_recognizer/tmp_data/y_matrix.txt', self.tags_flat)
        # saved_x = np.loadtxt('/Users/ziqipeng/Dropbox/bci/x/binary_recognizer/tmp_data/x_matrix.txt')
        # saved_y = np.loadtxt('/Users/ziqipeng/Dropbox/bci/x/binary_recognizer/tmp_data/y_matrix.txt')
        # print "saved_x.shape :", saved_x.shape
        # print "saved_y.shape :", saved_y.shape
        # # testing, load old data
        # if opt == 0:
        # 	xs = np.asarray(saved_x)
        # 	ys = np.asarray(saved_y)
        # elif opt == 1:
        # # new training
        # 	xs = np.asarray(self.all_features_flat)
        # 	ys = np.asarray(self.tags_flat)
        # else:
        # 	pass
        print type(xs)
        print np.shape(xs)
        print type(ys)
        print np.shape(ys)
        count = 0
        ds = SupervisedDataSet(4550, 1)
        for x, y in zip(xs, ys):
            if count == 0:
                print type(x)
                print type(y)
            # print y
            ds.addSample(x, [y])
            count += 1
        print "count %d" % count
        trainer = ann.net_feedforward(4550, ds)

    def cross_validation(self, model_f, if_train):
        data = self.acquire_module_data(0)
        xs = data['xs']
        ys = data['ys']
        clf = svm.NuSVC()
        clf_rbf = clf
        clf_rbf.fit(xs, ys)
        # xs = np.asarray(self.all_features_flat)
        # ys = np.asarray(self.tags_flat)
        print "==================="
        num = np.shape(xs)[0]
        train_size = int((num/4)*3)
        print "++++++++"
        print train_size
        x_test = np.asarray(xs[train_size:])
        y_test = np.asarray(ys[train_size:])
        # clf = svm.NuSVC()
        # clf_rbf = clf
        # x_train = np.asarray(xs[train_size:])
        # y_train = np.asarray(ys[train_size:])
        # clf_rbf.fit(x_test, y_test)
        # clf_rbf.fit(x_train, y_train)
        print "x_test"
        print np.shape(x_test)
        # print x_test
        print "y_test"
        print np.shape(y_test)
        # print y_test
        if if_train == 1:
            x_train = np.asarray(xs[:train_size])
            y_train = np.asarray(ys[:train_size])
            print "x_train"
            print type(x_train)
            print np.shape(x_train)
            # print x_train
            print "y_train"
            print type(x_train)
            print np.shape(y_train)
            # print y_train
            clf = svm.NuSVC()
            clf_rbf = clf
            clf_rbf.fit(x_train, y_train)
            clf_linear = svm.NuSVC(kernel='linear')
            clf_poly = svm.NuSVC(kernel='poly')
            clf_sigmoid = svm.NuSVC(kernel='sigmoid')
            rbf = clf_rbf.fit(x_train, y_train)
            linear = clf_linear.fit(x_train, y_train)
            poly = clf_poly.fit(x_train, y_train)
            sigmoid = clf_sigmoid.fit(x_train, y_train)
            rbf_score = rbf.score(x_test, y_test)
            linear_score = linear.score(x_test, y_test)
            poly_score = poly.score(x_test, y_test)
            sigmoid_score = sigmoid.score(x_test, y_test)
            print 'rbf=%.8f\nlinear=%.8f\npoly=%.8f\nsigmoid=%.8f' % (rbf_score, linear_score, poly_score, sigmoid_score)
        else:
            model = joblib.load(model_f)
            cv_score = model.score(x_test, y_test)
            print "cross validation score: %.8f" % cv_score

    def svm(self, xt, yt):
        print np.shape(self.all_features_flat)
        xs = np.asarray(self.all_features_flat)
        ys = np.asarray(self.tags_flat)
        self.clf = svm.NuSVC()
        clf_rbf = self.clf
        clf_linear = svm.NuSVC(kernel='linear')
        clf_poly = svm.NuSVC(kernel='poly')
        clf_sigmoid = svm.NuSVC(kernel='sigmoid')
        # clf_precomputed = svm.NuSVC(kernel='precomputed')
        self.clf.fit(xs, ys)
        clf_rbf.fit(xs, ys)
        clf_linear.fit(xs, ys)
        clf_poly.fit(xs, ys)
        clf_sigmoid.fit(xs, ys)
        # clf_precomputed.fit(xs, ys)
        tscore_rbf = self.clf.fit(xs, ys).score(xs, ys)
        tscore_linear = clf_linear.fit(xs, ys).score(xs, ys)
        tscore_poly = clf_poly.fit(xs, ys).score(xs, ys)
        tscore_sigmoid = clf_sigmoid.fit(xs, ys).score(xs, ys)
        # tscore_precomputed = clf_percomputed.fit(xs, ys).score(xs, ys)
        print 'rbf=%.8f\nlinear=%.8f\npoly=%.8f\nsigmoid=%.8f' % (tscore_rbf, tscore_linear, tscore_poly, tscore_sigmoid)
        joblib.dump(clf_rbf, 'models/svm_rbf.pkl')
        joblib.dump(clf_linear, 'models/svm_linear.pkl')
        joblib.dump(clf_poly, 'models/svm_poly.pkl')
        joblib.dump(clf_sigmoid, 'models/svm_sigmoid.pkl')
        # joblib.dump(clf_precomputed, 'models/svm_precomputed.pkl')
        print "Done with classifier\nPredict..."
        self.rt_test2()


    # print "Starting predict on test feature 0: "
    # rt_data_dict = self.rt_test()
    # rt_data = rt_data_dict.values()
    # self.real_time_vote = []
    # for i in rt_data:
    # result = self.clf.predict(i)
    # 	self.real_time_vote.append(result)
    # like_vote = 0
    # dislike_vote = 0
    # for e in self.real_time_vote:
    # 	if e == 1:
    # 		like_vote += 1
    # 	else:
    # 		dislike_vote += 1
    # if like_vote > dislike_vote:
    # 	result = 1
    # 	print "Predict as like!!!"
    # elif like_vote < dislike_vote:
    # 	result = -1
    # 	print "Predict as dislike!!!"
    # else:
    # 	result = 0
    # 	print "Predict as 'Never Mind'!!!"
    # print np.array(self.all_features())
    # print self.test_data
    # result = clf.predict(self.test_data)
    # print result
    # if result == 0:
    # 	print "predicted as dislike!"
    # else:
    # 	print "predicted as like!"
    # print "predict on test feature 1: "
    # result1 = clf.predict(self.test_data1)
    # print result1
    # if result == 0:
    # 	print "predicted as dislike!"
    # else:
    # 	print "predicted as like!"
    # return clf

    def rt_test(self):
        cur_f = self.el.realtime_data()
        rt_data = cur_f.values()
        rt_dic = rt_data[0]
        self.t_features = []
        rt_list = []
        chs = []
        for k, v in rt_dic.iteritems():
            chs.append(k)
            rt_list.append(v)
        print np.shape(rt_list)
        rt_dic = self.feature_psd(rt_list, chs)
        return rt_dic


    def cross_validation_old(self):
        clf = joblib.load('svm.pkl')
        result = -1
        vfs_dic = self.el.validation_loader()
        for n, f in vfs_dic:
            # initial a not existed label
            if_correct = 0
            label = -2
            for l in self.tag_dic:
                if l in f:
                    label = self.tag_dic[l]
            rt_data = f.values()
            rt_dic = rt_data[0]
            self.t_features = []
            rt_list = []
            chs = []
            count = 0
            c_like = c_dislike = c_idk = 0
            for k, v in rt_dic.iteritems():
                length = len(v)
                break
            term = int(math.floor(length / 64))
            for k, v in rt_dic.iteritems():
                rt_list.append(v.tolist())
                chs.append(k)
            rt_use = (np.array(rt_list)).T
            point = 0
            for t in range(term - 1):
                sec_wid = []
                for i in range(128):
                    sec_wid.append(rt_use[point + i])
                sec_wid_use = np.array(sec_wid).T
                sec_wid2 = []
                for l in sec_wid_use:
                    sec_wid2.append(np.array(l))
                sec_dic = self.feature_psd(sec_wid2, chs)
                sec_flat = []
                sec_list = []
                for k1, v1 in sec_dic.iteritems():
                    sec_list.append(v1)
                sec_flat = (np.array(sec_list)).flatten()
                result = clf.predict(sec_flat)
                # right classify
                if (result == 1) and (label == 0):
                    if_correct = 1
                elif result == 0 and (label == 2 or label == 1):
                    if_correct = 1
                else:
                    pass
                # USE SEC_DIC TO DO 1-SECOND BASED MACHINE LEARNING REAL TIME!!!!
                point += 64
            if c_like > c_dislike:
                predict = 1
                print "Predict as like!!!"
            elif c_like < c_dislike:
                predict = -1
                print "Predict as dislike!!!"
            else:
                predict = 0
                print "Predict as 'Never Mind'!!!"
            return predict


    def rt_test2(self):
        clf = joblib.load('models/svm_sigmoid.pkl')
        knn = joblib.load('models/neigh.pkl')
        result = -1
        cur_f = self.el.realtime_data()
        rt_data = cur_f.values()
        rt_dic = rt_data[0]
        self.t_features = []
        rt_list = []
        chs = []
        count = 0
        c_like_svm = c_like_knn = c_dislike_svm = c_dislike_knn = c_idk_svm = c_idk_kmm = 0
        for k, v in rt_dic.iteritems():
            length = len(v)
            break
        term = int(math.floor(length / 64))
        for k, v in rt_dic.iteritems():
            rt_list.append(v.tolist())
            chs.append(k)
        rt_use = (np.array(rt_list)).T
        # print rt_use
        # print np.shape(rt_use)
        point = 0
        count = 0
        for t in range(term - 1):
            sec_wid = []
            for i in range(128):
                sec_wid.append(rt_use[point + i])
            sec_wid_use = np.array(sec_wid).T
            # print sec_wid_use
            sec_wid2 = []
            for l in sec_wid_use:
                sec_wid2.append(np.array(l))
            # print sec_wid2
            # print "sec_wid2.size=", np.shape(sec_wid2)
            sec_dic = self.feature_psd(sec_wid2, chs)
            if count == (term - 2):
                plt_wid = sec_wid2
                plt_chs = chs
            sec_flat = []
            sec_list = []
            for k1, v1 in sec_dic.iteritems():
                sec_list.append(v1)
            sec_flat = (np.array(sec_list)).flatten()
            result_svm = clf.predict(sec_flat)
            result_knn = knn.predict(sec_flat)
            if result_svm == 1:
                c_like_svm += 1
            else:
                c_dislike_knn += 1
            if result_knn == 1:
                c_like_knn += 1
            else:
                c_dislike_knn += 1
            count += 1
            # print sec_dic.keys()
            # print "key_num\n", len(sec_dic.keys())
            # for k, v in sec_dic.iteritems():
            # 	print len(v)
            # USE SEC_DIC TO DO 1-SECOND BASED MACHINE LEARNING REAL TIME!!!!
            point += 64
        if c_like_svm > c_dislike_svm:
            svm_predict = 1
            print "SVM predict as like!!!"
        elif c_like_svm < c_dislike_svm:
            svm_predict = -1
            print "SVM predict as dislike!!!"
        else:
            svm_predict = 0
            print "SVM predict as 'Never Mind'!!!"
        if c_like_knn > c_dislike_knn:
            knn_predict = 1
            print "KNN predict as like!!!"
        elif c_like_knn < c_dislike_knn:
            knn_predict = -1
            print "KNN predict as dislike!!!"
        else:
            knn_predict = 0
            print "KNN predict as 'Never Mind'!!!"
        self.feature_psd_plot(plt_wid, plt_chs, 1)
        return {'knn_result': knn_predict, 'svm_result': svm_predict}


    # 	for t in range(term):
    # 		for i in range(128):

    # 	chs.append(k)
    # 	rt_list.append(v)
    # 	count += 1

    # # print rt_list
    # rt_dic = self.feature_psd(rt_list, chs)

    def rt_test1(self):
        test_file = self.el.realtime_data()
        rt_data = test_file.values()
        rt_dic = rt_data[0]
        print "t_dic\n", rt_dic
        print rt_dic['O1']
        self.all_feature_test = []
        test_vote = []
        for k, v in rt_dic.iteritems():
            one_video_data = []
            one_video_ch = []
            for data in v:
                one_video_data.append(data)
                video_terms = int(math.floor(len(data) / 64))
            start = 0
            for i in range(video_terms):
                wid_sec = []
                for one_ch_data in one_video_data:
                    wid_sec_ch = one_ch_data[start:(start + 128)]
                    wid_sec.append(wid_sec_ch)
                start += 64
                if np.shape(wid_sec)[1] == 128:
                    sec_psd_dic = self.feature_psd(wid_sec, one_video_ch)
                    sec_psd_list = []
                    new_sec_dic = {}
                    for k, v in sec_psd_dic.iteritems():
                        new_k = self.feature_dict[k]
                        new_sec_dic[new_k] = v
                    key_list = new_sec_dic.keys()
                    key_list.sort()
                    for key in key_list:
                        sec_psd_list.append(new_sec_dic[key])
                    print "***********************************"
                    print np.shape(sec_psd_list)
                    print "***********************************"
                    self.all_features_test.extend(sec_psd_list)
        return self.all_features_test


    def remove_artifact(self, data):
        pass


    def emg(self, data):
        pass


    def eog(self, data):
        pass


    def pca(self):
        pass


    def pca_util(self, d):
        np.random.seed(4294967295)
        mu_vec1 = np.array([0, 0, 0])
        cov_mat1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20).T
        assert class1_sample.shape == (3, 20), "The matrix has not the dimension 3x20"
        mu_vec2 = np.array([1, 1, 1])
        cov_mat2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 20).T
        # print class2_sample
        assert class1_sample.shape == (3, 20), "The matrix has not the dimensions 3x20"
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d import proj3d

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        mpl.rcParams['legend.fontsize'] = 10
        ax.plot(class2_sample[0, :], class2_sample[1, :], class1_sample[2, :], 'o', markersize=8, color='blue', alpha=0.5,
                label='class1')
        ax.plot(class2_sample[0, :], class2_sample[1, :], class2_sample[2, :], '^', markersize=8, color='red', alpha=0.5,
                label='class2')
        plt.title('samples for class 1 and 2')
        ax.legend(loc='upper right')
        # plt.draw()
        # plt.show()
        all_samples = np.concatenate((class1_sample, class2_sample), axis=1)
        # print "all_samples"
        # print all_samples
        assert all_samples.shape == (3, 40), "The matrix has not the 3x40"
        self.pca_dev(all_samples, d)


    def line(self):
        inspect.currentframe().f_back.f_lineno


    def pca_dev(self, all_samples, d):
        print "compare"
        mean_list = []
        for layer in all_samples:
            mean_list.append([np.mean(layer)])
        # print layer
        # break
        mean_vector = np.array(mean_list)
        # print "mean_ary"
        # print mean_ary
        # print all_samples[0, :]
        # mean_x = np.mean(all_samples[0, :])
        # mean_y = np.mean(all_samples[1, :])
        # mean_z = np.mean(all_samples[2, :])
        # mean_vector = np.array([[mean_x], [mean_y], [mean_z]])
        # print "mean_vector"
        # print mean_vector
        # print('mean vec: \n', mean_vector)
        # print all_samples.shape
        n = len(mean_vector)
        scatter_matrix = np.zeros((n, n))
        # print "scatter_matrix 1st"
        # print scatter_matrix
        # print all_samples.shape[1]
        for i in range(all_samples.shape[1]):
            # if i == 1:
            # 	print all_samples[:, i]
            n = len(all_samples[:, i])
            scatter_matrix += (all_samples[:, i].reshape(n, 1) - mean_vector).dot(
                (all_samples[:, i].reshape(n, 1) - mean_vector).T)
        # print 'Scatter matrix: \n', scatter_matrix
        # alternative
        # print "cov_mat"
        # print all_samples[0, :]
        # cov_mat = np.cov([all_samples[0, :], all_samples[1, :], all_samples[2, :]])
        # print 'Covariance Matrix:\n', cov_mat
        eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)
        # eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
        # print self.line()
        # for i in range(len(eig_val_sc)):
        # n = len(eig_vec_sc[:, i])
        # print n
        # eigvec_sc = eig_vec_sc[:, i].reshape(1, n).T
        # eigvec_cov = eig_vec_cov[:, i].reshape(1, 3).T
        # assert eigvec_sc.all() == eigvec_cov.all(), 'eigenvectors are not identifcal'
        for i in range(len(eig_val_sc)):
            n = len(eig_vec_sc[:, i])
            eigv = eig_vec_sc[:, i].reshape(1, n).T
            np.testing.assert_array_almost_equal(scatter_matrix.dot(eigv), eig_val_sc[i] * eigv, decimal=6, err_msg='',
                                                 verbose=True)
        from matplotlib.patches import FancyArrowPatch


        class Arrow3D(FancyArrowPatch):

            def __init__(self, xs, ys, zs, *args, **kwargs):
                FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
                self._verts3d = xs, ys, zs

            def draw(self, renderer):
                xs3d, ys3d, zs3d = self._verts3d
                xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
                self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

            # FancyArrowPatch.draw(self, renderer)

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(all_samples[0, :], all_samples[1, :], all_samples[2, :], 'o', markersize=8, color='green', alpha=0.2)
        # ax.plot([mean_x], [mean_y], [mean_z], 'o', markersize=10, color='red', alpha=0.5)
        # for v in eig_vec_sc.T:
        # 	a = Arrow3D([mean_x, v[0]], [mean_y, v[1]], [mean_z, v[2]], mutation_scale=20, lw=3, color="r")
        # 	ax.add_artist(a)
        # ax.set_xlabel('x_values')
        # ax.set_ylabel('y_values')
        # ax.set_zlabel('z_values')
        # plt.title('Eigenvectors')
        # plt.show()
        for ev in eig_vec_sc:
            np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
        eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:, i]) for i in range(len(eig_val_sc))]
        eig_pairs.sort()
        eig_pairs.reverse()
        for i in eig_pairs:
            print(i[0])
        print eig_pairs
        # print eig_pairs[0][1].reshape(3, 1)
        n = len(eig_pairs[0][1])
        print 'eig_pairs\n', eig_pairs
        # print d
        for i in range(d - 1):
            if i == 0:
                matrix_w = np.hstack((eig_pairs[i][1].reshape(n, 1), eig_pairs[i + 1][1].reshape(n, 1)))
            # print "i=%d" % i
            else:
                matrix_w = np.hstack((matrix_w, eig_pairs[i + 1][1].reshape(n, 1)))
            # print 'i=%d' % i
        transformed = matrix_w.T.dot(all_samples)
        print 'matrix_w.shape\n', np.shape(matrix_w)
        print 'all_samples.shape\n', np.shape(all_samples)
        assert transformed.shape == (d, (np.shape(all_samples))[1])
        plt.plot(transformed[0, 0:20], transformed[1, 0:20], 'o', markersize=7, color='orange', alpha=0.5, label='class1')
        plt.plot(transformed[0, 20:40], transformed[1, 20:40], '^', markersize=7, color='purple', alpha=0.5, label='class2')
        plt.xlim([-4, 4])
        plt.ylim([-4, 4])
        plt.xlabel('x_values')
        plt.ylabel('y_values')
        plt.legend()
        plt.title('Transformed samples with class labels')
        plt.draw()
        plt.show()


    # print matrix_w

    def pca_built(self, all_samples):
        from sklearn.decomposition import PCA as sklearnPCA

        sklearn_pca = sklearnPCA(n_components=2)
        sklearn_transf = sklearn_pca.fit_transform(all_samples.T)
        sklearn_transf = sklearn_transf * (-1)
        plt.plot(sklearn_transf[0:20, 0], sklearn_transf[0:20, 1], 'o', markersize=7, color='yellow', alpha=0.5,
                 label='class1')
        plt.plot(sklearn_transf[20:40, 0], sklearn_transf[20:40, 1], '^', markersize=7, color='black', alpha=0.5,
                 label='class2')
        plt.xlabel('x_values')
        plt.ylabel('y_values')
        plt.xlim([-4, 4])
        plt.ylim([-4, 4])
        plt.legend()
        plt.title('Transformed samples with class labels from built PCA')
        plt.draw()
        plt.show()


    def ica(self, data):
        pass


    def knn(self, opt):
        xy_dic = self.acquire_module_data(opt)
        xs = xy_dic['xs']
        ys = xy_dic['ys']
        neigh = KNeighborsClassifier(n_neighbors=10)
        neigh.fit(xs, ys)
        score = neigh.fit(xs, ys).score(xs, ys)
        print "knn fit score: %.8f" % score
        joblib.dump(neigh, 'models/neighb.pkl')


    # neigh.ppredict([n0, n1, ...])

    def km(self, data):
        xy_dic = self.acquire_module_data(opt)
        xs = xy_dic['xs']
        ys = xy_dic['ys']


    # TODO

    def hmm(self, data):
        pass


    def nmf(self, data):
        pass


    def classify(self, data):
        pass


    def normalize_train(self, wid, ch_name, terms):
        wid = np.asarray(self.data, dtype=np.float64)
        pmax = sys.float_info.min
        pmin = sys.float_info.max
        for i in range(7 * terms):
            for n in range(2):
                s = 0
                for i in range(60):
                    wid = wid[s, s + 128]
                    psd_wid = self.feature_psd(wid, ch_name)
                    for k, v in psd_wid.iteritems():
                        if np.amax(v) > pmax:
                            pmax = np.amax(v)
                        if np.amin(v) < pmin:
                            pmin = np.min(v)
                    s += 64
                    if i == 59:
                        s = 0
        nm_psd = dict((el, []) for el in psd_w.keys())
        for i in range(3 * self.train_terms):
            for n in range(2):
                s = 0
                if n == 0:
                    label = 'h_'
                else:
                    label = 'u_'
                for i in range(60):
                    w = wid[s, s + 128]
                    psd_w = self.feature_psd(w, ch_name)
                    for k, v in psd_w.iteritems():
                        nm_psd[label + k] = np.asarray(list(map((lambda x: (x - pmin) / (pmax - pmin)), v)))
                    s += 64
                    if i == 59:
                        s = 0
        # return a normalized dict
        return nm_psd


    def normalize_dev(self, type):
        wid = np.asarray(self.data, dtype=np.float64)
        pmax = sys.float_info.min
        pmin = sys.float_info.max
        s = 0
        for i in range(self.terms):
            wid = wid[s, s + 128]
            psd_wid = self.feature_psd(wid)
            for k, v in psd_wid.iteritems():
                if np.amax(v) > pmax:
                    pmax = np.amax(v)
                if np.amin(v) < pmin:
                    pmin = np.min(v)
            s += 64
        nm_psd = dict((el, []) for el in psd_w.keys())
        s = 0
        for i in range(self.terms):
            w = wid[s, s + 128]
            psd_w = self.feature_psd(w)
            for k, v in psd_w.iteritems():
                nm_psd[k] = np.asarray(list(map((lambda x: (x - pmin) / (pmax - pmin)), v)))
            s += 64
        return nm_psd


    def feature_nmf(self, wid):
        pass


    def feature_psd_mean(self, wid, ch_name):
        freq = np.fft.fftfreq(128, 1.0 / 128)
        fftw = []
        for i in wid:
            ffti = fft(i)
            fftw.append(ffti)
        fftwid = np.asarray(fftw, dtype=np.float64)
        fdic = {}
        ch0 = 0
        for cn, d in zip(ch_name, wid):
            k0 = cn + "_delta"
            delta = mne.filter.band_pass_filter(x=d, Fs=128, Fp1=0.5, Fp2=4, l_trans_bandwidth=0.0)
            fdic[k0] = delta
            fft_delta = fft(delta)
            freq = np.fft.fftfreq(delta.shape[-1])
            k1 = cn + "_theta"
            theta = mne.filter.band_pass_filter(x=d, Fs=128, Fp1=4, Fp2=8)
            fdic[k1] = theta
            k2 = cn + "_alpha"
            alpha = mne.filter.band_pass_filter(x=d, Fs=128, Fp1=8, Fp2=16)
            fdic[k2] = alpha
            k3 = cn + "_beta"
            beta = mne.filter.band_pass_filter(x=d, Fs=128, Fp1=16, Fp2=32)
            fdic[k3] = beta
            k4 = cn + "_gamma"
            gamma = mne.filter.band_pass_filter(x=d, Fs=128, Fp1=32, Fp2=63.5)
            fdic[k4] = gamma
        psd_dic = dict((el, []) for el in fdic.keys())
        check = 0
        for k, v in fdic.iteritems():
            f, psd_dic[k] = signal.welch(v, fs=128, nperseg=128)
        return psd_dic


    def feature_psd(self, wid, ch_name):
        # print "hiiiiiiiiiiii"
        freq = np.fft.fftfreq(128, 1.0 / 128)
        fftw = []
        for i in wid:
            ffti = fft(i)
            fftw.append(ffti)
        fftwid = np.asarray(fftw, dtype=np.float64)
        fdic = {}
        ch0 = 0
        for cn, d in zip(ch_name, wid):
            k0 = cn + "_delta"
            delta = mne.filter.band_pass_filter(x=d, Fs=128, Fp1=0.5, Fp2=4, l_trans_bandwidth=0.0)
            fdic[k0] = delta
            fft_delta = fft(delta)
            freq = np.fft.fftfreq(delta.shape[-1])

            # plt.figure(1)
            # if ch0 == 0:
            # 	plt.subplot(211)
            # 	plt.plot(np.arange(128), delta)
            # 	plt.subplot(212)
            # 	plt.plot(freq, fft_delta)
            # plt.show()
            # ch0 += 1

            k1 = cn + "_theta"
            theta = mne.filter.band_pass_filter(x=d, Fs=128, Fp1=4, Fp2=8)
            fdic[k1] = theta
            k2 = cn + "_alpha"
            alpha = mne.filter.band_pass_filter(x=d, Fs=128, Fp1=8, Fp2=16)
            fdic[k2] = alpha
            k3 = cn + "_beta"
            beta = mne.filter.band_pass_filter(x=d, Fs=128, Fp1=16, Fp2=32)
            fdic[k3] = beta
            k4 = cn + "_gamma"
            gamma = mne.filter.band_pass_filter(x=d, Fs=128, Fp1=32, Fp2=63.5)
            fdic[k4] = gamma
        psd_dic = dict((el, []) for el in fdic.keys())
        check = 0
        for k, v in fdic.iteritems():
            f, psd_dic[k] = signal.welch(v, fs=128, nperseg=128)
            # if check==0:
            # plt.figure(212)
            # plt.semilogy(f, psd_dic[k])
            # plt.xlabel('frequency [Hz]')
            # plt.ylabel('PSD')
            # print type(psd_dic[k])
            # print len(psd_dic[k])
            # print psd_dic[k]
            check += 1
        # print psd_dic
        # for k, v in psd_dic.iteritems():
        # print "@@@@@@@@@@@@@@@@@@@"
        # print psd_dic
        # print "@@@@@@@@@@@@@@@@@@@"
        # print check
        # print "yesyesyes"
        # print psd_dic
        return psd_dic


    def feature_psd_plot(self, wid, ch_name, if_draw):
        # print "hiiiiiiiiiiii"
        freq = np.fft.fftfreq(128, 1.0 / 128)
        fftw = []
        for i in wid:
            ffti = fft(i)
            fftw.append(ffti)
        fftwid = np.asarray(fftw, dtype=np.float64)
        fdic = {}
        ch0 = 0
        for cn, d in zip(ch_name, wid):
            k0 = cn + "_delta"
            delta = mne.filter.band_pass_filter(x=d, Fs=128, Fp1=0.5, Fp2=4, l_trans_bandwidth=0.0)
            fdic[k0] = delta
            fft_delta = fft(delta)
            freq = np.fft.fftfreq(delta.shape[-1])
            k1 = cn + "_theta"
            theta = mne.filter.band_pass_filter(x=d, Fs=128, Fp1=4, Fp2=8)
            fdic[k1] = theta
            k2 = cn + "_alpha"
            alpha = mne.filter.band_pass_filter(x=d, Fs=128, Fp1=8, Fp2=16)
            fdic[k2] = alpha
            k3 = cn + "_beta"
            beta = mne.filter.band_pass_filter(x=d, Fs=128, Fp1=16, Fp2=32)
            fdic[k3] = beta
            k4 = cn + "_gamma"
            gamma = mne.filter.band_pass_filter(x=d, Fs=128, Fp1=32, Fp2=63.5)
            fdic[k4] = gamma
        psd_dic = dict((el, []) for el in fdic.keys())
        check = 0
        print psd_dic
        if if_draw == 0:
            for k, v in fdic.iteritems():
                f, psd_dic[k] = signal.welch(v, fs=128, nperseg=128)
                check += 1
            print "check =========================== %d" % check
        elif if_draw == 1:
            count = 0
            plt.figure(1)
            for k, v in fdic.iteritems():
                if count >= 5:
                    break
                if count == 0:
                    t_plot = np.arange(0.0, 1.0, 1.0 / 128.0)
                if "O2" in k:
                    sp = plt.subplot(5, 1, count + 1)
                    plt.plot(t_plot, v)
                    sp.set_title(k)
                    f, psd_dic[k] = signal.welch(v, fs=128, nperseg=128)
                    count += 1
            plt.show()
        return psd_dic


    def fft_train(self):
        tw = []
        for i in range(0, self.terms):
            pass


    def spectrum_test(self):
        data = data_cosine(N=1024, A=0.1, sampling=1024, freq=200)
        print data
        print len(data)
        print type(data)
        p = Periodogram(data, sampling=1024)
        p()
        p.plot(marker='o')


    def spectrum_test1(self):
        ar, ma, rho = arma_estimate(marple_data, 15, 15, 30)


    def psd(self):
        el = dle.DataLoaderEmotiv()
        dic = {}
        dic = el.edf_loader()

        for k in dic:
            if k == "F3":
                x = np.fft(dic[k])
                plt.plot(abs(x))
                pl.show()


class Chdir:
    def __init__(self, newPath):
        self.savedPath = os.getcwd()
        os.chdir(newPath)

    def __del__(self):
        os.chdir(self.savedPath)


if __name__ == "__main__":
    lr = LikeRecog()
    # lr.run_train(0, 1)
    # lr.cross_validation('', 1)
    # lr.rt_test2()
    # lr.pca_util(2)
    # lr.feature_psd(lr.data)
    # lr.svm("")
    # lr.fft()
    # lr.ann(0)
    # lr.knn(0)

