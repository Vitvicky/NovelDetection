# coding:UTF-8
# !/usr/bin/env python
from sklearn import svm
import arff
import numpy as np
import random
import time
from random import shuffle
from ADWIN import *


class src_tar_generate(object):

    def __init__(self, data_name):
        self.data_name = data_name
        self.cusion_size = 100
        self.class_dict = {}
        self.original_data = None
        self.sliding_window_error = []
        self.data_buffer = []
        self.global_set = set()

        self.read_data()

    def train_svm(self, train_data_feature, train_data_label):
        svm_model = svm.SVC(C=20.0, kernel='rbf', gamma=0.1)
        svm_model.fit(train_data_feature, train_data_label, sample_weight=None)
        return svm_model

    def read_data(self):
        read_path = '/home/wzy/Coding/Data/' + self.data_name + '/ori/' + self.data_name + '.arff'
        data_set = arff.load(open(read_path, 'rb'))
        self.original_data = np.array(data_set['data'])

    def initial_classifier(self):
        initial_data = self.original_data[0:1000]
        self.original_data = self.original_data[1000:]
        initial_data_feature = []
        initial_data_label = []

        for i in initial_data:
            initial_data_feature.append(i[:-1])
            initial_data_label.append(i[-1])

        initial_svm_model = self.train_svm(initial_data_feature, initial_data_label)

        return initial_svm_model

    def detect_drift(self, sliding_window_error):
        change_point = -1

        # sw = self.gateway.jvm.java.util.ArrayList()
        # for i in xrange(len(sliding_window_error)):
        #     # print "sliding_window_error "+str(i)+" data point "
        #     # process = [float(x) for x in sliding_window_error[i]]
        #     sw.append(float(sliding_window_error[i]))
        #
        # # print "sw: ", sw
        # change_point = self.app.detectSourceChange(sw)
        conf = 0.2
        change_point = detect_change(sliding_window_error, conf)
        return change_point

    def get_novel_class(self, novel_class_1, novel_class_2):
        novel_class_list1 = []
        novel_class_list2 = []
        no_novel_class_list = []
        print "before novel generate, length of ori data is: ", len(self.original_data)
        for index in range(0, len(self.original_data)):
            current_data = self.original_data[index]
            if current_data[-1] == novel_class_1:
                novel_class_list1.append(current_data)
            elif current_data[-1] == novel_class_2:
                novel_class_list2.append(current_data)
            else:
                no_novel_class_list.append(current_data)
        
        self.original_data = np.array(no_novel_class_list)
        novel_class_list1 = np.array(novel_class_list1)
        novel_class_list2 = np.array(novel_class_list2)
        print "after novel generate, length of ori data is: ", len(self.original_data)
        print "the novel class 1 length is: "+str(len(novel_class_list1))+" and novel class 2 length " \
                                                                          "is: ", len(novel_class_list2)

        return novel_class_list1, novel_class_list2

    def detect_point_list(self):
        change_point_list = []
        update_data_feature = []
        update_data_label = []
        max_window_size = 5000
        novel_class_list1, novel_class_list2 = self.get_novel_class('class6', 'class7')
        len_ori_data = len(self.original_data)
        print "before point detection, ori data length is: ", len(self.original_data)
        # initial classifier and predict point
        svm_model = self.initial_classifier()
        # data_index = 0
        for data_index in range(1000, 61000):
            print "Current data index: ", data_index
            instance_feature = self.original_data[data_index][:-1]
            instance_label = self.original_data[data_index][-1]
            predict_label = svm_model.predict(instance_feature)
            # print "predict label: "+str(predict_label)+" real label: ", instance_label
            if predict_label == instance_label:
                predict_score = 0
            else:
                predict_score = 1

            # print "predict_score: ", predict_score
            if len(self.sliding_window_error) < 2 * self.cusion_size:
                self.data_buffer.append(self.original_data[data_index])
                self.sliding_window_error.append(predict_score)
                # data_index += 1

            # just keep the slicing_window size same with data_buffer size
            elif len(self.sliding_window_error) < max_window_size:
                self.data_buffer.append(self.original_data[data_index])
                self.sliding_window_error.append(predict_score)

                change_point = self.detect_drift(self.sliding_window_error)
                print "change_point :", change_point
                # find the change point
                if change_point != -1 and change_point != 0:
                    # add the real change point to collector
                    left_index_window = data_index - len(self.sliding_window_error)
                    real_change_point = left_index_window + change_point
                    print "current window size is: ", len(self.sliding_window_error)
                    print "\n less than max_window_size, current real change point is: ", real_change_point
                    change_point_list.append(real_change_point)

                    # remove the previous points of change point
                    self.data_buffer = self.data_buffer[change_point:]
                    self.sliding_window_error = self.sliding_window_error[change_point:]

                    # update classifier
                    # 1. if the data in buffer is not long, add more data to current buffer to train svm
                    if len(self.sliding_window_error) < 2 * self.cusion_size:
                        data_extension = self.original_data[data_index:data_index+self.cusion_size]
                        update_data_feature = [item[:-1] for item in self.data_buffer] + \
                                              [item[:-1] for item in data_extension]
                        update_data_label = [item[-1] for item in self.data_buffer] + \
                                            [item[-1] for item in data_extension]

                    # 2. if the data in buffer is enough, directly use whole buffer
                    else:
                        update_data_feature = [item[:-1] for item in self.data_buffer]
                        update_data_label = [item[-1] for item in self.data_buffer]

                    svm_model = self.train_svm(update_data_feature, update_data_label)

            # window reach the max size
            else:
                del self.sliding_window_error[-1]
                self.sliding_window_error.append(predict_score)
                del self.data_buffer[-1]
                self.data_buffer.append(self.original_data[data_index])

                change_point = self.detect_drift(self.sliding_window_error)
                print "change_point :", change_point
                # detect change point
                if change_point != -1 and change_point != 0:
                    # add the real change point to collector
                    left_index_window = data_index - len(self.sliding_window_error)
                    real_change_point = left_index_window + change_point
                    print "\n more than max_window_size, current real change point is: ", real_change_point
                    change_point_list.append(real_change_point)

                    # remove the previous points of change point
                    self.data_buffer = self.data_buffer[change_point:]
                    self.sliding_window_error = self.sliding_window_error[change_point:]

                    # update classifier for the buffer is full
                    if len(self.sliding_window_error) < 2*self.cusion_size:
                        update_data_feature = self.data_buffer[:-1] + self.original_data[
                                                                      data_index:data_index + self.cusion_size][:-1]
                        update_data_label = self.data_buffer[-1] + self.original_data[
                                                                   data_index:data_index + self.cusion_size][-1]

                    else:
                        update_data_feature = [item[:-1] for item in self.data_buffer]
                        update_data_label = [item[-1] for item in self.data_buffer]

                    svm_model = self.train_svm(update_data_feature, update_data_label)

                # else:
                #     data_index += 1

            print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        return change_point_list, novel_class_list1, novel_class_list2

    def generate_source_target(self, change_point_list, rate):
        # global_set = set()
        source_data_list = []
        target_data_list = []
        block = []
        start = 0
        novel_label = -1
        for change_index in range(len(change_point_list) - 1):
            # start to iteration
            if start == 0:
                block = self.original_data[0:change_point_list[change_index]]
                self.global_set = self.summary_label_block(block)
                print "In index of 0 and "+str(change_point_list[change_index])+" , the label set is: ", self.global_set
                source_data_list, target_data_list = self.generate_data(start, block, novel_label, rate)
                print "This block's source, label set is: ", self.summary_label_block(source_data_list)
                print "This block's target, label set is: ", self.summary_label_block(target_data_list)
                start = 1
            else:
                left_index = change_point_list[change_index]
                right_index = change_point_list[change_index+1]
                block = self.original_data[left_index:right_index]
                current_set = self.summary_label_block(block)

                print "In index of "+str(change_point_list[change_index])+" and " +\
                      str(change_point_list[change_index]) + " , the label set is: ", current_set
                # find the novel class
                for label in current_set:
                    if label not in self.global_set:
                        novel_label = label

                source_data_block, target_data_block = self.generate_data(start, block, novel_label, rate)
                print "This block's source, label set is: ", self.summary_label_block(source_data_block)
                print "This block's target, label set is: ", self.summary_label_block(target_data_block)
                source_data_list += source_data_block
                target_data_list += target_data_block
                self.global_set.add(novel_label)
                print "========================================================================================"

        return source_data_list, target_data_list

    def summary_label_block(self, block):
        block_set = set()
        block_label = [item[-1] for item in block]
        for label in block_label:
            if label not in block_set:
                block_set.add(label)
        return block_set

    def generate_data(self, start, block, novel_label, rate):
        # just assign data to 2 buffer
        source_block_list = []
        target_block_list = []
        novel_block_list = []
        if start == 0:
            for instance in block:
                random_number = random.random()
                if random_number <= rate:
                    source_block_list.append(instance)
                else:
                    target_block_list.append(instance)

        # assign data to 2 buffer and gather novel class point
        else:
            for instance in block:
                if novel_label != -1 and instance[-1] == novel_label:
                    novel_block_list.append(instance)
                else:
                    random_number = random.random()
                    if random_number <= rate:
                        source_block_list.append(instance)
                    else:
                        target_block_list.append(instance)

        # shuffle novel and target buffer
        if len(novel_block_list) > 0:
            target_block_list += novel_block_list
            shuffle(target_block_list)

        return source_block_list, target_block_list



generate = src_tar_generate("Syndata_002")
# time.sleep(5)
# change_point_list, novel_class_list1, novel_class_list2 = generate.detect_point_list()
novel_class_list1, novel_class_list2 = generate.get_novel_class('class6', 'class7')
change_point_list = [6198,  7532, 8564, 9792, 11000, 12082, 13209, 14340, 15493, 16707,
                     17921, 19088, 20242, 21478, 22508, 23653, 24924, 26180, 27325, 28494, 29585, 30900, 32079,
                     33221, 34274, 35520, 36652, 37710, 38918, 40109, 41258, 42371, 43514, 44737, 45864, 47192,
                     48167, 49309, 50629, 51719, 52714, 54015, 55215, 56309, 57391, 58560, 59712, 61000]
# save_change_point = np.array(change_point_list)
# np.savetxt("save_change_point.txt", save_change_point)

# change_point_list = np.loadtxt("save_change_point.txt")
# change_point_list = [int(i) for i in change_point_list]
print change_point_list
# generate.generate_source_target(change_point_list, 0.2)

