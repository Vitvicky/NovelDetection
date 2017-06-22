# coding:UTF-8
#read data
from sklearn.datasets import load_svmlight_file

X_train, y_train = load_svmlight_file("C:\DataSet\Stream\mnist_100k_instances.data")
def load_data(path):
    index = 0
    with open(path) as file:
        for rows in file:
            lineText = []
            if index<10:
                line = rows.split(' ')[1:-1]
                process = [x.replace(':',',') for x in line]
                for item in process:
                    position,value = item.split(',')
                    lineText.append(int(position),int(value))
                print lineText
                index+=1

load_data('C:\DataSet\Stream\mnist_100k_instances.data')
