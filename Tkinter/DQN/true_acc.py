import numpy as np
import pandas as pd
import random

test_dataset = pd.read_csv("dataRL/test99.csv")
test_label = pd.read_csv("dataRL/test_label99.csv")
dataset = pd.read_csv("dataRL/train99.csv")
label = pd.read_csv("dataRL/train_label99.csv")
def pred_label(classifiers):
	if len(classifiers.values) == 0:
		print "initial classifiers"
		return
	else:
		add_labels = classifiers.apply(lambda x: x.sum(), axis=1).values
		pred_labels = []
		for item1 in add_labels:
			if item1 > 0:
				pred_labels.append(1)
			else:
				pred_labels.append(-1)
		return pred_labels

def caculate_acc(pred_value,label):
	diff_labels = pred_value - label.values.T[0]
	count = 0
	for item2 in diff_labels:
		if item2 == 0:
			count+=1
		else:
			count = count
	accuracy = float(count)/len(label.values.T[0])

	return accuracy


arr1 = []
for item in np.array(dataset.columns.values).astype('int'):
	arr1.append(item-1)
validate1 = dataset.iloc[:,arr1]

pred_labels_1 = pred_label(validate1)
acc1 = caculate_acc(pred_labels_1,label)

print "train total acc=",acc1

arr2 = []
for item2 in np.array(test_dataset.columns.values).astype('int'):
	arr2.append(item2-1)
validate2 = test_dataset.iloc[:,arr2]

pred_labels_2 = pred_label(validate2)
acc2 = caculate_acc(pred_labels_2,test_label)

print "test total acc=",acc2

# arr = [84,34,74,6,8,69,54,58,4,96,38,94,21,64,2,30,99,85,5,71,77,40,49,57,83,67,18,52,37,1,32,31,15,80,86,63,91,44,65,56,62,36,75,50,14,51,25,60,61,97,90,29,70,22,53,11,93,95,13,12,19,76,92,46,98,23,3,87,41,42,66,10,26]
# arr = [12,84,34,74,83,51,95,16,72,90,6,24,59,9,88,8,77,69,39,54,65,7,57,91,58,44,15,32,22,60,4,81,31,20,97,73,55,96,36,11,33,29,35,38,40,62,48,14,94,78,56,50,23,76,63,49,47,10,98,21,87,1,42,70,68,85,67,80,82,3,26,19,46,79,13,53,64,89,52,43,18]
# arr = [54,1,88,35,16,78,47,58,53,40,34,77,62,48,52,69,14,43,15,33,38,31,19,56,57,63,99,27,23,10,22,82,12,45,8,7,74,67,92,90,73,55,98,85,26,25,66,39,18,97,96,28,64,71,79,70,95,83,68,49,75,4,51,32,94,76,72,29,86]
# arr = [72,17,35,65,63,59,94,99,19,33,8,16,36,53,7,97,70,39,57,12,21,32,66,40,44,90,58,82,22,41,37,45,87,15,6,62,60,31,24,23,29,30,73,56,48,18,67,46,80,25,64,4,84,10,55,75,26,42,43,47,49,2,11,77,1,3,79,5,27,34,88,74,76,98,38]
# arr = [72,17,35,65,63,59,94,99,19,33,8,16,36,53,7,97,70,39,57,12,21,32,66,40,44,90,58,82,22,41,37,45,87,15,6,62,60,31,24,23,29,30,73,56,48,18,67,46,80,25,64,4,84,10,55,75,26,42,43,47,49,2,11,77,1,3,79,5,27,34,88,74,76,98,38]
arr = [77,89,20,23,39,87,62,13,68,74,18,57,11,48,50,91,35,10,36,61,27,38,28,88,4,93,6,75,76,64,83,71,59,9,44,73,82,22,43,99,65,8,85,26,33,95,49,32,19,42,56,66,34,79,40,2,16,80,92,58,30,51,96,54,98,7,67,25,84,60,70,29,45,1,52,53,46,72,3,31,94]
result =[x-1 for x in arr]
# print result
#===========train===========
validate_train = dataset.iloc[:,result]
#===========test===========
validate_test = test_dataset.iloc[:,result]

# print validate

pred_labels_train = pred_label(validate_train)
pred_labels_test = pred_label(validate_test)

# ===========train===========
acc_train = caculate_acc(pred_labels_train,label)
#===========test===========
acc_test = caculate_acc(pred_labels_test,test_label)

print "choose train acc = ",acc_train,"choose test acc = ",acc_test,'choose length=',len(arr)
