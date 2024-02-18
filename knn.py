
# **********************************************
# Import packages
# Layout of the code
# Function Definitions
# Variable Setting
# Main Execution

# **********************************************
# Import packages
import pandas as pd
import numpy as np
import math
import os
import time
import sys
#import matplotlib.pyplot as plt

# **********************************************
# Function Definitions
def f_calc_distance(r1, r2):
	dist  = 0.0
	for i in range(len(r1)-1):
		dist += (r1[i] - r2[i])**2
	return math.sqrt(dist)

def f_normalize_Min_Max(df):
	# Normalization by Min-Max
	# x scaled = (x - min(x)) / (max(x) - min(x))
	for (columnName, columnData) in df.items():
		min_x = min(columnData)
		max_x = max(columnData)
		j = df.columns.get_loc(columnName)

		for (i,x) in columnData.items():
			tmp = (x - min_x) / (max_x - min_x)
			df.iloc[i, j] = tmp

		if j == len(df.count())-2: # to exclude the last column from dataset, it is class column
			return df

	return df


def f_classification_with_normalization(df_train, df_test, knn):
	print("Min-Max Normalization")
	# Normalization by Min-Max
	df_train_normalized = df_train.copy()
	df_test_normalized = df_test.copy()

	# Loop each column
	for (columnName, columnData) in df_train_normalized.items():

		# Get min max from training data
		# print("Normalize columne: " + columnName)
		min_x = min(columnData)
		max_x = max(columnData)

		# Normalize the training data
		j = df_train_normalized.columns.get_loc(columnName)

		# Loop each item in column
		for (i,x) in columnData.items():
			if (max_x - min_x) == 0:
				tmp = 0
			else:
				tmp = (x - min_x) / (max_x - min_x)
			df_train_normalized.iloc[i, j] = tmp

		# Normalize the testing data based on the min max of training
		j = df_test_normalized.columns.get_loc(columnName)

		# Loop each item in column
		for (i,x) in df_test_normalized[columnName].items():
			if (max_x - min_x) == 0:
				tmp = 0
			else:
				tmp = (x - min_x) / (max_x - min_x)
			df_test_normalized.iloc[i, j] = tmp

		# exclude the last column from dataset, it is classification column
		if j == len(df_train_normalized.count())-2: 
			break

	return f_classification(df_train_normalized, df_test_normalized, knn) # call the classification with the nomalized datasets

def f_classification(df_train, df_test, knn):
	print("Classifying")
	l_out = list()

	for j, rw in df_test.iterrows():

		l_dist = list()

		for index, r in df_train.iterrows():
			dist = f_calc_distance(rw, r)
			l_dist.append((r, dist))

		l_dist.sort(key=lambda a: a[1])

		l_nei = list()

		# get the nearest neighbour based on knn value
		for i in range(knn):
			l_nei.append(l_dist[i][0])

		output_values = [r[-1] for r in l_nei]
		prediction = (max(set(output_values), key=output_values.count)) # get the majority class among nearest neighbour

		l_out.append((pd.concat([rw, pd.Series(prediction, index=['predicted_class'])])))

	df_classified = pd.DataFrame(l_out)

	return df_classified # return df_test with predicted_class column

def f_k_cross_validation(df, k_fold, knn):
	l_sample = list()
	df_metrics = pd.DataFrame()

	# shuffle and split the dataset
	df_shuffled = df.sample(frac=1).reset_index(drop=True)
	e = -1
	for i in range(k_fold):
	    s = e + 1
	    e = int((len(df_shuffled))*(i+1)/k_fold) - 1
	    if e > len(df_shuffled)-1 or len(df_shuffled)-e < len(df_shuffled)/k_fold:
	        e = len(df_shuffled) - 1
	    l_sample.append(df_shuffled[s:e+1])

	for i in range(k_fold):
		df_test_kfold = pd.DataFrame()
		df_test_kfold = l_sample[i]
		df_test_kfold = df_test_kfold.reset_index(drop=True)

		df_train_kfold = pd.DataFrame()
		# the rest is training
		for j in range(k_fold):
			if i != j:
				df_train_kfold = pd.concat([df_train_kfold, l_sample[j]])
				df_train_kfold = df_train_kfold.reset_index(drop=True)

		df_classified = f_classification_with_normalization(df_train_kfold, df_test_kfold, knn)
		df_metrics = pd.concat([df_metrics, f_calc_eval_metrics(df_classified)])
	return df_metrics
	
def f_calc_eval_metrics(df):
	print("Calculate the evaluation metrics")
	list_metrics = list()
	accuracy = round(np.count_nonzero(df.predicted_class == df.Class) / len(df.Class), 3)

	# Accuracy is for total, not by class
	list_metrics.append((-99, "All", "Accuracy", accuracy))

	# For each class, calculate the precision and recall
	for i in np.unique(df.Class):
		correct_classed = np.count_nonzero((df.predicted_class == df.Class) & (df.Class == i))	
		total = np.count_nonzero((df.predicted_class == i))
		if (total == 0):
			precision = 0
		else:
			precision = round(correct_classed / total, 3)
		list_metrics.append((i, i, "Precision", precision))

		false_classed = np.count_nonzero((df.predicted_class != df.Class) & (df.Class == i))
		recall = round(correct_classed / (correct_classed + false_classed), 3)
		list_metrics.append((i, i, "Recall", recall))

	return pd.DataFrame(list_metrics, columns = ["Class_int", "Class", "Metric", "Value"])

def f_calc_mean_metric_k_fold(df):
	print("Calculate the average of metrics for k-fold")
	avg_accuracy = df.loc[df['Metric'] == "Accuracy", 'Value'].mean()
	l_kf = list()
	l_kf.append(round(avg_accuracy,3))
	for i in np.unique(df.loc[(df['Class_int'] != -99)].Class):
		avg_p = df.loc[(df['Metric'] == "Precision") & (df['Class_int'] == i), 'Value'].mean()
		avg_r = df.loc[(df['Metric'] == "Recall") & (df['Class_int'] == i), 'Value'].mean()
		l_kf.append(round(avg_p, 3))
		l_kf.append(round(avg_r, 3))
	return l_kf

# Plotting function for reporting purpose
#def f_plotting_metrics(df, flag_incl_baseline_knn):
#
#	if flag_incl_baseline_knn:
#		plt.scatter(df[df["Class"] == "All"]["knn"], df[df["Class"] == "All"]["Value"],
#	         marker='o', label = "Accuracy")
#	else:
#		plt.plot(df[df["Class"] == "All"]["knn"], df[df["Class"] == "All"]["Value"], linewidth = 3,
#	         marker='o', label = "Accuracy")
#
#	i = 0
#	for x,y in zip(df[df["Class"] == "All"]["knn"], df[df["Class"] == "All"]["Value"]):
#		if i % 5 == 0:
#			plt.annotate("{:.3f}".format(y), (x,y), textcoords="offset points", xytext=(0,10), ha='center')
#		i = i + 1
#	if (not flag_incl_baseline_knn):
#		plt.ylim(0.9, 1)
#	plt.xlabel("knn")
#	plt.ylabel("Accuracy")
#	plt.legend()
#	plt.savefig('Accuracy_Holdout')
#	plt.close()
#
#	plt.plot(df[df["Class"] == "All"]["knn"], df[df["Class"] == "All"]["Value"], linewidth = 3,
#         marker='o', label = "Holdout")
#	plt.plot(df[df["Class"] == "All"]["knn"], df[df["Class"] == "All"]["kfold_avg"], linewidth = 3,
#	         marker='o', label = "Cross Validation")
#
	#i = 0
	#for x,y in zip(df[df["Class"] == "All"]["knn"], df[df["Class"] == "All"]["Value"]):
	#	if i % 5 == 0:
	#   		plt.annotate("{:.3f}".format(y), (x,y), textcoords="offset points", xytext=(0,-15), ha='center', color='blue')
	#	i = i + 1
#
	#i = 0
	#for x,y in zip(df[df["Class"] == "All"]["knn"], df[df["Class"] == "All"]["kfold_avg"]):
	#	if i % 5 == 0:
	#   		plt.annotate("{:.3f}".format(y), (x,y), textcoords="offset points", xytext=(0,15), ha='center', color='orange')
	#	i = i + 1

#	if (not flag_incl_baseline_knn):
#		plt.ylim(0.9, 1)
#	plt.xlabel("knn")
#	plt.ylabel("Accuracy")
#	plt.legend()
#	plt.savefig('Accuracy_HO_CV')
#	plt.close()
#
#	df_p = df[df["Metric"] == "Precision"]
#	for i in np.unique(df_p.Class_int):
#		tmp = df_p[df_p["Class_int"] == i]
#		plt.plot( tmp["knn"], tmp["Value"], linewidth = 3,
#	         marker='o', label = "Class " + str(i))
#		#plt.plot( tmp["knn"], tmp["kfold_avg"], linewidth = 3,
#	    #     marker='o', label = "Class " + str(i) + " (k-fold avg)")
#	plt.xlabel("knn")
#	plt.ylabel("Precision")
#	plt.legend()
#	plt.savefig('Precision')
#	plt.close()
#
#	df_rc = df[df["Metric"] == "Recall"]
#	for i in np.unique(df_rc.Class_int):
#		tmp = df_rc[df_rc["Class_int"] == i]
#		plt.plot(tmp["knn"], tmp["Value"], linewidth = 3,
#	         marker='o', label = "Class " + str(i))
#		#plt.plot( tmp["knn"], tmp["kfold_avg"], linewidth = 3,
#	    #     marker='o', label = "Class " + str(i) + " (k-fold avg)")
#	plt.xlabel("knn")
#	plt.ylabel("Recall")
#	plt.legend()
#	plt.savefig('Recall')
#	print("Complete exporting plottings")
# End function definition

# **********************************************
# Variable setting
folder_name = os.path.dirname(__file__) # + "/data/" 
data_separator = " "
k_fold = 5
l_knn = list(range(1, 4, 2))

flag_incl_baseline_knn  = False # to include the highest knn
flag_export_result = False
flag_export_plot = False

# input hard setting filename
if len(sys.argv) > 1:
	file_train =  sys.argv[1] # "wine-training"
if len(sys.argv) > 2:
	file_test =  sys.argv[2] # "wine-test" 

# input by command line
fn_train = folder_name + '/' + file_train
fn_test = folder_name+ '/' + file_test

start = time.time() # logging

df_train = pd.read_csv(fn_train, sep = data_separator)
df_test = pd.read_csv(fn_test, sep = data_separator)

df_metrics = pd.DataFrame()
if flag_incl_baseline_knn:
	l_knn.append(len(df_train))

# **********************************************
# Main Execution

for knn in l_knn:
	print('************************')
	print("knn = " + str(knn))
	df_classified = f_classification_with_normalization(df_train, df_test, knn) #f_classification_with_normalization
	metrics = f_calc_eval_metrics(df_classified)
	metrics["knn"]  = knn
	print("Accuracy = " + str(list(metrics.loc[(metrics['Metric'] == "Accuracy") & (metrics['knn'] == knn), 'Value'])))
	print('************************')
	df_metrics = pd.concat([df_metrics, metrics])

	if knn == 1:
		print("Report the predicted class labels of each instance in the test set with k = 1:")
		print(str(list(df_classified["predicted_class"])))

		# file output
		#print(file_test + "_with_predicted_class_labels_k_" + str(knn) + ".csv")
		#df_test_classified = df_test.copy()
		#df_test_classified["predicted_class"] = df_classified["predicted_class"]
		#df_test_classified.to_csv(file_test + "_with_predicted_class_labels_k_" + str(knn) + ".csv", sep='\t', encoding='utf-8')

# Run Cross validation for k = 3 and k-fold = 5
knn = 3

print('************************')
print(str(k_fold) + " K-fold CV - knn = " + str(knn))
metric_cv = (f_k_cross_validation(pd.concat([df_train,df_test]), k_fold, knn))
l_kf_metric = f_calc_mean_metric_k_fold(metric_cv)

print("K-Fold Accuracy = " + str(list(metric_cv.loc[(metric_cv['Metric'] == "Accuracy"), 'Value'])))
print("Average Accuracy = " + str(l_kf_metric[0]))
print('************************')


df_metrics["kfold_avg"] = 0

# For reporting only, these flags are defaulted by False
if (flag_export_result):
	df_metrics.to_csv("metrics.csv", sep='\t', encoding='utf-8')
#if (flag_export_plot):
#	f_plotting_metrics(df_metrics, flag_incl_baseline_knn)

# Endding
end = time.time()
print("The time of execution of above program is :", round((end-start), 1), "s")
