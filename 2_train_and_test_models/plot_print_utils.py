from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, confusion_matrix, log_loss
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

### This file contains some functions for plotting or printing model performance metrics.
# For the most part, these aren't part of the final workflow used in the paper, but I've left them here anyways for reference.


def plot_metrics(params, hist, cb = None):
	epochs = range(1, params.epochs + 1)

	plt.figure(figsize = (12,8))

	plt.subplot(221)
	plt.plot(epochs, hist.history["acc"], '.-', color = '#31E080', label = "Training Accuracy")
	if "val_acc" in hist.history.keys():
		plt.plot(epochs, hist.history["val_acc"], '.-', color = '#5042F4', label = "Validation Accuracy")
	plt.legend()

	plt.subplot(222)
	plt.plot(epochs, hist.history["loss"], '.-', color = '#31E080', label = "Training Loss")
	if "val_loss" in hist.history.keys():
		plt.plot(epochs, hist.history["val_loss"], '.-', color = '#5042F4', label = "Validation Loss")
	plt.legend()
	
	plt.subplot(223)
	plt.plot(epochs, cb.auprcs, 'k.-', label = "Validation auPRC")
	plt.legend()

	plt.savefig(params.figures_path + "_metrics", dpi = 300)


def print_confusion_matrix(y, probs, threshold = 0.5):
	npthresh = np.vectorize(lambda t: 1 if t >= threshold else 0)
	pred = npthresh(probs)
	conf_matrix = confusion_matrix(y, pred)
	print("Confusion Matrix at t = 0.5:\n", conf_matrix)
	print("Precision at t = 0.5:\t", conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[0][1]))
	print("Recall at t = 0.5:\t", conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[1][0]), "\n")


def plot_PRC(params, target_data = True):
	if target_data:
		labels = params.target_val_labels
		probs = params.target_val_probs
	else:
		labels = params.source_val_labels
		probs = params.source_val_probs

	precision, recall, _ = precision_recall_curve(labels, probs)
	
	plt.figure()
	plt.step(recall, precision, color='b', alpha=0.2, where='post')
	plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.05])

	plt.title("auPRC: " + str(average_precision_score(labels, probs)))

	if target_data:
		plt.savefig(params.figures_path + "_targetPRC", dpi = 300)
	else:
		plt.savefig(params.figures_path + "_sourcePRC", dpi = 300)


