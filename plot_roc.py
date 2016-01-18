import numpy as np
import matplotlib.pyplot as plt
import os, sys
from emails import get_emails
from sklearn.metrics import roc_curve, auc

def get_result(mail, results_dir="training/keras/models/lstm2layer2dropout/results"):
    filename = os.path.join(results_dir, mail + ".txt")
    with open(filename) as f:
        resultset = [d.split(',') for d in f.read().splitlines()]
        y_score = resultset[0]
        y = resultset[1]
        y_score = [float(x) for x in y_score]
        y = [int(x) for x in y]
        return y_score, y

if __name__ == '__main__':
    emails = get_emails()

    for mail in emails:
        if len(sys.argv) > 1:
            y_score, y = get_result(mail, sys.argv[1])
        else:
            y_score, y = get_result(mail)

        # Compute micro-average ROC curve and ROC area
        fpr, tpr, _ = roc_curve(y, y_score, pos_label=1)
        roc_auc = auc(fpr, tpr)

        ##############################################################################
        # Plot of a ROC curve for a specific class
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve for %s' % mail)
        plt.legend(loc="lower right")
        plt.show()
