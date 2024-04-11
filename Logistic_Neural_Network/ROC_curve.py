'''A scoring function reported scores for 10 positive data points and 10 negative ones as follows:
No.	Label	Score
1	Positive	25
2	Positive	21
3	Positive	20
4	Positive	19
5	Positive	18
6	Positive	17
7	Positive	14
8	Positive	13
9	Positive	10
10	Positive	5
11	Negative	18
12	Negative	15
13	Negative	13
14	Negative	12
15	Negative	10
16	Negative	8
17	Negative	5
18	Negative	4
19	Negative	3
20	Negative	3
Compute FPR and TPR for each of the following score thresholds 3, 4, … , 25, 
and plot the ROC curve of the scoring function
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Data
labels = np.array(['Positive'] * 10 + ['Negative'] * 10)
scores = np.array([25, 21, 20, 19, 18, 17, 14, 13, 10, 5, 18, 15, 13, 12, 10, 8, 5, 4, 3, 3])

# Calculate FPR, TPR for different score thresholds
thresholds = range(3, 26)
fpr_list = []
tpr_list = []

for threshold in thresholds:
    predictions = scores >= threshold
    true_positives = np.sum((labels == 'Positive') & predictions)
    false_positives = np.sum((labels == 'Negative') & predictions)
    true_negatives = np.sum((labels == 'Negative') & ~predictions)
    false_negatives = np.sum((labels == 'Positive') & ~predictions)
    
    tpr = true_positives / (true_positives + false_negatives)
    fpr = false_positives / (false_positives + true_negatives)
    
    tpr_list.append(tpr)
    fpr_list.append(fpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr_list, tpr_list, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc(fpr_list, tpr_list))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show(block=True)
print('FPR:',tpr_list)
print('TPR:',fpr_list)