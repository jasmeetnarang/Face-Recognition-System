
import matplotlib.pyplot as plt
import numpy as np
from GenModel import network, pca_test_x,cat_test_y
from sklearn.metrics import roc_curve,auc
from itertools import cycle
from keras.models import load_model

#load the model
loaded_model = load_model("models/model9.h5")
result = loaded_model.evaluate(pca_test_x, cat_test_y)

predictions = network.predict(pca_test_x)
# y_pred = np.argmax(predictions, axis = 1)

print("Predictions: ", predictions)

# get authentic and imposter distribution using threshold values
p = 0.06

genuine = []
imposter = []
for i in range(0, predictions.shape[0]):
    for j in range(0, predictions.shape[1]):
        if predictions[i][j] > p:
            genuine.append(predictions[i][j])
        else:
            imposter.append(predictions[i][j])
print("authentic: ",len(genuine))
print("imposter: ",len(imposter))

plt.hist(imposter, facecolor='g', alpha=0.50, label='Imposter')
plt.hist(genuine, facecolor='y', alpha=0.50, label='Genuine')

# Adding labels and titles to the plot
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.title('Imposter vs Genuine Distribution')
plt.grid(True)
# draw the key/legend
plt.legend()
# show the plot
plt.show()


# print(y_pred)
n_classes = 26

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(cat_test_y[:,i], predictions[:,i])
    roc_auc[i] = auc(fpr[i], tpr[i])

lw = 2
colors = cycle(['aqua', 'darkorange', 'cornflowerblue','red','blue', 'green',
                'yellow'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw)

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
# plt.legend(loc="lower right")
plt.show()

# get the CMC curve
tpr_list = []
ranks = list(range(1, 16))

for k in range(len(tpr)):
    tpr_list.append(np.mean(tpr[k]))

tpr_list.sort(reverse=True)
True_positive_rates = tpr_list[:15]
print("True positive rate: ",len(True_positive_rates))
print("ranks: ", len(ranks))
plt.plot(ranks, tpr_list[:15])
plt.xlabel('Ranks')
plt.ylabel('True_positive_rates')
plt.show()
