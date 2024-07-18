import numpy as np

from util.evaluation import Evaluation
import torch
from sklearn.metrics import classification_report


"""
This file shows how to see the experimental results.
"""

if __name__ == '__main__':
    dataset_set = "plaid"
    f1_maro_list = []

    # for unknown_class in range(12):
    unknown_class = "10"
    unknown_class = str(unknown_class)

    data = torch.load(f'checkpoints/{dataset_set}/{unknown_class}/eval_openmax.pkl')    # softmax
    label = data.label
    predict = data.predict

    eval = Evaluation(predict, label)
    f1_maro_list.append(eval.f1_macro)
    # print('Accuracy:', f"%.3f" % eval.accuracy)
    # print('F1-measure:', f'{eval.f1_measure:.3f}')
    # print('F1-macro:', f'{eval.f1_macro:.3f}')
    # print('F1-macro (weighted):', f'{eval.f1_macro_weighted:.3f}')
    # print('precision:', f'{eval.precision:.3f}')
    # print('precision (macro):', f'{eval.precision_macro:.3f}')
    # print('precision (weighted):', f'{eval.precision_weighted:.3f}')
    # print('recall:', f'{eval.recall:.3f}')
    # print('recall (macro):', f'{eval.recall_macro:.3f}')
    # print('recall (weighted):', f'{eval.recall_weighted:.3f}')
    print(classification_report(label, predict, digits=3))

    # f1_plaid = np.load("results/cooll_pred.npy")
    # print(f1_plaid)
    # print(np.array(f1_maro_list))
