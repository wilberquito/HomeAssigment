import matplotlib.pyplot as plt
import sklearn.metrics as metrics


def confusion_matrix(y_test, y_pred, classes):
    cm = metrics.confusion_matrix(y_test, y_pred, labels=classes)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                          display_labels=classes)
    disp.plot()
    plt.show()
