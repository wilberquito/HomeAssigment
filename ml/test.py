
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def test(model, X_test, y_test, model_name='nonameyet', parent_folder='results/'):

  y_pred = model.predict(X_test)

  precision_ = [round(precision_score(y_test, y_pred), 3)]
  recall_score_ = [round(recall_score(y_test, y_pred), 3)]
  f1_score_ = [round(f1_score(y_test, y_pred), 3)]
  accuracy_score_ = [round(accuracy_score(y_test, y_pred), 3)]

  data = {
    'precision': precision_,
    'recall_score': recall_score_,
    'f1_score': f1_score_,
    'accuracy_score': accuracy_score_
  }

  metrics = pd.DataFrame(data, index=[model_name])
  metrics.to_csv(parent_folder + model_name + '.csv', index=True)

  return metrics
