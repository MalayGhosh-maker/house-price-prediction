import seaborn as sns
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from sklearn.linear_model import LassoCV

def train_evaluation(x_train, y_train, x_eval, y_eval, x_test, y_test):

    lasso_model = LassoCV(eps=0.0001,alphas=10,cv=10,max_iter=1000)
    lasso_model.fit(x_train,y_train)
    y_hat_lasso = lasso_model.predict(x_eval)

    MAE_lasso = mean_absolute_error(y_eval,y_hat_lasso)
    MSE_lasso = mean_squared_error(y_eval,y_hat_lasso)
    r2_lasso = r2_score(y_eval,y_hat_lasso)
    RMSE_lasso = np.sqrt(MSE_lasso)
    r2_lasso

    return lasso_model



