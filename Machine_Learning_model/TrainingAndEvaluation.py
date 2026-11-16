import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV

def train_evaluation(sc_x_train, y_train, sc_x_eval, y_eval, sc_x_test, y_test):

    #lr_model = LinearRegression()
    #lr_model.fit(sc_x_train,y_train)
    #y_eval_pred = lr_model.predict(sc_x_eval)
    #r2 = r2_score(y_eval,y_eval_pred)

    #ridge_model = RidgeCV(alphas=0.000000994)
    #ridge_model.fit(sc_x_train,y_train)
    #y_eval_ridge = ridge_model.predict(sc_x_eval)
    #r2 = r2_score(y_eval, y_eval_ridge)

    lasso_cv = LassoCV(eps=0.5,alphas=100,cv=5,max_iter=1000)
    lasso_cv.fit(sc_x_train,y_train)
    y_eval_lasso=lasso_cv.predict(sc_x_eval)
    r2 = r2_score(y_eval,y_eval_lasso)

    res = y_eval_lasso - y_eval
    sns.kdeplot(res)

    return lasso_cv



