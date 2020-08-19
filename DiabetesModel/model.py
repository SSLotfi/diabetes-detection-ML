import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from joblib import dump, load


def RunModel():

    diabetes_data = pd.read_csv('../datasets_228_482_diabetes.csv')

    diabetes_data_copy = diabetes_data.copy(deep = True)
    diabetes_data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetes_data_copy[['Glucose',
    'BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

    diabetes_data_copy['Glucose'].fillna(diabetes_data_copy['Glucose'].mean(), inplace = True)
    diabetes_data_copy['BloodPressure'].fillna(diabetes_data_copy['BloodPressure'].mean(), inplace = True)
    diabetes_data_copy['SkinThickness'].fillna(diabetes_data_copy['SkinThickness'].median(), inplace = True)
    diabetes_data_copy['Insulin'].fillna(diabetes_data_copy['Insulin'].median(), inplace = True)
    diabetes_data_copy['BMI'].fillna(diabetes_data_copy['BMI'].median(), inplace = True)

    diabetes_data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetes_data_copy[['Glucose',
    'BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

    x = diabetes_data_copy.drop(["Outcome"], axis=1)
    y = diabetes_data_copy.Outcome

    X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=42, stratify=y)

    LRModel_ = LogisticRegression(solver='lbfgs', multi_class='multinomial',random_state=33)
    RFModel_ = RandomForestClassifier(n_estimators=100, criterion='gini',max_depth=5, random_state=33)
    KNNModel_ = KNeighborsClassifier(n_neighbors= 10, weights ='uniform', algorithm='auto')
    NNModel_ = MLPClassifier(solver='lbfgs',hidden_layer_sizes=(1000, 20),learning_rate='constant',activation='relu', power_t=0.4, max_iter=250)

    VotingClassifierModel = VotingClassifier(estimators=[('LRModel',LRModel_),('RFModel',RFModel_),('KNNModel',KNNModel_),('NNModel',NNModel_)], voting= 'soft')

    model_pipe = Pipeline([('standardscalar', StandardScaler()) , ('model', VotingClassifierModel)])

    model_pipe.fit(X_train, y_train)
    print('fitting done.....')

    print('VotingClassifierModel Train Score is : ' , model_pipe.score(X_train, y_train))
    print('VotingClassifierModel Test Score is : ' , model_pipe.score(X_test, y_test))
    print('----------------------------------------------------')

    dump(model_pipe, "diabetesdetectionmodel.pkl", True)
    print('dumping complete.....')

RunModel()