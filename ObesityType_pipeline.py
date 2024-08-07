################################################
# End-to-End Obesity Type Machine Learning Pipeline
################################################


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.inspection import permutation_importance
from sklearn.tree import export_text
from sklearn.inspection import PartialDependenceDisplay
import joblib
import streamlit as st
import pandas as pd
import os

# Mevcut çalışma dizinini alın
current_dir = os.path.dirname(os.path.abspath(__name__))

# Veri setlerinin dosya yollarını belirleyin
obesity_data_path = os.path.join(current_dir, 'data', 'ObesityDataSet_raw_and_data_sinthetic.csv')

################################################
# Helper Functions
################################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


def create_bar_plot(dataframe, column, ax):
    bar_plot = sns.countplot(x=column, data=dataframe, ax=ax, palette="bright")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    # Add labels to the bars
    for p in bar_plot.patches:
        bar_plot.annotate(format(p.get_height(), '.0f'),
                          (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha='center', va='center',
                          xytext=(0, 10),
                          textcoords='offset points')

    ax.set_title('Distribution of Weight Categories')
    ax.set_xlabel('Weight Category')
    ax.set_ylabel('Count')


def create_pie_chart(dataframe, column, ax):
    pie_plot = dataframe[column].value_counts().plot.pie(autopct="%1.1f%%", ax=ax, colors=sns.color_palette("bright"))
    ax.set_ylabel('')  # Remove the default y-label


def visualize_data(dataframe, column):
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    create_bar_plot(dataframe, column, ax[0])
    create_pie_chart(dataframe, column, ax[1])

    plt.show()

def create_histogram(dataframe, figsize=(15, 15)):
    dataframe.hist(figsize=figsize)
    plt.show()

def create_density_plot(dataframe, layout=(6, 5), figsize=(15, 15)):
    dataframe.plot(kind="density", layout=layout, subplots=True, sharex=False, sharey=False, figsize=figsize)
    plt.show()


def plot_obesity_types(dataframe, obesity_levels, title, xlabel='Obesity Type', ylabel='Count'):
    # Filter the data by NObeyesdad and Gender
    obese_data = dataframe[dataframe['NOBEYESDAD'].isin(obesity_levels)]

    # Create the countplot
    sns.countplot(x='NOBEYESDAD', hue='GENDER', data=obese_data)
    plt.xticks(rotation=45)

    # Set the labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Display the plot
    plt.show()


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def encode_column(dataframe, column_name):
    le = LabelEncoder()
    dataframe[f"{column_name}_encoded"] = le.fit_transform(dataframe[column_name])
    print(dict(zip(le.classes_, le.transform(le.classes_))))
    return dataframe,le


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def obesity_data_prep(dataframe):
    dataframe.columns = [col.upper() for col in dataframe.columns]

    dataframe.drop_duplicates(inplace=True)

    # BMI
    dataframe['BMI'] = round(dataframe['WEIGHT'] / (dataframe['HEIGHT'] ** 2), 2)
    dataframe['AGE'] = dataframe['AGE'].astype(int)
    dataframe['WEIGHT'] = dataframe['WEIGHT'].round(decimals=2)
    dataframe['HEIGHT'] = dataframe['HEIGHT'].round(decimals=2)

    # Function to calculate BMR
    def calculate_bmr(row):
        if row['GENDER'] == 'Male':
            return 88.362 + (13.397 * row['WEIGHT']) + (4.799 * (row['HEIGHT'] * 100)) - (5.677 * row['AGE'])
        else:
            return 447.593 + (9.247 * row['WEIGHT']) + (3.098 * (row['HEIGHT'] * 100)) - (4.330 * row['AGE'])

    # Applying the function to create the BMR column
    dataframe['BMR'] = dataframe.apply(calculate_bmr, axis=1)


    # Ideal weight data with formulas
    ideal_weight_woman = {
        (13, 19): lambda height: 24 * (height ** 2),
        (20, 29): lambda height: 24 * (height ** 2),
        (30, 39): lambda height: 24.9 * (height ** 2),
        (40, 49): lambda height: 27 * (height ** 2),
        (50, 59): lambda height: 26.5 * (height ** 2),
        (60, 69): lambda height: 28 * (height ** 2),
        (70, 79): lambda height: 29 * (height ** 2)
    }

    ideal_weight_man = {
        (13, 19): lambda height: 24 * (height ** 2),
        (20, 29): lambda height: 24 * (height ** 2),
        (30, 39): lambda height: 24.9 * (height ** 2),
        (40, 49): lambda height: 26 * (height ** 2),
        (50, 59): lambda height: 26.5 * (height ** 2),
        (60, 69): lambda height: 28 * (height ** 2),
        (70, 79): lambda height: 29 * (height ** 2)
    }

    def get_ideal_weight(age, height, is_male):
        if not is_male:  # Female
            for interval, weight in ideal_weight_woman.items():
                if interval[0] <= age <= interval[1]:
                    if callable(weight):
                        return weight(height)
                    return weight
        else:  # Male
            for interval, weight in ideal_weight_man.items():
                if interval[0] <= age <= interval[1]:
                    if callable(weight):
                        return weight(height)
                    return weight
        return None

    # Assign ideal weight
    dataframe['IDEAL_WEIGHT'] = dataframe.apply(
        lambda row: get_ideal_weight(row['AGE'], row['HEIGHT'], row['GENDER'] == 'Male'), axis=1)


    dataframe['CH2O'] = dataframe['CH2O'].astype(int)
    dataframe['FAF'] = dataframe['FAF'].astype(int)
    dataframe['TUE'] = dataframe['TUE'].astype(int)

    # Function to calculate DCI based on FAF
    def calculate_dci(bmr, faf):
        if faf == 0:
            return 1.2 * bmr
        elif faf == 1:
            return 1.375 * bmr
        elif faf == 2:
            return 1.55 * bmr
        elif faf == 3:
            return 1.725 * bmr
        else:
            return bmr  # default case if FAF is not 0, 1, 2, or 3

    dataframe['DCI'] = dataframe.apply(
        lambda row: calculate_dci(row['BMR'], row['FAF']), axis=1)



    visualize_data(dataframe, 'NOBEYESDAD')
    create_histogram(dataframe, figsize=(15, 15))
    create_density_plot(dataframe, layout=(6, 5), figsize=(15, 15))

    overweight_levels = ['Overweight_Level_I', 'Overweight_Level_II']
    obesity_levels = ['Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']

    plot_obesity_types(dataframe, overweight_levels, 'Number of Females and Males with Overweight Types')
    plot_obesity_types(dataframe, obesity_levels, 'Number of Females and Males with Obesity Types')

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe, cat_th=3, car_th=20)

    cat_cols = [col for col in cat_cols if 'NOBEYESDAD' not in col]

    df = encode_column(dataframe,'NOBEYESDAD')

    df = one_hot_encoder(dataframe, cat_cols, drop_first=True)

    #cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)
    #cat_cols
    #num_cols

    X = df.drop(columns=["NOBEYESDAD", "NOBEYESDAD_encoded", "BMR", "IDEAL_WEIGHT" ,"DCI"], axis=1)
    y = df['NOBEYESDAD_encoded']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    return X, y, X_train, X_test, y_train, y_test

def train_and_evaluate_model(X_train, y_train, X_test, y_test, model_name):
    if model_name == 'decision_tree':
        pipeline = Pipeline([("scaler", MinMaxScaler()), ("dt", DecisionTreeClassifier())])
        params = {
            "dt__max_features": [0.2, 0.5, 0.7],
            "dt__min_samples_leaf": [1, 2, 3, 5, 10],
            "dt__max_depth": [5, 10, 15]
        }
    elif model_name == 'knn':
        pipeline = Pipeline([("scaler", MinMaxScaler()), ("knn", KNeighborsClassifier())])
        params = {
            "knn__n_neighbors": [3, 5, 7, 9, 11],
            "knn__metric": ['euclidean', 'manhattan'],
            "knn__weights": ['uniform', 'distance']
        }
    elif model_name == 'logistic_regression':
        pipeline = Pipeline([("scaler", MinMaxScaler()), ("lr", LogisticRegression(max_iter=10000))])
        params = {
            'lr__penalty': ['l2'],  # 'l1', 'elasticnet' removed because they may not work with all solvers
            'lr__C': [0.01, 0.1, 1, 10, 100],
            'lr__max_iter': [100, 1000, 10000],
            'lr__solver': ['lbfgs', 'saga']
        }
    else:
        raise ValueError("Unsupported model name")

    f1 = make_scorer(f1_score, average='macro')

    grid_search = GridSearchCV(pipeline, params, cv=5, verbose=True, refit=True, n_jobs=-1, scoring=f1)
    grid_search.fit(X_train, y_train)

    best_estimator = grid_search.best_estimator_
    best_score = grid_search.best_score_
    y_pred = best_estimator.predict(X_test)

    print(f"Best estimator for {model_name}:\n", best_estimator)
    print(f"Best score for {model_name}:", best_score)
    print(f"Classification report for {model_name}:\n", classification_report(y_test, y_pred))

    return best_estimator, best_score

def plot_confusion_matrix(y_test, y_pred, estimator, title='Confusion Matrix', cmap=plt.cm.autumn):
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=estimator.classes_)
    disp.plot(cmap=cmap)

    # Customize the plot
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.show()

    return plt
     # Return the matplotlib figure object

def fit_and_evaluate_models(X_train, y_train, X_test, y_test):
    pipelines = {
        'OvR_KNN': Pipeline([
            ("scaler", MinMaxScaler()),
            ("knn", OneVsRestClassifier(KNeighborsClassifier()))
        ]),
        'OvO_KNN': Pipeline([
            ("scaler", MinMaxScaler()),
            ("knn", OneVsOneClassifier(KNeighborsClassifier()))
        ]),
        'OvR_DT': Pipeline([
            ("scaler", MinMaxScaler()),
            ("dt", OneVsRestClassifier(DecisionTreeClassifier(random_state=42)))
        ]),
        'OvO_DT': Pipeline([
            ("scaler", MinMaxScaler()),
            ("dt", OneVsOneClassifier(DecisionTreeClassifier(random_state=42)))
        ]),
        'OvR_LR': Pipeline([
            ("scaler", MinMaxScaler()),
            ("lr", OneVsRestClassifier(LogisticRegression(penalty=None)))
        ]),
        'OvO_LR': Pipeline([
            ("scaler", MinMaxScaler()),
            ("lr", OneVsOneClassifier(LogisticRegression(penalty=None)))
        ])
    }

    param_grids = {
        'OvR_KNN': {
            'knn__estimator__n_neighbors': [3, 5, 7, 9, 11],
            'knn__estimator__metric': ['euclidean', 'manhattan'],
            'knn__estimator__weights': ['uniform', 'distance']
        },
        'OvO_KNN': {
            'knn__estimator__n_neighbors': [3, 5, 7, 9, 11],
            'knn__estimator__metric': ['euclidean', 'manhattan'],
            'knn__estimator__weights': ['uniform', 'distance']
        },
        'OvR_DT': {
            'dt__estimator__max_features': [0.2, 0.5, 0.7],
            'dt__estimator__min_samples_leaf': [1, 2, 3, 5, 10],
            'dt__estimator__max_depth': [5, 10, 15]
        },
        'OvO_DT': {
            'dt__estimator__max_features': [0.2, 0.5, 0.7],
            'dt__estimator__min_samples_leaf': [1, 2, 3, 5, 10],
            'dt__estimator__max_depth': [5, 10, 15]
        },
        'OvR_LR': {
            'lr__estimator__max_iter': [100, 1000, 100000, 10000],
            'lr__estimator__solver': ['lbfgs', 'saga']
        },
        'OvO_LR': {
            'lr__estimator__max_iter': [100, 1000, 100000, 10000],
            'lr__estimator__solver': ['lbfgs', 'saga']
        }
    }

    f1 = make_scorer(f1_score, average='macro')
    best_estimators = {}
    best_scores = {}

    for name, pipeline in pipelines.items():
        print(f"Training {name}...")
        grid_search = GridSearchCV(pipeline, param_grids[name], cv=5, verbose=True, refit=True, n_jobs=-1, scoring=f1)
        grid_search.fit(X_train, y_train)
        best_estimators[name] = grid_search.best_estimator_
        best_scores[name] = grid_search.best_score_

        y_pred = best_estimators[name].predict(X_test)
        print(f"Classification report for {name}:\n", classification_report(y_test, y_pred))
        print(f"Best score for {name}: {best_scores[name]}\n")

    for name, estimator in best_estimators.items():
        print(f"{name}: {estimator}")

    for name, score in best_scores.items():
        print(f"{name}: {score}")

    return best_estimators

def predict_and_plot_confusion_matrix(estimator_name, X_test, y_test, best_estimators):
    # Predict on test set
    y_pred = best_estimators[estimator_name].predict(X_test)

    # Generate and plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, best_estimators[estimator_name],
                          title=f'Confusion Matrix for {estimator_name}')



def plot_feature_importances(best_estimators, estimator_name, X_train):
    import numpy as np
    # Ensure the pipeline is fitted
    pipeline = best_estimators[estimator_name]

    # Extract the OneVsOne classifier from the pipeline
    classifier = pipeline.named_steps['dt']

    # Initialize an array to collect feature importances
    all_feature_importances = np.zeros(X_train.shape[1])

    # Loop through all individual decision trees in the OneVsOneClassifier
    for estimator in classifier.estimators_:
        # Each estimator should be a fitted DecisionTreeClassifier
        if hasattr(estimator, "feature_importances_"):
            all_feature_importances += estimator.feature_importances_

    # Average the feature importances
    mean_feature_importances = all_feature_importances / len(classifier.estimators_)

    # Get feature names
    feature_names = X_train.columns

    # Create a DataFrame for feature importances
    feature_importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': mean_feature_importances})

    # Sort the DataFrame by importance
    feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importances_df['Feature'], feature_importances_df['Importance'])
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importances from OneVsOne Decision Tree Classifiers")
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest importance on top
    plt.savefig('Feature_Importances_from_OneVsOne_Decision_Tree_Classifiers.png', bbox_inches='tight')
    plt.show()



def plot_permutation_importances(best_estimators, estimator_name, X_test, y_test, n_repeats=10, random_state=42):
    # Extract the pipeline for the specified estimator
    pipeline = best_estimators[estimator_name]

    # Perform permutation feature importance
    result = permutation_importance(pipeline, X_test, y_test, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)

    # Get feature names
    feature_names = X_test.columns

    # Create a DataFrame for permutation importances
    perm_importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': result.importances_mean})

    # Sort the DataFrame by importance
    perm_importances_df = perm_importances_df.sort_values(by='Importance', ascending=False)

    # Plot permutation importances
    plt.figure(figsize=(10, 6))
    plt.barh(perm_importances_df['Feature'], perm_importances_df['Importance'])
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Permutation Feature Importances")
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest importance on top
    # Save the plot with bbox_inches='tight'
    plt.savefig('permutation_importances.png', bbox_inches='tight')
    plt.show()




def plot_decision_path(ovo_pipeline, X_test, instance_index=0):
    """
    Extract, print, and visualize the decision path for a single instance
    using a decision tree from a OneVsOneClassifier within a pipeline.

    Parameters:
    - ovo_pipeline: The fitted OneVsOneClassifier pipeline.
    - X_test: The test dataset (DataFrame).
    - instance_index: The index of the instance to explain (default is 0).
    """
    # Extract a single decision tree from the OneVsOneClassifier
    dt_classifier = ovo_pipeline.named_steps['dt'].estimators_[0]

    # Get the decision path for the specified instance
    decision_path = dt_classifier.decision_path(X_test.iloc[[instance_index]])

    # Export the decision tree to text
    tree_rules = export_text(dt_classifier, feature_names=list(X_test.columns))

    # Display the decision path
    print(f"Decision path for instance {instance_index}:\n")
    print(tree_rules)

    # Visualize the decision path
    node_indicator = decision_path[0]
    leave_id = dt_classifier.apply(X_test.iloc[[instance_index]].values)

    # Obtain ids of the nodes involved in the decision path for the sample
    sample_id = 0
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:node_indicator.indptr[sample_id + 1]]

    plt.figure(figsize=(10, 6))
    for node_id in node_index:
        if leave_id[sample_id] == node_id:
            node_text = f"Leaf node {node_id}"
        else:
            node_text = f"Node {node_id}"
        feature = dt_classifier.tree_.feature[node_id]
        threshold = dt_classifier.tree_.threshold[node_id]
        plt.plot(node_id, feature, marker='o', markersize=5, label=node_text)
    plt.xlabel("Node ID")
    plt.ylabel("Feature Index")
    plt.title("Decision Path of a Single Prediction")
    plt.legend()
    plt.savefig('Decision_Path_Single_Prediction.png', bbox_inches='tight')
    plt.show()

    return decision_path


def plot_partial_dependence(ovo_pipeline, X_test, features, class_of_interest=0):
    """
    Generate and display partial dependence plots for a specified class and a list of features
    from a fitted OneVsOneClassifier pipeline.

    Parameters:
    - ovo_pipeline: The fitted OneVsOneClassifier pipeline.
    - X_test: The test dataset (DataFrame).
    - features: List of feature names to plot partial dependence for.
    - class_of_interest: The class for which to plot the partial dependence (default is 0).
    """
    # Generate partial dependence plots for the specified class
    PartialDependenceDisplay.from_estimator(ovo_pipeline, X_test, features=features, target=class_of_interest, feature_names=X_test.columns)

    plt.show()


################################################
# Pipeline Main Function
################################################

def main():
    df = pd.read_csv(obesity_data_path)
    X, y, X_train, X_test, y_train, y_test = obesity_data_prep(df)
    best_estimator_dt, best_score_dt = train_and_evaluate_model(X_train, y_train, X_test, y_test, 'decision_tree')
    best_estimator_knn, best_score_knn = train_and_evaluate_model(X_train, y_train, X_test, y_test, 'knn')
    best_estimator_lr, best_score_lr = train_and_evaluate_model(X_train, y_train, X_test, y_test, 'logistic_regression')

    # Example usage with decision tree
    y_pred_dt = best_estimator_dt.predict(X_test)
    cm_dt = plot_confusion_matrix(y_test, y_pred_dt, best_estimator_dt, title='Confusion Matrix for Decision Tree')

    # Example usage with KNN
    y_pred_knn = best_estimator_knn.predict(X_test)
    cm_knn = plot_confusion_matrix(y_test, y_pred_knn, best_estimator_knn, title='Confusion Matrix for KNN')

    # Example usage with logistic regression
    y_pred_lr = best_estimator_lr.predict(X_test)
    cm_lr = plot_confusion_matrix(y_test, y_pred_lr, best_estimator_lr, title='Confusion Matrix for Logistic Regression')

    # Example usage
    best_estimators = fit_and_evaluate_models(X_train, y_train, X_test, y_test)

    # # Example usage
    # predict_and_plot_confusion_matrix('OvO_DT', X_test, y_test, best_estimator_OvO_DT)

    # Example usage
    feature_importances_df = plot_feature_importances(best_estimators,'OvO_DT', X_train)

    # Example usage
    perm_importances_df = plot_permutation_importances(best_estimators, 'OvO_DT', X_test, y_test)

    # Example usage
    decision_path_plt = plot_decision_path(best_estimators['OvO_DT'], X_test, instance_index=0)

    #Example usage
    features_to_plot = ['BMI', 'HEIGHT', 'WEIGHT', 'FCVC']
    partial_dependence_plt = plot_partial_dependence(best_estimators['OvO_DT'], X_test, features_to_plot, class_of_interest=0)

    # Save the model
    joblib.dump(best_estimators['OvO_DT'], 'ovo_decision_tree_model.pkl')

    return {
        #(best_estimators)['OvO_DT']

    # "best_estimator_dt": best_estimator_dt,
    # "cm_dt": cm_dt,
    # "cm_knn": cm_knn,
    # "cm_lr": cm_lr,
    "feature_importances": feature_importances_df,
    "permutation_importances": perm_importances_df
    # "decision_path": decision_path
    # "partial_dependence": partial_dependence_plt
        }

if __name__ == "__main__":
    print("İşlem başladı")
    main()



