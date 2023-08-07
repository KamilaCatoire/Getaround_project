import argparse
import pandas as pd
import time
import mlflow
import matplotlib.pyplot as plt
from mlflow.models.signature import infer_signature
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor, plot_importance
from sklearn.tree import DecisionTreeRegressor
import os

def get_feature_imp(run_name, model, preprocessor, numeric_features, categorical_features, bool_features):
    # Feature importance
    feature_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
    feature_names += numeric_features
    feature_names += bool_features

    # Extract coefficient values
    coefs = model.named_steps['regressor'].coef_

    # Sort them and reverse (highest on top)
    inds = np.argsort(coefs)[::-1]
    sorted_coefs = coefs[inds]
    sorted_features = np.array(feature_names)[inds]

    fig = plt.figure(figsize=(10, 5))
    plt.bar(sorted_features, sorted_coefs)
    plt.title('Feature importances')
    plt.xlabel('Features')
    plt.ylabel('Coefficient')
    plt.xticks(rotation=90)
    fig.savefig(f'images/feat_imp_{run_name}', bbox_inches='tight')


# if __name__ == "__main__":

# Set your variables for your environment
EXPERIMENT_NAME="experiment-mlflow-getaround"

# Set tracking URI to your Heroku application
os.environ["APP_URI"]="https://mymlflow-getaround-kc-36d5c8058f66.herokuapp.com/"
mlflow.set_tracking_uri(os.environ["APP_URI"])

# Set experiment's info 
mlflow.set_experiment(EXPERIMENT_NAME)

# Get our experiment info
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

# # Setting experiment
# experiment_name = "GetAround_car_rental_price_predictor"
# client = mlflow.tracking.MlflowClient()
# if (mlflow.get_experiment(0).name == "Default") & (mlflow.get_experiment_by_name(experiment_name) is None): # rename and log in the "Default" experiment if existing
#     client.rename_experiment(0, experiment_name)
# mlflow.set_experiment(experiment_name)
# experiment = mlflow.get_experiment_by_name(experiment_name)

print("training model...")

# Time execution
start_time = time.time()

# Call mlflow autolog
mlflow.sklearn.autolog()

# Parse arguments given in shell script 'run.sh' or in Command-Line
# parser = argparse.ArgumentParser()
# parser.add_argument("--regressor", default = 'LR', choices = ['LR', 'Ridge', 'RF'])
# parser.add_argument("--cv", type = int, default = None)
# parser.add_argument("--alpha", type = float, nargs = "*")
# parser.add_argument("--max_depth", type = int, nargs="*")
# parser.add_argument("--min_samples_leaf", type = int, nargs="*")
# parser.add_argument("--min_samples_split", type = int, nargs="*")
# parser.add_argument("--n_estimators", type = int, nargs="*")
# args = parser.parse_args()

# Import dataset
df = pd.read_csv("get_around_pricing_project.csv", index_col = 0)

# Drop irrelevant rows
df = df[(df['mileage'] >= 0) & (df['engine_power'] > 0)]

# X, y split 
target_col = 'rental_price_per_day'
Y = df[target_col]
X = df.drop(target_col, axis = 1)

# Features categorization
numerical_features = []
binary_features = []
categorical_features = []
for i,t in X.dtypes.items():
    if ('float' in str(t)) or ('int' in str(t)) :
        numerical_features.append(i)
    elif ('bool' in str(t)):
        binary_features.append(i)
    else :
        categorical_features.append(i)

# Regroup fewly-populated category labels in label 'other'
for feature in categorical_features:
    label_counts = X[feature].value_counts()
    fewly_populated_labels = list(label_counts[label_counts < 0.5 / 100 * len(X)].index)
    for label in fewly_populated_labels:
        X.loc[X[feature]==label,feature] = 'other'

# Train / test split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, shuffle=True)

# Features preprocessing 
categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
numerical_transformer = StandardScaler()
binary_transformer = FunctionTransformer(None, feature_names_out = 'one-to-one') #identity function
preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numerical_transformer, numerical_features),
            ("bin", binary_transformer, binary_features)
        ]
    )
mlflow.sklearn.autolog(log_post_training_metrics=False, silent=True)
    # # Model definition
    # grid_search_done = False
    # if args.regressor == 'LR':
    #     model = LinearRegression()
    # else: # If a model can have hyperparameters to be tuned, allow a GridSearch with cross-validation
    #     regressor_args = {option : parameters for option, parameters in vars(args).items() if (parameters is not None and option not in ['cv', 'regressor'])}
    #     regressor_params = {param_name : values for param_name, values in regressor_args.items()}
    #     if args.regressor == 'Ridge':
    #         regressor = Ridge()
    #     elif args.regressor == 'RF':
    #         regressor = RandomForestRegressor()
    #     model = GridSearchCV(regressor, param_grid = regressor_params, cv = args.cv, verbose = 3)
    #     grid_search_done = True

    # # Pipeline 
    # predictor = Pipeline(steps=[
    #     ('features_preprocessing', feature_preprocessor),
    #     ("model", model)
    # ])

    # Log experiment to MLFlow
print("Linear Regression Training ...")
run_name = 'linear_regression_baseline'
with mlflow.start_run(run_name=run_name, ) as run:
    model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])
    model.fit(X_train, Y_train)
    print("Training done.")
    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)
    mlflow.log_metric("training_r2_score",r2_score(Y_train, Y_train_pred))
    mlflow.log_metric("training_mean_absolute_error",mean_absolute_error(Y_train, Y_train_pred))
    mlflow.log_metric("training_mean_squared_error",mean_squared_error(Y_train, Y_train_pred))
    mlflow.log_metric("training_root_mean_squared_error",mean_squared_error(Y_train, Y_train_pred, squared=False))
    mlflow.log_metric("testing_r2_score",r2_score(Y_test, Y_test_pred))
    mlflow.log_metric("testing_mean_absolute_error",mean_absolute_error(Y_test, Y_test_pred))
    mlflow.log_metric("testing_mean_squared_error",mean_squared_error(Y_test, Y_test_pred))
    mlflow.log_metric("testing_root_mean_squared_error",mean_squared_error(Y_test, Y_test_pred, squared=False))
    try:
        get_feature_imp(run_name, model, preprocessor, numerical_features, categorical_features, binary_features)
        mlflow.log_artifact(f'images/feat_imp_{run_name}.png')
    except:
        pass
    mlflow.end_run()

print("Decision Tree Training ...")

run_name = 'decision_tree'
with mlflow.start_run(run_name=run_name) as run:
    model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', DecisionTreeRegressor(max_depth=10, random_state=42))])
    model.fit(X_train, Y_train)
    
    print("Training done.")
    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)
    mlflow.log_metric("training_r2_score",r2_score(Y_train, Y_train_pred))
    mlflow.log_metric("training_mean_absolute_error",mean_absolute_error(Y_train, Y_train_pred))
    mlflow.log_metric("training_mean_squared_error",mean_squared_error(Y_train, Y_train_pred))
    mlflow.log_metric("training_root_mean_squared_error",mean_squared_error(Y_train, Y_train_pred, squared=False))
    mlflow.log_metric("testing_r2_score",r2_score(Y_test, Y_test_pred))
    mlflow.log_metric("testing_mean_absolute_error",mean_absolute_error(Y_test, Y_test_pred))
    mlflow.log_metric("testing_mean_squared_error",mean_squared_error(Y_test, Y_test_pred))
    mlflow.log_metric("testing_root_mean_squared_error",mean_squared_error(Y_test, Y_test_pred, squared=False))
    try:
        get_feature_imp(run_name, model, preprocessor, numerical_features, categorical_features, binary_features)
        mlflow.log_artifact(f'images/feat_imp_{run_name}.png')
    except:
        pass
    mlflow.end_run()

print("Random Forest Training ...")
run_name = 'random_forest'
with mlflow.start_run(run_name=run_name) as run:
    model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', RandomForestRegressor(max_depth=10, random_state=42))])
    model.fit(X_train, Y_train)
    
    print("Training done.")
    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)
    mlflow.log_metric("training_r2_score",r2_score(Y_train, Y_train_pred))
    mlflow.log_metric("training_mean_absolute_error",mean_absolute_error(Y_train, Y_train_pred))
    mlflow.log_metric("training_mean_squared_error",mean_squared_error(Y_train, Y_train_pred))
    mlflow.log_metric("training_root_mean_squared_error",mean_squared_error(Y_train, Y_train_pred, squared=False))
    mlflow.log_metric("testing_r2_score",r2_score(Y_test, Y_test_pred))
    mlflow.log_metric("testing_mean_absolute_error",mean_absolute_error(Y_test, Y_test_pred))
    mlflow.log_metric("testing_mean_squared_error",mean_squared_error(Y_test, Y_test_pred))
    mlflow.log_metric("testing_root_mean_squared_error",mean_squared_error(Y_test, Y_test_pred, squared=False))
    try:
        get_feature_imp(run_name, model, preprocessor, numerical_features, categorical_features, binary_features)
        mlflow.log_artifact(f'images/feat_imp_{run_name}.png')
    except:
        pass
    mlflow.end_run()

print("Ridge Training ...")
run_name = 'ridge'
with mlflow.start_run(run_name=run_name, ) as run:
    model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', Ridge(alpha=1, random_state=42))])
    model.fit(X_train, Y_train)
    print("Training done.")
    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)
    mlflow.log_metric("training_r2_score",r2_score(Y_train, Y_train_pred))
    mlflow.log_metric("training_mean_absolute_error",mean_absolute_error(Y_train, Y_train_pred))
    mlflow.log_metric("training_mean_squared_error",mean_squared_error(Y_train, Y_train_pred))
    mlflow.log_metric("training_root_mean_squared_error",mean_squared_error(Y_train, Y_train_pred, squared=False))
    mlflow.log_metric("testing_r2_score",r2_score(Y_test, Y_test_pred))
    mlflow.log_metric("testing_mean_absolute_error",mean_absolute_error(Y_test, Y_test_pred))
    mlflow.log_metric("testing_mean_squared_error",mean_squared_error(Y_test, Y_test_pred))
    mlflow.log_metric("testing_root_mean_squared_error",mean_squared_error(Y_test, Y_test_pred, squared=False))
    try:
        get_feature_imp(run_name, model, preprocessor, numerical_features, categorical_features, binary_features)
        mlflow.log_artifact(f'images/feat_imp_{run_name}.png')
    except:
        pass
    mlflow.end_run()

print("Lasso Training ...")
run_name = 'lasso'
with mlflow.start_run(run_name=run_name, ) as run:
    model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', Lasso(alpha=1, random_state=42))])
    model.fit(X_train, Y_train)
    print("Training done.")
    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)
    mlflow.log_metric("training_r2_score",r2_score(Y_train, Y_train_pred))
    mlflow.log_metric("training_mean_absolute_error",mean_absolute_error(Y_train, Y_train_pred))
    mlflow.log_metric("training_mean_squared_error",mean_squared_error(Y_train, Y_train_pred))
    mlflow.log_metric("training_root_mean_squared_error",mean_squared_error(Y_train, Y_train_pred, squared=False))
    mlflow.log_metric("testing_r2_score",r2_score(Y_test, Y_test_pred))
    mlflow.log_metric("testing_mean_absolute_error",mean_absolute_error(Y_test, Y_test_pred))
    mlflow.log_metric("testing_mean_squared_error",mean_squared_error(Y_test, Y_test_pred))
    mlflow.log_metric("testing_root_mean_squared_error",mean_squared_error(Y_test, Y_test_pred, squared=False))
    try:
        get_feature_imp(run_name, model, preprocessor, numerical_features, categorical_features, binary_features)
        mlflow.log_artifact(f'images/feat_imp_{run_name}.png')
    except:
        pass
    mlflow.end_run()

print("RandomForest with feature selection Training ...")
run_name = 'random_forest_feature_selection_40'
with mlflow.start_run(run_name=run_name) as run:
    model = Pipeline(steps=[('preprocessor', preprocessor), ('feature_selector', SelectKBest(f_regression, k=40)), ('regressor', RandomForestRegressor(max_depth=10, random_state=42))])
    model.fit(X_train, Y_train)
    print("Training done.")
    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)
    mlflow.log_metric("training_r2_score",r2_score(Y_train, Y_train_pred))
    mlflow.log_metric("training_mean_absolute_error",mean_absolute_error(Y_train, Y_train_pred))
    mlflow.log_metric("training_mean_squared_error",mean_squared_error(Y_train, Y_train_pred))
    mlflow.log_metric("training_root_mean_squared_error",mean_squared_error(Y_train, Y_train_pred, squared=False))
    mlflow.log_metric("testing_r2_score",r2_score(Y_test, Y_test_pred))
    mlflow.log_metric("testing_mean_absolute_error",mean_absolute_error(Y_test, Y_test_pred))
    mlflow.log_metric("testing_mean_squared_error",mean_squared_error(Y_test, Y_test_pred))
    mlflow.log_metric("testing_root_mean_squared_error",mean_squared_error(Y_test, Y_test_pred, squared=False))
    try:
        get_feature_imp(run_name, model, preprocessor, numerical_features, categorical_features, binary_features)
        mlflow.log_artifact(f'images/feat_imp_{run_name}.png')
    except:
        pass
    mlflow.end_run()

print("GridSearchCV RandomForest Training ...")
# Grid of values to be tested
run_name = 'random_forest_gridsearch'
with mlflow.start_run(run_name=run_name, ) as run:
    params = {
    'max_depth': [10, 12, 14, 16, 18, 20],
    'min_samples_split': [2, 4, 8, 10, 12, 14, 16],
    'n_estimators': [60, 80, 100, 200, 300, 400, 500]
    }
    model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', GridSearchCV(RandomForestRegressor(random_state=42), params, cv=5, n_jobs=-1))])
    model.fit(X_train, Y_train)
    print("Training done.")
    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)
    mlflow.log_metric("training_r2_score",r2_score(Y_train, Y_train_pred))
    mlflow.log_metric("training_mean_absolute_error",mean_absolute_error(Y_train, Y_train_pred))
    mlflow.log_metric("training_mean_squared_error",mean_squared_error(Y_train, Y_train_pred))
    mlflow.log_metric("training_root_mean_squared_error",mean_squared_error(Y_train, Y_train_pred, squared=False))
    mlflow.log_metric("testing_r2_score",r2_score(Y_test, Y_test_pred))
    mlflow.log_metric("testing_mean_absolute_error",mean_absolute_error(Y_test, Y_test_pred))
    mlflow.log_metric("testing_mean_squared_error",mean_squared_error(Y_test, Y_test_pred))
    mlflow.log_metric("testing_root_mean_squared_error",mean_squared_error(Y_test, Y_test_pred, squared=False))
    mlflow.log_param("best_params", model.named_steps['regressor'].best_params_)
    try:
        get_feature_imp(run_name, model, preprocessor, numerical_features, categorical_features, binary_features)
        mlflow.log_artifact(f'images/feat_imp_{run_name}.png')
    except:
        pass
    mlflow.end_run()

print("XGBRegressor Training ...")
run_name = 'xgbr'
with mlflow.start_run(run_name=run_name) as run:
    model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', XGBRegressor(n_estimators=200, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8, alpha=0.1, random_state=42))])
    model.fit(X_train, Y_train)
    
    print("Training done.")
    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)
    mlflow.log_metric("training_r2_score",r2_score(Y_train, Y_train_pred))
    mlflow.log_metric("training_mean_absolute_error",mean_absolute_error(Y_train, Y_train_pred))
    mlflow.log_metric("training_mean_squared_error",mean_squared_error(Y_train, Y_train_pred))
    mlflow.log_metric("training_root_mean_squared_error",mean_squared_error(Y_train, Y_train_pred, squared=False))
    mlflow.log_metric("testing_r2_score",r2_score(Y_test, Y_test_pred))
    mlflow.log_metric("testing_mean_absolute_error",mean_absolute_error(Y_test, Y_test_pred))
    mlflow.log_metric("testing_mean_squared_error",mean_squared_error(Y_test, Y_test_pred))
    mlflow.log_metric("testing_root_mean_squared_error",mean_squared_error(Y_test, Y_test_pred, squared=False))
    try:
        get_feature_imp(run_name, model, preprocessor, numerical_features, categorical_features, binary_features)
        mlflow.log_artifact(f'images/feat_imp_{run_name}.png')
    except:
        pass
    mlflow.end_run()

print("All training is done!")
print(f"---Total training time: {time.time()-start_time}")