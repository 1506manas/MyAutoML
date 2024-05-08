import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor

def data_cleaning(df):
    imputer = SimpleImputer(strategy='mean')
    df_cleaned = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df_cleaned

def feature_scaling(df):
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_scaled

def handle_date_columns(df):
    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = pd.to_datetime(df[col])
        except ValueError:
            df[col] = pd.factorize(df[col])[0]
    return df

def train_classification_models(X_train, y_train):
    models = [
        RandomForestClassifier(),
        GradientBoostingClassifier(),
        SVC(),
        LogisticRegression(),
        KNeighborsClassifier()
    ]
    trained_models = []
    for model in models:
        model.fit(X_train, y_train)
        trained_models.append(model)
    return trained_models

def train_regression_models(X_train, y_train):
    models = [
        RandomForestRegressor(),
        GradientBoostingRegressor(),
        LinearRegression(),
        Ridge(),
        Lasso(),
        ElasticNet(),
        KNeighborsRegressor()
    ]
    trained_models = []
    for model in models:
        model.fit(X_train, y_train)
        trained_models.append(model)
    return trained_models

def evaluate_classification_models(models, X_test, y_test):
    f1_scores = []
    for model in models:
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        f1_scores.append(f1)
    return f1_scores

def evaluate_regression_models(models, X_test, y_test):
    mse_scores = []
    r2_scores = []
    for model in models:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mse_scores.append(mse)
        r2_scores.append(r2)
    return mse_scores, r2_scores

def main():
    st.title("AutoML with Streamlit")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            encoding = st.text_input("Enter encoding (e.g., 'utf-8', 'latin1', 'ISO-8859-1'):")
            try:
                df = pd.read_csv(uploaded_file, encoding=encoding)
            except Exception as e:
                st.error(f"Error reading file: {e}")
                return

        df = handle_date_columns(df)

        if st.button("Display first 10 rows"):
            st.write(df.head(10))

        if st.button("Display all features"):
            st.write(df.columns.tolist())

        if st.button("Display correlation"):
            st.write(df.corr())

        if st.button("Display Description"):
            st.write(df.describe())

        target_variable = st.text_input("Enter the name of the target variable:")

        model_type = st.radio("Select model type:", ("Classification", "Regression"))

        if st.button("Start training"):
            if target_variable not in df.columns:
                st.error(f"Target variable '{target_variable}' not found in the dataset.")
            else:
                label_encoder = LabelEncoder()
                for col in df.select_dtypes(include=['object']).columns:
                    df[col] = label_encoder.fit_transform(df[col])

                X = df.drop(columns=[target_variable])
                y = df[target_variable]

                X_cleaned = data_cleaning(X)

                X_scaled = feature_scaling(X_cleaned)

                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

                if model_type == "Classification":
                    trained_models = train_classification_models(X_train, y_train)
                    f1_scores = evaluate_classification_models(trained_models, X_test, y_test)
                    st.write("F1 Scores:", f1_scores)
                else:
                    trained_models = train_regression_models(X_train, y_train)
                    mse_scores, r2_scores = evaluate_regression_models(trained_models, X_test, y_test)
                    st.write("Mean Squared Error Scores:", mse_scores)
                    st.write("R^2 Scores:", r2_scores)

if __name__ == "__main__":
    main()
