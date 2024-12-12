import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def main():
    st.title("Upload File")
    
    uploaded_file = st.file_uploader("", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        st.write("Select Target:")
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        target = st.selectbox("", numeric_columns, key='target_select')
        
        categorical_columns = df.select_dtypes(include=['object']).columns
        selected_cat = st.radio("", categorical_columns)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            avg_by_cat = df.groupby(selected_cat)[target].mean()
            ax1.bar(avg_by_cat.index, avg_by_cat.values, color='lightblue')
            ax1.set_title(f'Average {target} by {selected_cat}')
            ax1.set_xlabel(selected_cat)
            ax1.set_ylabel(f'{target} average')
            st.pyplot(fig1)
            plt.close()
        
        with col2:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            correlations = df.select_dtypes(include=['float64', 'int64']).corr()[target].abs()
            correlations = correlations[correlations.index != target]
            ax2.bar(correlations.index, correlations.values, color='blue')
            ax2.set_title(f'Correlation Strength of Numerical Variables with {target}')
            ax2.set_xlabel('Numerical Variables')
            ax2.set_ylabel('Correlation Strength (Absolute Value)')
            plt.xticks(rotation=45)
            st.pyplot(fig2)
            plt.close()
        
        st.write("Select features for training:")
        cols = st.columns(7)
        selected_features = []
        
        for idx, col in enumerate(df.columns):
            with cols[idx % 7]:
                if st.checkbox(col, key=f'feat_{col}'):
                    selected_features.append(col)
        
        if st.button("Train") and selected_features:
            X = df[selected_features]
            y = df[target]
            
            numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
            categorical_features = X.select_dtypes(include=['object']).columns
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features),
                    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
                ] if len(categorical_features) > 0 else [
                    ('num', StandardScaler(), numeric_features)
                ]
            )
            
            model = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42))
            ])
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            r2_score = model.score(X_test, y_test)
            
            st.write(f"The R2 score is: {r2_score:.2f}")
            st.session_state['model'] = model
            st.session_state['features'] = selected_features
        
        prediction_input = st.text_input("Enter values for prediction (comma-separated):")
        if st.button("Predict") and 'model' in st.session_state and prediction_input:
            try:
                values = [x.strip().lower() if isinstance(x, str) else x 
                         for x in prediction_input.split(",")]
                
                if len(values) != len(st.session_state['features']):
                    st.error("Please enter values for all selected features")
                else:
                    input_df = pd.DataFrame([values], columns=st.session_state['features'])
                    prediction = st.session_state['model'].predict(input_df)[0]
                    st.write(f"Predicted {target}: {prediction:.2f}")
            except Exception as e:
                st.error("Error in prediction. Please check input format.")

if __name__ == "__main__":
    main()