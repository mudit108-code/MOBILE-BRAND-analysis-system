import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Configure Streamlit page
st.set_page_config(page_title="Mobile Phone Sales Analysis", layout="wide")

# Load CSV file
st.sidebar.header("Upload your CSV file")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.sidebar.write("File uploaded successfully!")
    st.write("### Preview of the dataset", data.head())

    # Data Filtering Functionality
    st.header("Data Filtering")
    company = st.selectbox("Select a Manufacturer", options=data['Manufacturer'].unique())
    year = st.selectbox("Select a Year", options=sorted(data['Year'].unique()))

    if st.button("Filter Data"):
        filtered_data = data[(data['Manufacturer'] == company) & (data['Year'] == year)]
        if not filtered_data.empty:
            st.write("Filtered Data", filtered_data)
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Model', y='Units Sold (million )', data=filtered_data)
            plt.title(f'Units Sold by {company} in {year}')
            plt.xlabel('Model')
            plt.ylabel('Units Sold (million)')
            plt.xticks(rotation=45)
            st.pyplot(plt)
        else:
            st.write("No data available for the selected company and year.")

    # Predictive Analytics Functionality
    st.header("Predictive Analytics")
    future_years_count = st.slider("Select number of future years to predict", 1, 10, 5)
    
    def train_predictive_model(df, company, future_years_count):
        company_data = df[df['Manufacturer'] == company]
        company_data = company_data[['Year', 'Units Sold (million )']].dropna()
        
        if len(company_data) < 2:
            st.write(f"Not enough data available for future trend prediction for {company}.")
            return

        X = company_data['Year'].values.reshape(-1, 1)
        y = company_data['Units Sold (million )'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        st.write(f'Root Mean Squared Error for {company}: {rmse:.2f}')

        future_years = np.arange(X.max() + 1, X.max() + future_years_count + 1).reshape(-1, 1)
        future_predictions = model.predict(future_years)

        plt.figure(figsize=(12, 6))
        plt.scatter(X_test, y_test, color='blue', label='Actual')
        plt.plot(X_test, y_pred, color='red', label='Predicted')
        plt.plot(future_years, future_predictions, color='green', linestyle='dashed', label='Future Trend')
        plt.xlabel('Year')
        plt.ylabel('Units Sold (million)')
        plt.title(f'Actual vs Predicted Sales for {company} and Future Trend')
        plt.legend()
        plt.grid()
        st.pyplot(plt)

        future_df = pd.DataFrame({'Year': future_years.flatten(), 'Predicted Units Sold (million)': future_predictions})
        st.write(future_df)

    # Interactive function call for prediction
    if st.button("Train Predictive Model"):
        train_predictive_model(data, company, future_years_count)

    # Comparison with Best Brand
    st.header("Comparison with Best-Selling Brand")

    def compare_with_best_brand(df, selected_brand):
        total_sales_per_brand = df.groupby('Manufacturer')['Units Sold (million )'].sum()
        best_brand = total_sales_per_brand.idxmax()

        if selected_brand == best_brand:
            st.write(f"{selected_brand} is already the best-selling brand!")
            return

        comparison_data = df[(df['Manufacturer'].isin([selected_brand, best_brand]))]
        comparison_summary = comparison_data.pivot_table(index='Year', columns='Manufacturer', values='Units Sold (million )', aggfunc='sum', fill_value=0)

        plt.figure(figsize=(12, 6))
        comparison_summary.plot(kind='bar', stacked=True)
        plt.title(f'Units Sold Comparison: {selected_brand} vs {best_brand}')
        plt.xlabel('Year')
        plt.ylabel('Units Sold (million)')
        plt.xticks(rotation=45)
        plt.legend(title='Manufacturer')
        plt.grid()
        st.pyplot(plt)

    # Button to show comparison
    if st.button("Compare with Best Brand"):
        compare_with_best_brand(data, company)

else:
    st.sidebar.write("Awaiting CSV file upload...")

