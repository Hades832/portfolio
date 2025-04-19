import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import streamlit as st
import math
import pylab

# Initialize session state
if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = {}

# Sidebar inputs
with st.sidebar:
    st.header("Investment Parameters")
    default_stocks = 'AAPL,HMC,NVDA,TSLA,AMZN,GOOGL'
    stocks_input = st.text_input('Enter stock symbols (comma separated):', default_stocks)
    stocks = [s.strip().upper() for s in stocks_input.split(',') if s.strip()]
    
    start_date = st.date_input('Start date', datetime(2024, 4, 1))
    end_date = st.date_input('End date', datetime(2025, 4, 10))
    
    st.header("Model Parameters")
    n_estimators = st.slider('Number of Trees', 10, 200, 100)
    max_depth = st.slider('Max Depth', 1, 20, 10)

# Investment amounts and durations
st.header("Investment Details")
investment_info = {}
cols = st.columns(len(stocks))

for i, stock in enumerate(stocks):
    with cols[i]:
        st.subheader(stock)
        investment_info[stock] = {
            'amount': st.number_input(f'Amount (USD)', min_value=1.0, value=1000.0, key=f'amount_{stock}'),
            'duration': st.selectbox('Duration',
                                     ['1 Week', '1 Month', '3 Months', '6 Months', '1 Year'],
                                     key=f'duration_{stock}')
        }

# Duration mapping
duration_map = {
    '1 Week': 5,
    '1 Month': 21,
    '3 Months': 63,
    '6 Months': 126,
    '1 Year': 252
}

# Initialize dictionary to store the results
portfolio_results = {}

# Begin processing each stock
st.header("Predictions")
for stock in stocks:
    st.subheader(f"Predictions for {stock}")

    data = yf.download(stock, start=start_date, end=end_date)
    if data.empty:
        st.warning(f"No data found for {stock}.")
        continue

    num_days = duration_map[investment_info[stock]['duration']]
    
    initial_price = (data['High'].iloc[-1] + data['Low'].iloc[-1]) / 2
    shares = investment_info[stock]['amount'] / initial_price
    num_days = duration_map[investment_info[stock]['duration']]
    
    for day in range(num_days):
        # Low prediction
        X_low = data[['Open', 'Close', 'High', 'Volume']]
        y_low = data['Low']
        model_low = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=32)
        model_low.fit(X_low, y_low)
        latest_data = data.tail(1)[['Open', 'Close', 'High', 'Volume']]
        next_day_prediction_Low = model_low.predict(latest_data)[0]
        
        # High prediction
        X_high = data[['Open', 'Close', 'Low', 'Volume']]
        y_high = data['High']
        model_high = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=32)
        model_high.fit(X_high, y_high)
        latest_data = data.tail(1)[['Open', 'Close', 'Low', 'Volume']]
        next_day_prediction_High = model_high.predict(latest_data)[0]
        
        # Open prediction
        X_open = data[['High', 'Close', 'Low', 'Volume']]
        y_open = data['Open']
        model_open = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=32)
        model_open.fit(X_open, y_open)
        latest_data = data.tail(1)[['High', 'Close', 'Low', 'Volume']]
        next_day_prediction_Open = model_open.predict(latest_data)[0]
        
        # Close prediction
        X_close = data[['High', 'Open', 'Low', 'Volume']]
        y_close = data['Close']
        model_close = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=32)
        model_close.fit(X_close, y_close)
        latest_data = data.tail(1)[['High', 'Open', 'Low', 'Volume']]
        next_day_prediction_Close = model_close.predict(latest_data)[0]
        
        # Update data with predictions
        new_data = pd.DataFrame({
            ('Low', stock): [next_day_prediction_Low], 
            ('High', stock): [next_day_prediction_High], 
            ('Open', stock): [next_day_prediction_Open], 
            ('Close', stock): [next_day_prediction_Close], 
            ('Volume', stock): [data['Volume'].iloc[-1]]
        }, index=[data.index[-1] + pd.Timedelta(days=1)])

        data = pd.concat([data, new_data])

        final_high = data[('High', stock)].iloc[-1]
        final_low = data[('Low', stock)].iloc[-1]
        final_price_of_stock = (final_high + final_low) / 2

    # Calculate the final investment value for this stock
    final_value_of_investment = final_price_of_stock * shares
    portfolio_results[stock] = {
        'Investment Amount': investment_info[stock]['amount'],
        'Final Predicted Value': final_value_of_investment,
        'Shares Purchased': shares,
        'Final Price of Stock': final_price_of_stock
    }

    # Display the stock data for verification
    st.write(f"{stock} - Final predicted stock price: {final_price_of_stock}")
    st.write(f"{stock} - Final predicted value of investment: {final_value_of_investment}")

# Calculate total portfolio value
total_portfolio_value = sum(result['Final Predicted Value'] for result in portfolio_results.values())

# Display the summary of the portfolio
st.header("Portfolio Summary")
portfolio_df = pd.DataFrame(portfolio_results).T
portfolio_df['Gain/Loss'] = portfolio_df['Final Predicted Value'] - portfolio_df['Investment Amount']

# Display the portfolio table
st.write(portfolio_df)

