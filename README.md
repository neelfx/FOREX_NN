
# Table of contents

- Purpose

- Synopsis

- Analysis

- Wayforward

# Purpose

Predict the closing price for the next one hour of a time-series dataset for the Pound Sterling / US Dollar (GBPUSD) foreign exchange rate. 
The price is a measure of a currency compared to a base currency. 

With GBPUSD this mean how many dollars £1 can buy.

Traders are buying and selling on the decentralized foreign exchange market 24 hours and 5 days per week. It is the largest market in the world with trillions of dollars being traded daily. Traders buy and sell the currencies based on market information in an attempt to make a profit when all sales are settled.


# Synopsis

The data is retrieved through an online platform called QuantConnect and is based on the historical data for GBPUSD from 19th March 2020 on the 1 hour resolution

# Feature Selection


## Trading indicators

Assumptions have been made about the criteria for investment. 

1 - The Relative Standard Deviation of the zipcode should be in the 0.6 quartile of the dataset.
2 - This will then be further filtered by highest ROI and then 10 zip codes will be modeled and then from this 5 will be selected based on the quality of the projection.

### Candlestick charts
 - Price is usually displayed so that the open, high, low and close of the time frame is visually represented a Japanese candlestick
 
![candles](images/chart.png)

### Bollinger Bands
 - Overlay of upper and lower bands from the bollinger indicator, this represents the upper and lower bounds of expected volatility.
 
![bol](images/bolbands.png)

### Exponential Moving Averages
 - Used for analysing the strength and speed of a trend in price.
 
![ema](images/ema50_200.png)



## LSTM model prediction

Using the LSTM neural network a RMSE of 0.00212 ( 21 pips)  was achieved using 2 hidden layers, the first with 50 nodes and the second with 20 nodes and over 10 epochs.


# Way forward

- incorporate additional novel datasets that through web scraping and premium APIs, alternative data relating to financial markets is expensive.
- Use multi output LSTMs to choose target close price and stop loss.
- connect to brokers to place trades.
