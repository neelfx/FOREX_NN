import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from numpy import hstack
import pickle
import datetime

class LSTM_Algo(QCAlgorithm):

    def Initialize(self):
        self.SetCash(100000)
        self.SetStartDate(2020, 4, 3)
        self.SetEndDate(2020, 10, 20)
        
        #1. Request the forex data
        self.gbpusd = self.AddForex("GBPUSD", Resolution.Hour, Market.Oanda)
        
        #2. Set the brokerage model
        self.SetBrokerageModel(BrokerageName.OandaBrokerage)
        
        self.scaler = MinMaxScaler(feature_range = (0,1))
        
        self.bband = self.BB("GBPUSD", 10, 2, MovingAverageType.Exponential, Resolution.Hour)
        self.stoc = self.STO("GBPUSD", 14, 1, 5, Resolution.Hour)
        self.ema8 = self.EMA("GBPUSD", 8, MovingAverageType.Exponential, Resolution.Hour)
        self.ema21 = self.EMA("GBPUSD", 21, MovingAverageType.Exponential, Resolution.Hour)
        self.ema50 = self.EMA("GBPUSD", 50, MovingAverageType.Exponential, Resolution.Hour)
        self.ema200 = self.EMA("GBPUSD", 200, MovingAverageType.Exponential, Resolution.Hour)
        
        #custom rollingwindow
        self.df = pd.DataFrame(columns=['time', 'ema8', 'ema21', 'ema50', 'ema200', 'upperband', 'lowerband', 'middleband', 'stochastic', 'open', 'high', 'low'])
        #self.df['time'] = pd.to_datetime(self.df['time'])
        
        #self.df['time'] = datetime.date.today()
        
        self.df.set_index('time')
        
        deserialized=bytes(self.ObjectStore.ReadBytes("LSTM"))
        self.model=pickle.loads(deserialized)
        self.SetWarmUp(200)
    
    
    def AddToWindow(self, data):
        # index = None
        # if len(self.df) == 0:
        #     self.df['time'] = data['GBPUSD'].Time
        #     index = data['GBPUSD'].Time
            
        # else:
        
        new_candle = pd.DataFrame(columns=['time', 'ema8', 'ema21', 'ema50', 'ema200', 'upperband', 'lowerband', 'middleband', 'stochastic', 'open', 'high', 'low'])
        new_candle['time'] = data['GBPUSD'].Time
        new_candle['open'] = data['GBPUSD'].Open
        new_candle['high'] = data['GBPUSD'].High
        new_candle['low'] = data['GBPUSD'].Low
        new_candle['ema8'] = self.ema8.Current.Value
        new_candle['ema21'] = self.ema21.Current.Value
        new_candle['ema50'] = self.ema50.Current.Value
        new_candle['ema200'] = self.ema200.Current.Value
        new_candle['upperband'] = self.bband.UpperBand.Current.Value
        new_candle['lowerband'] = self.bband.LowerBand.Current.Value
        new_candle['middleband'] = self.bband.MiddleBand.Current.Value
        new_candle['stochastic'] = self.stoc.StochK.Current.Value
        new_candle.set_index('time')
        self.df = pd.concat([self.df, new_candle])
        self.df = self.df.tail(6)
     
    def PredictPrice(self):

        ema_8_test = self.df['ema8']
        ema_21_test = self.df['ema21']
        ema_50_test = self.df['ema50']
        ema_200_test = self.df['ema200']
        
        b_upper_test = self.df['upperband']
        b_lower_test = self.df['lowerband']
        b_mid_test = self.df['middleband']
        
        stoch_test = self.df['stochastic']
        
        open_test = self.df['open']
        high_test = self.df['high']
        low_test = self.df['low']
        
        t_ema_50_array = np.array(ema_50_test).reshape((len(ema_50_test), 1))
        t_ema_200_array = np.array(ema_200_test).reshape((len(ema_200_test), 1))
        
        t_ema_8_array = np.array(ema_8_test).reshape((len(ema_8_test), 1))
        t_ema_21_array = np.array(ema_21_test).reshape((len(ema_21_test), 1))
        
        t_b_upper_array = np.array(b_upper_test).reshape((len(b_upper_test), 1))
        t_b_lower_array = np.array(b_lower_test).reshape((len(b_lower_test), 1))
        t_b_mid_array = np.array(b_mid_test).reshape((len(b_mid_test), 1))
        t_stoch_array = np.array(stoch_test).reshape((len(stoch_test), 1))
        
        t_open_array = np.array(open_test).reshape((len(open_test), 1))
        t_high_array = np.array(high_test).reshape((len(high_test), 1))
        t_low_array = np.array(low_test).reshape((len(low_test), 1))
        
        ema_50_test_scaled = self.scaler.fit_transform(t_ema_50_array)
        ema_200_test_scaled = self.scaler.fit_transform(t_ema_200_array)
        
        ema_8_test_scaled = self.scaler.fit_transform(t_ema_8_array)
        ema_21_test_scaled = self.scaler.fit_transform(t_ema_21_array)
        
        b_upper_test_scaled = self.scaler.fit_transform(t_b_upper_array)
        b_lower_test_scaled = self.scaler.fit_transform(t_b_lower_array)
        b_mid_test_scaled = self.scaler.fit_transform(t_b_mid_array)
        
        stoch_test_scaled = self.scaler.fit_transform(t_stoch_array)
        
        open_test_scaled = self.scaler.fit_transform(t_open_array)
        high_test_scaled = self.scaler.fit_transform(t_high_array)
        low_test_scaled = self.scaler.fit_transform(t_low_array)
        
        # define test input sequence
        t_in_seq1 = ema_50_test_scaled
        t_in_seq2 = ema_200_test_scaled
        
        t_in_seq3 = ema_8_test_scaled
        t_in_seq4 = ema_21_test_scaled
        
        t_in_seq5 = b_upper_test_scaled
        t_in_seq6 = b_lower_test_scaled
        t_in_seq7 = b_mid_test_scaled
        t_in_seq8 = stoch_test_scaled
        
        t_in_seq9 = open_test_scaled
        t_in_seq10 = high_test_scaled
        t_in_seq11 = low_test_scaled
        
        
        # convert to [rows, columns] structure
        t_in_seq1 = t_in_seq1.reshape((len(t_in_seq1), 1))
        t_in_seq2 = t_in_seq2.reshape((len(t_in_seq2), 1))
        
        t_in_seq3 = t_in_seq3.reshape((len(t_in_seq3), 1))
        t_in_seq4 = t_in_seq4.reshape((len(t_in_seq4), 1))
        t_in_seq5 = t_in_seq5.reshape((len(t_in_seq5), 1))
        t_in_seq6 = t_in_seq6.reshape((len(t_in_seq6), 1))
        t_in_seq7 = t_in_seq7.reshape((len(t_in_seq7), 1))
        t_in_seq8 = t_in_seq8.reshape((len(t_in_seq8), 1))
        t_in_seq9 = t_in_seq9.reshape((len(t_in_seq9), 1))
        t_in_seq10 = t_in_seq10.reshape((len(t_in_seq10), 1))
        t_in_seq11 = t_in_seq11.reshape((len(t_in_seq11), 1))
        
        dataset = hstack((t_in_seq1, t_in_seq2, t_in_seq3, t_in_seq4, t_in_seq5, t_in_seq6, t_in_seq7, t_in_seq8, t_in_seq9, t_in_seq10, t_in_seq11))

        # Generate predictions on test data 
        pred_scaled = model.predict(dataset)
        
        # Convert the predictions back to original scale 
        predicted = self.scaler.inverse_transform(pred_scaled)
        return predicted
        
        
        
    def OnData(self, data):
        
        # Add to window

        #self.AddToWindow(data)
        
        # print(len(self.df))
        # if True:
        #     price = self.PredictPrice()
        price = self.ema50.Current.Value
    #3. Using "Portfolio.Invested" submit 1 order for 20000 GBPUSD:
        # if (not self.Portfolio.Invested) and (len(self.df) == 6):
        #     if (data['GBPUSD'].Close < price) and (price - data['GBPUSD'].Close) > 0.0050 :
        #         self.MarketOrder("GBPUSD", 20000)
        #         self.StopMarketOrder("GBPUSD", -1, price - 0.0050)
        #     elif (data['GBPUSD'].Close - price) > 0.0050 :
        #         self.MarketOrder("GBPUSD", -20000)
        if (not self.Portfolio.Invested):
            if (data['GBPUSD'].Close < self.ema50.Current.Value) :
                self.MarketOrder("GBPUSD", 20000)
                self.StopMarketOrder("GBPUSD", -1, self.ema50.Current.Value - 0.0050)
            else:
                self.MarketOrder("GBPUSD", -20000)
                self.StopMarketOrder("GBPUSD", 1, self.ema50.Current.Valuee + 0.0050)                
        # Check difference between predicted & current price
        # if difference is equal to or above distance to stop loss level place trade