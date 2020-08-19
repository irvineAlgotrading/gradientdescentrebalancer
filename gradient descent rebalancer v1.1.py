#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import requests
import pandas as pd
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt


# In[2]:


import os
import pandas as pd

TSLAdata=pd.read_csv("C:/Users/Andy/Downloads/TSLA/TSLA_2020_2020.txt",delimiter="\,")
TSLAdata.columns = ["timestamp", "open", "high", "low", "close", "volume"]
#TSLAdata


# In[3]:


# GLOBAL

tickers = ["TSLA"]
mins_ago_to_fetch = 100000  # see also filter_history_by_date()
ticker_history = {}
hist_length = 0
average_returns = {}
cumulative_returns = {}

def fetch_all():
  for ticker in tickers:
    ticker_history[ticker] = fetch_history(ticker)

def fetch_history(ticker):
  hist = pd.read_csv("C:/Users/Andy/Downloads/TSLA/TSLA_2020_2020.txt",delimiter="\,")
  hist.columns = ["timestamp", "Open", "High", "Low", "Close", "Volume"]
  #hist = TSLAdata
  hist = index_history(hist)
  hist = filter_history_by_date(hist)
  return hist

def index_history(hist):
  # index by date so we can easily filter by a given timeframe
  hist = hist.set_index('timestamp')
  hist.index = pd.to_datetime(hist.index, unit='ns')
  return hist

def filter_history_by_date(hist):
  result = hist[hist.index.year >= 2016]
  # result = result[result.index.day == 1] # every first of month, etc.
  return result

fetch_all()
hist_length = len(ticker_history[tickers[0]])


# In[4]:


ticker_history['TSLA']


# In[7]:


#TSLAdata[‘close’].fillna(value=TSLAdata[‘Item_Weight’].mean(), inplace=True)


# ![alt text](https://cdn-images-1.medium.com/max/750/0*XcQBOZQBItuoaFKi)
# 
# Daily return of an asset is the ratio between the price variation over the initial price.

# ![alt text](https://cdn-images-1.medium.com/max/750/0*_SHk9qM3JOSGKeNE)
# 
# The excess return over a certain period is the difference between the average return and the actual return.

# In[9]:


# Calculate returns and excess returns

def add_all_returns():
  for ticker in tickers:
    hist = ticker_history[ticker]
    hist['return'] = (hist['Close'] - hist['Open']) / hist['Open']
    average = hist["return"].mean()
    average_returns[ticker] = average
    cumulative_returns[ticker] = (hist["return"] + 1).prod() - 1
    hist['excess_return'] = hist['return'] - average
    ticker_history[ticker] = hist

add_all_returns()

# display data
cumulative_returns


# Several things just happened here:
# 
# We added the return and excess_return columns to every row of our historic data frames
# We stored the average return of every asset
# We stored the cumulative return for every historic we have

# Now it’s time to combine the excess returns into the n × k Excess Return Matrix X, where n is the number of observations (days, weeks, months, …) and k is the number of assets in out portfolio.

# ![alt text](https://cdn-images-1.medium.com/max/1000/0*pGapNlxAygWJdwQt)

# In[10]:


# Excess matrix

excess_matrix = np.zeros((hist_length, len(tickers)))

for i in range(0, hist_length):
  for idx, ticker in enumerate(tickers):
    excess_matrix[i][idx] = ticker_history[ticker].iloc[i]['excess_return']


# In[11]:


# Excess matrix

excess_matrix = np.zeros((hist_length, len(tickers)))

for i in range(0, hist_length):
  for idx, ticker in enumerate(tickers):
    excess_matrix[i][idx] = ticker_history[ticker].iloc[i]['excess_return']


# The first line creates an n × k matrix and the loops assign the corresponding values to each ticker.

# In[12]:


pretty_matrix = pd.DataFrame(excess_matrix).copy()
pretty_matrix.columns = tickers
pretty_matrix.index = ticker_history[tickers[0]].index

pretty_matrix


# ![alt text](https://cdn-images-1.medium.com/max/800/0*MKgTBXtRYApe1ycw)

# **Risk Modeling**
# 
# There are many approaches that can be used to optimize a portfolio. In the present article we will analyze the variance and covariance of individual assets in order to minimize the global risk.
# 
# To this end, we will use our Excess Return Matrix to compute the Variance-covariance Matrix Σ from it.

# In[13]:


# Variance co-variance matrix

product_matrix = np.matmul(excess_matrix.transpose(), excess_matrix)
var_covar_matrix = product_matrix / hist_length


# In[14]:


pretty_matrix = pd.DataFrame(var_covar_matrix).copy()
pretty_matrix.columns = tickers
pretty_matrix.index = tickers

pretty_matrix


# ![alt text](https://cdn-images-1.medium.com/max/800/0*00nkyzYnQm3T28xN)

# cov x, y is the covariance between asset X and asset Y
# 
# When x = y the value is the variance of the asset

# Before we can jump into the actual portfolio optimization, our next target is the Correlation Matrix, where every item is defined like:

# ![alt text](https://cdn-images-1.medium.com/max/600/0*iPVjT37R1LW6FMZp)

# The correlation between assets X and Y is their covariance divided by the product of their standard deviations.
# 
# We already have cov(X, Y) stored in var_covar_matrix so we need a k × k matrix with the products of each standard deviation.
# 
# Let’s compute the individual standard deviations:

# In[18]:


# Standard Deviation

std_deviations = np.zeros((len(tickers), 1))

for idx, ticker in enumerate(tickers):
  std_deviations[idx][0] = np.std(ticker_history[ticker]['return'])


# In[19]:


pretty_matrix = pd.DataFrame(std_deviations).copy()
pretty_matrix.columns = ['Std Dev']
pretty_matrix.index = tickers

pretty_matrix


# To generate the matrix with the standard deviation products, we multiply the above by its transpose.

# In[20]:


# Std Deviation products matrix

sdev_product_matrix = np.matmul(std_deviations, std_deviations.transpose())


# In[21]:


pretty_matrix = pd.DataFrame(sdev_product_matrix).copy()
pretty_matrix.columns = tickers
pretty_matrix.index = tickers

pretty_matrix


# So now, we can finally compute the Correlation Matrix, as we defined before.

# In[22]:


# Correlation matrix

correlation_matrix = var_covar_matrix / sdev_product_matrix


# In[23]:


pretty_matrix = pd.DataFrame(correlation_matrix).copy()
pretty_matrix.columns = tickers
pretty_matrix.index = tickers

pretty_matrix


# In the Correlation Matrix:
# 
# The correlation of an asset’s returns with itself is always 1
# 
# Correlation values range from –1 to 1
# 
# Values tending to 1 mean that two random variables tend to have linear relationship
# 
# Correlation values tending to –1 (anticorrelation) mean that two assets tend to have opposite behaviors
# 
# Correlation values of 0 mean that two random variables are independent

# **Portfolio optimization**
# 
# Given the average return and the variance of our assets, now it’s time to decide how much money is allocated in each one.
# 
# At this point, we would like to find a combination of investments that minimizes the global variance of the portfolio.

# ![alt text](https://cdn-images-1.medium.com/max/600/0*T-7HtN5qlYnlU1ay)

# The weights array is the output we aim to get from our portfolio optimizer. The weight of every asset can range from 0 to 1, and the overall sum must be 1.

# Given the weights array, we can define the weighted standard deviation as:
# 
# ![alt text](https://cdn-images-1.medium.com/max/800/0*3hUQbQITXFV8ARY0)

# So the global variance of our portfolio can now be defined as:
# 
# ![alt text](https://cdn-images-1.medium.com/max/800/0*liHv3gq7jchT_9BZ)
# 
# Where W is a 1 × k matrix with the weighted standard deviations , C is the Correlation Matrix described above and the result is a 1 × 1 matrix with the global portfolio variance.
# 
# This is the value that we want to minimize, but how can we do it? 
# We could define functions that computed the global variance for given weight arrays, explore all the possible candidates and rate them.
# 
# However, finding the absolute minimal value for an equation with k variables is an NP problem. The amount of calculations would grow exponentially with k if we attempted to evaluate every possible solution. Waiting 10³⁰ centuries to get an answer doesn’t look like an appealing scenario, does it?
# 
# So the best alternative in our hands is to use Machine Learning to explore a diverse subset of the search space for us, and let it explore variants of branches with potential to perform better than their siblings.

# **Optimize weights using Tensorflow**

# In[26]:


# Optimize weights to minimize variance

def minimize_volatility():

  # Define the model
  # Portfolio Volatility = Sqrt (Transpose (Wt.SD) * Correlation Matrix * Wt. SD)

  ticker_weights = tf.Variable(np.full((len(tickers), 1), 1.0 / len(tickers))) # our variables
  weighted_std_devs = tf.multiply(ticker_weights, std_deviations)

  product_1 = tf.transpose(weighted_std_devs)
  product_2 = tf.matmul(product_1, correlation_matrix)
  
  portfolio_variance = tf.matmul(product_2, weighted_std_devs)
  portfolio_volatility = tf.sqrt(tf.reduce_sum(portfolio_variance))


  # Run
  learn_rate = 0.01
  steps = 5000
  
  init = tf.global_variables_initializer()

  # Training using Gradient Descent to minimize variance
  train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(portfolio_volatility)

  with tf.Session() as sess:
    sess.run(init)
    for i in range(steps):
      sess.run(train_step)
      if i % 1000 == 0 :
        print("[round {:d}]".format(i))
        print("Weights", ticker_weights.eval())
        print("Volatility: {:.2f}%".format(portfolio_volatility.eval() * 100))
        print("")
        
    return ticker_weights.eval()

weights = minimize_volatility()

pretty_weights = pd.DataFrame(weights * 100, index = tickers, columns = ["Weight %"])
pretty_weights


# In[27]:


# Optimize weights to minimize variance

def minimize_volatility():

  # Define the model
  # Portfolio Volatility = Sqrt (Transpose (Wt.SD) * Correlation Matrix * Wt. SD)

  ticker_weights = tf.Variable(np.full((len(tickers), 1), 1.0 / len(tickers))) # our variables
  weighted_std_devs = tf.multiply(ticker_weights, std_deviations)

  product_1 = tf.transpose(weighted_std_devs)
  product_2 = tf.matmul(product_1, correlation_matrix)
  
  portfolio_variance = tf.matmul(product_2, weighted_std_devs)
  portfolio_volatility = tf.sqrt(tf.reduce_sum(portfolio_variance))

  # Constraints: sum([0..1, 0..1, ...]) = 1

  lower_than_zero = tf.greater( np.float64(0), ticker_weights )
  zero_minimum_op = ticker_weights.assign( tf.where (lower_than_zero, tf.zeros_like(ticker_weights), ticker_weights) )

  greater_than_one = tf.greater( ticker_weights, np.float64(1) )
  unity_max_op = ticker_weights.assign( tf.where (greater_than_one, tf.ones_like(ticker_weights), ticker_weights) )

  result_sum = tf.reduce_sum(ticker_weights)
  unity_sum_op = ticker_weights.assign(tf.divide(ticker_weights, result_sum))
  
  constraints_op = tf.group(zero_minimum_op, unity_max_op, unity_sum_op)

  # Run
  learning_rate = 0.01
  steps = 5000
  
  init = tf.global_variables_initializer()

  # Training using Gradient Descent to minimize variance
  optimize_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(portfolio_volatility)

  with tf.Session() as sess:
    sess.run(init)
    for i in range(steps):
      sess.run(optimize_op)
      sess.run(constraints_op)
      if i % 2500 == 0 :
        print("[round {:d}]".format(i))
        print("Weights", ticker_weights.eval())
        print("Volatility: {:.2f}%".format(portfolio_volatility.eval() * 100))
        print("")
        
    sess.run(constraints_op)
    return ticker_weights.eval()

weights = minimize_volatility()

pretty_weights = pd.DataFrame(weights * 100, index = tickers, columns = ["Weight %"])
pretty_weights


# In[28]:


# Optimize weights to maximize return/risk

import time
start = time.time()

def maximize_sharpe_ratio():
  
  # Define the model
  
  # 1) Variance
  
  ticker_weights = tf.Variable(tf.random_uniform((len(tickers), 1), dtype=tf.float64)) # our variables
  weighted_std_devs = tf.multiply(ticker_weights, std_deviations)
  
  product_1 = tf.transpose(weighted_std_devs)
  product_2 = tf.matmul(product_1, correlation_matrix)
  
  portfolio_variance = tf.matmul(product_2, weighted_std_devs)
  portfolio_volatility = tf.sqrt(tf.reduce_sum(portfolio_variance))

  
  # 2) Return
  
  returns = np.full((len(tickers), 1), 0.0) # same as ticker_weights
  for ticker_idx in range(0, len(tickers)):
    returns[ticker_idx] = cumulative_returns[tickers[ticker_idx]]
  
  portfolio_return = tf.reduce_sum(tf.multiply(ticker_weights, returns))
  
  # 3) Return / Risk
  
  sharpe_ratio = tf.divide(portfolio_return, portfolio_volatility)
  
  # Constraints
  
  # all values positive, with unity sum
  weights_sum = tf.reduce_sum(ticker_weights)
  constraints_op = ticker_weights.assign(tf.divide(tf.abs(ticker_weights), tf.abs(weights_sum) ))
  
  # Run
  learning_rate = 0.0001
  learning_rate = 0.0015
  steps = 10000
  
  # Training using Gradient Descent to minimize cost
  
  optimize_op = tf.train.GradientDescentOptimizer(learning_rate, use_locking=True).minimize(tf.negative(sharpe_ratio))
  #2# optimize_op = tf.train.AdamOptimizer(learning_rate, use_locking=True).minimize(tf.negative(sharpe_ratio))
  #3# optimize_op = tf.train.AdamOptimizer(learning_rate=0.00005, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False).minimize(tf.negative(sharpe_ratio))
  #4# optimize_op = tf.train.AdagradOptimizer(learning_rate=0.01, initial_accumulator_value=0.1, use_locking=False).minimize(tf.negative(sharpe_ratio))
  
  
  init = tf.global_variables_initializer()
  
  with tf.Session() as sess:
    ratios = np.zeros(steps)
    returns = np.zeros(steps)
    sess.run(init)
    for i in range(steps):
      sess.run(optimize_op)
      sess.run(constraints_op)
      ratios[i] = sess.run(sharpe_ratio)
      returns[i] = sess.run(portfolio_return) * 100
      if i % 2000 == 0 : 
        sess.run(constraints_op)
        print("[round {:d}]".format(i))
        #print("Ticker weights", sess.run(ticker_weights))
        print("Volatility {:.2f} %".format(sess.run(portfolio_volatility)))
        print("Return {:.2f} %".format(sess.run(portfolio_return)*100))
        print("Sharpe ratio", sess.run(sharpe_ratio))
        print("")
    
    sess.run(constraints_op)
    # print("Ticker weights", sess.run(ticker_weights))
    print("Volatility {:.2f} %".format(sess.run(portfolio_volatility)))
    print("Return {:.2f} %".format(sess.run(portfolio_return)*100))
    print("Sharpe ratio", sess.run(sharpe_ratio))
    return sess.run(ticker_weights)

weights = maximize_sharpe_ratio()

print("Took {:f}s to complete".format(time.time() - start))
pretty_weights = pd.DataFrame(weights * 100, index = tickers, columns = ["Weight %"])

pretty_weights


# In[29]:


# PLOTTING

import matplotlib.pyplot as plt

def line_plot(line1, label1=None, units='', title=''):
    fig, ax = plt.subplots(1, figsize=(16, 9))
    ax.plot(line1, label=label1, linewidth=2)
    ax.set_ylabel(units, fontsize=14)
    ax.set_title(title, fontsize=18)
    ax.legend(loc='best', fontsize=18)

def lines_plot(line1, line2, label1=None, label2=None, units='', title=''):
    fig, ax = plt.subplots(1, figsize=(16, 9))
    ax.plot(line1, label=label1, linewidth=2)
    ax.plot(line2, label=label2, linewidth=2)
    ax.set_ylabel(units, fontsize=14)
    ax.set_title(title, fontsize=18)
    ax.legend(loc='best', fontsize=18)

