import yfinance as yf
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict as dd


# Optional ticker pre-import.
def import_ticker_data(ticker_list, min_time=-1, perf=False):
    ''' 
    Returns ticker:price-data pairs.
            
        Parameters:
            ticker_list (list): A list of ticker symbols.
            (Optional) min_time: If a ticker does not have this many days of data, it is not included.
            (Optional) perf (bool): If 'True', the computation time is printed.
            
    '''

    if perf:
        start = time.perf_counter()

    ticker_dict = {}
    # Iterate over all tickers, collect 'Close' data.
    for ticker in ticker_list:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='max')['Close']
        # We only take tickers which meet the 'min_time' threshold.
        if hist.shape[0] >= min_time:
            ticker_dict[ticker] = hist

    if perf:
        finish = time.perf_counter()
        print(f"Computation time: {round(finish - start, 4)} sec")

    return ticker_dict

# Testing function.


def markov_strat(ticker_data, initial_size, memory= 2, train_prop=0.75, 
                 buy_fee=0, sell_fee=0, perf=False):
    ''' 
    Returns final account balance, percentage profit, and account balance data
    for a selection of tickers, using a Markov Chain strategy.
    
        Parameters:
            ticker_data (dict): ticker:historical_price_data pairs.
            initial_size (float): Initial portfolio size (in dollars.)
            (Optional) memory (int): The amount of memory for the markov chain.
            (Optional) train_prop (float, 0-1): Proportion of data to be used for training.
            (Optional) buy_fee (float): The proportional buy fee, in decimal form.
            (Optional) sell_fee (float): The proportional sell fee, in decimal form.
            (Optional) perf (bool): If 'True', the computation time is printed.
            
        Returns:
            Final account balances, percentage profits, all account balance data.
            
    '''

    if perf:
        start = time.perf_counter()

    # Initialise return dictionaries.
    final_balances = {}
    hist_balance = dd(list)
    profit = {}

    for ticker in ticker_data:
        # Define the train:test data splits.
        train_size = round(len(ticker_data[ticker]) * train_prop)
        train_data = ticker_data[ticker][:train_size]
        test_data = ticker_data[ticker][train_size:]
        train_ticker_data = {ticker: train_data}

        # Get the markov chain only using the training data.
        markov_chain = markov(train_ticker_data, memory=memory, perf=False)[1][ticker]

        # Initialise account states.
        balance = initial_size
        holding = False  # Tracks if there is currently a position.
        position = 0  # Tracks current position size.
        buy_price, sell_price = (0, 0)
        for i in range(2, len(test_data)):

            # We split the cases depending on whether there is a position.
            if not holding:
                priors = list(reversed([int(test_data[i - j] <= test_data[i - j - 1])
                         for j in range(1, memory + 2)]))
                
                if memory == 0:
                    new_prior = [0, priors[0]]
                else:
                    new_prior = priors[:-1] + [0] + priors[-1:]
                    
                # Reorder due to numpy indexing order.            
                random_num = np.random.random()

                # With probability corresponding to the Markov Chain, buy in.
                if markov_chain[tuple(new_prior)] > random_num:
                    holding = True
                    position = balance * (1 - buy_fee)
                    buy_price = test_data[i]

            else:
                priors = list(reversed([int(test_data[i - j] <= test_data[i - j - 1])
                         for j in range(1, memory + 2)]))
                
                if memory == 0:
                    new_prior = [1, priors[0]]
                else:
                    new_prior = priors[:-1] + [1] + priors[-1:]
                    
                random_num = np.random.random()

                # With probability corresponding to the Markov Chain, sell.
                if markov_chain[tuple(new_prior)] > random_num:
                    holding = False
                    sell_price = test_data[i]
                    balance = position * \
                        (sell_price / buy_price) * (1 - sell_fee)

            # Keep track of balance over time.
            hist_balance[ticker].append(balance)

        final_balances[ticker] = balance
        profit[ticker] = balance / initial_size - 1

    if perf:
        final = time.perf_counter()
        print(f"Computation time: {round(final - start, 4)} sec")

    return final_balances, profit, hist_balance


# Markov chain gen.
def markov(tickers, memory=0, states=2, stag_thresh=0.002, perf=False):
    '''
    Returns a tuple of dictionaries containing 2x2 numpy arrays containing
    (i) the number of times each event sequence occured and (ii) the markov chain.
    
        Parameters:
            tickers (list or dict): Either a list of tickers, or a dictionary of ticker:price data.
            (Optional) memory (int): How many prior states to keep track of. 
            (Optional) states (+int): How many states to consider (default 2; inc./dec.)
            (Optional) stag_thresh (float): The price change threshold for which a stock is considered stagnant.
            (Optional) perf (bool): If 'True', the computation time is printed.
    '''

    if perf:
        start = time.perf_counter()

    # If the 'tickers' are given as a list, generate the ticker data.
    if isinstance(tickers, list):
        tickers = import_ticker_data(tickers)

    # Initialise the return dictionaries.
    occurences_dict = {}
    markov_chain_dict = {}

    for ticker in tickers:
        data = tickers[ticker]  # Stores price data for the ticker.

        # Initilialise the markov chain.
        occurences = np.zeros((states,) * (2 + memory))
        
        for i in range(2 + memory, len(data)):
            # Get any conditional price changes (as boolean).
            jumps = [int(data[i - j] <= data[i - j - 1])
                     for j in range(memory + 2)]
            
            # Get jumps into numpy-index-order.
            jumps = list(reversed(jumps[2:])) + jumps[:2]
            
            # Increment the occurences depending on the event-pair.
            occurences[tuple(jumps)] += 1

        markov_chain = occurences.copy()
        dim = len(np.shape(occurences))
        # Fix the axis to sum over to be the event space.
        axis = dim - 2
        
        for index in np.ndindex(np.shape(occurences)):
            # Replace with conditional probability.   
            slicer = [index[i] if i != axis else slice(None) for i in range(len(index))]
            markov_chain[index] /= sum(occurences[tuple(slicer)])

        # Store these numpy arrays to the dictionaries.
        occurences_dict[ticker] = occurences
        markov_chain_dict[ticker] = markov_chain

    if perf:
        finish = time.perf_counter()
        print(f"Computation time: {round(finish - start, 4)} sec")
        return occurences_dict, markov_chain_dict

    return occurences_dict, markov_chain_dict

    
def graph_test_data(results, ticker):
    """
    Shows a graph of previously generated account balance data for a ticker.
    
        Parameters:
            results (dict): A dictionary of dictionaries of ticker balance data.
            ticker (str): The desired ticker symbol.

    """
    balance_data = results[2][ticker]
    trading_days = list(range(1, len(balance_data) + 1))
    plt.plot(trading_days, balance_data)
    plt.title(f'{ticker} Markov Chain Strategy')
    plt.xlabel('Day')
    plt.ylabel('Account Balance ($)')
    plt.show()
    

# (Experimental) Monte Carlo simulation for a single stock.
def monte_carlo(ticker, trials, m_perf, rounding=-1, **kwargs):
    
    '''
    Repeat the Markov Chain Strategy simulation and average the results.
    May take a significant amount of time.
    
        Parameters:
            ticker (str): The ticker symbol of choice.
            trials (+int): The number of trials.
            m_perf (bool): If 'true', the computation time is printed.
            (Optional) rounding (int): Number of decimal places to round. If blank, no rounding is done.
            **kwargs (mix): All other inputs for the simulation (exc. tickers).
                
    '''
    if m_perf:
        start = time.perf_counter()
        
    balances = []
    data = import_ticker_data([ticker])
    for _i in range(trials):
        out = markov_strat(data, **kwargs)
        balance = out[0][ticker]
        balances.append(balance)
    
    if m_perf:
        final = time.perf_counter()
        print(f"Computation Time: {round(final - start, 4)} sec")
        
    if rounding != -1:     
        return round(sum(balances) / len(balances), rounding)
    
    return sum(balances) / len(balances)

    
# Predictor (not produced yet).
def markov_predict():
    pass
