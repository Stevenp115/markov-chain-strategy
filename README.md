# markov-chain-strategy
A simple Markov Chain stock-trading strategy backtester in Python. The actual performance is strictly tied to the perfomance of the stock itself, and is intended to simulate a 'baseline' strategy.

### Built With
* [yfinance](https://pypi.org/project/yfinance/)
* [Matplotlib](https://matplotlib.org/)
* [NumPy](https://numpy.org/)

## Methodology
The price data for the stocks is scraped from Yahoo Finance via the yfinance library. Then in order to generate the transition matrix of the Markov Chain, we keep track of the states "profit" and "loss" from one day to the next; these are gathered from taking iterating through the price data. The only information stored in the memory of the matrix is the state at a current time, and the previous timestep.

In the strategy, the value in the markov chain, rather than determining if the trade occurs or not, gives the probablity that a purchase/sale will be made (given the prior state).

## Usage
To find the Markov Chain transition matrix for a given `ticker`, then we can do the following:
```python
from markov_test import markov

print(markov(ticker))
```

To backtest the strategy on tickers `ticker1` and `ticker2`, assuming buy/sell fees of 0.5%, an initial portfolio size of $100, and a train-test split of 0.75, we can run the following:
```python
from markov_test import *

ticker_data = import_ticker_data([ticker1, ticker2])
markov_strat(ticker_data, 100, train_prop=0.75, buy_fee=0.005, sell_fee=0.005, perf=False)
             
# Optionally set 'perf' to True if you want the computation time printed.         
```

To run a Monte Carlo simulation with 1000 trials on a particular ticker, `ticker1` (using the same parameters as above), we can run the following:
```python
from markov_test import *
monte_carlo(ticker1, 1000, True, 3, initial_size=100, train_prop=0.75, buy_fee=0.005, sell_fee=0.005)

```
