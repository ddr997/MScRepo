import yfinance as yf

class DataFetcher:
    def __init__(self, ticker):
        self.ticker = yf.Ticker(ticker)

    def getData(self, period):
        return self.ticker.history()

if __name__ == '__main__':
    test = DataFetcher("AAPL")
    print(test.getData("max"))