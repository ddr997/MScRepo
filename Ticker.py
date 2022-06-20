import yfinance as yf

class Ticker:
    def __init__(self, ticker):
        self.ticker = yf.Ticker(ticker)

    def getData(self, period: int):
        return self.ticker.history(
            period=str(period)+"d"
        )

if __name__ == '__main__':
    test = Ticker("AAPL")