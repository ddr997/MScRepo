from UI.central.Plot import Plot
class StateData:

    def __init__(self):

        # data fields
        self.stock = ""
        self.period = 0
        self.stockDataColumns = {}


        # indicators
        self.selectedIndicators = []


        self.model = ""

        # object fields
        self.ticker = None
        self.dataFrame = None
        self.plot = Plot()