import pandas as pd
import numpy as np 

class ExpEuro:
    """
    A class to calculate option values for European options using the finite-difference method.

    Attributes:
        Volatility (float): The volatility of the underlying asset.
        InterestRate (float): The risk-free interest rate.
        OptionType (str): The type of option ('C' for Call, 'P' for Put).
        StrikePrice (float): The strike price of the option.
        TimeToExpiration (float): The time to expiration in years.
        NumberOfAssetSteps (int): The number of asset steps in the numerical grid.
    """

    def __init__(self, StrikePrice, Volatility, InterestRate, TimeToExpiration, NumberOfAssetSteps, OptionType):
        self.Volatility = Volatility
        self.InterestRate = InterestRate
        self.OptionType = OptionType
        self.StrikePrice = StrikePrice
        self.TimeToExpiration = TimeToExpiration
        self.NumberOfAssetSteps = NumberOfAssetSteps

    def option_value(self):
        NAS = self.NumberOfAssetSteps
        dS = 2 * self.StrikePrice / NAS
        S = np.arange(0, (NAS + 1) * dS, dS)  # Use np.arange for array creation
        dt = 0.9 / self.Volatility**2 / NAS**2
        NTS = int(self.TimeToExpiration / dt) + 1
        dt = self.TimeToExpiration / NTS
        t = np.arange(self.TimeToExpiration, -dt, -dt)  # Use np.arange for time steps
        V = np.zeros((len(S), len(t)))  # Use np.zeros for array initialization
        V = pd.DataFrame(V, index=S, columns=np.around(t, 3))
        q = 1
        if self.OptionType == "P":
            q = -1
        V.iloc[:, 0] = np.maximum(S - self.StrikePrice, 0)  # Use self.StrikePrice here

        for k in range(1, len(t)):
            for i in range(1, len(S) - 1):
                delta = (V.iloc[i + 1, k - 1] - V.iloc[i - 1, k - 1]) / (2 * dS)
                gamma = (V.iloc[i + 1, k - 1] - 2 * V.iloc[i, k - 1] + V.iloc[i - 1, k - 1]) / (dS**2)
                theta = (-0.5 * self.Volatility**2 * S[i]**2 * gamma) - (self.InterestRate * S[i] * delta) + (
                    self.InterestRate * V.iloc[i, k - 1])

                V.iloc[i, k] = V.iloc[i, k - 1] - (theta * dt)

            V.iloc[0, k] = V.iloc[0, k - 1] * (1 - self.InterestRate * dt)
            V.iloc[-1, k] = 2 * V.iloc[-2, k] - V.iloc[-3, k]  # Fixed boundary condition

        V = np.around(V, 3)

        return V
