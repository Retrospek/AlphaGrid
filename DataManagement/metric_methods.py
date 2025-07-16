import numpy as np

def calculate_sector_momentums(sector_separated_data, momentums) -> dict:
    for sector in sector_separated_data.keys():
        sector_data = sector_separated_data[sector] # Not using copy so it remains in a dynamic state

        # Calcuting the separate momentums for specific "sector"
        for momentum in momentums:
            sector_data[f"Opening_Momentum_{momentum}Days"] = sector_data["Open"] - sector_data["Open"].shift(momentum)
            sector_data[f"Closing_Momentum_{momentum}Days"] = sector_data["Close"] - sector_data["Close"].shift(momentum)
            sector_data[f"High_Momentum_{momentum}Days"] = sector_data["High"] - sector_data["High"].shift(momentum)
            sector_data[f"Low_Momentum_{momentum}Days"] = sector_data["Low"] - sector_data["Low"].shift(momentum)
            sector_data[f"Volume_Momentum_{momentum}Days"] = sector_data["Volume"] - sector_data["Volume"].shift(momentum)
            
    return sector_separated_data

def calculate_sector_moving_averages(sector_separated_data, moving_averages) -> dict:
    for sector in sector_separated_data.keys():
        sector_data = sector_separated_data[sector] # Not using copy so it remains in a dynamic state

        # Calcuting the separate momentums for specific "sector"
        for ma in moving_averages:
            sector_data[f"Opening_MovingAverage_{ma}Days"] = sector_data["Open"].rolling(window=ma).mean()
            sector_data[f"Closing_MovingAverage_{ma}Days"] = sector_data["Close"].rolling(window=ma).mean()
            sector_data[f"High_MovingAverage_{ma}Days"] = sector_data["High"].rolling(window=ma).mean()
            sector_data[f"Low_MovingAverage_{ma}Days"] = sector_data["Low"].rolling(window=ma).mean()
            sector_data[f"Volume_MovingAverage_{ma}Days"] = sector_data["Volume"].rolling(window=ma).mean()
            
        for bound in ["Intra", "Inter"]:
            for ma in moving_averages:  # Calculate for ALL periods, not just 14
                sector_data[f"{bound}_Gain_MovingAverage_{ma}Days"] = sector_data[f"{bound}_Gain"].rolling(window=ma).mean()
                sector_data[f"{bound}_Loss_MovingAverage_{ma}Days"] = sector_data[f"{bound}_Loss"].rolling(window=ma).mean()
                
    return sector_separated_data

def calculate_sector_relative_strengths(sector_separated_data, rsi_periods=[14]) -> dict:
    for sector in sector_separated_data.keys():
        sector_data = sector_separated_data[sector] # Not using copy so it remains in a dynamic state

        # Calcuting the separate momentums for specific "sector"
        for prd in rsi_periods:
            for bound in ["Intra", "Inter"]:    
                avg_gain = sector_data[f"{bound}_Gain_MovingAverage_{prd}Days"]
                avg_loss = sector_data[f"{bound}_Loss_MovingAverage_{prd}Days"]

                rs = np.where(avg_loss != 0, avg_gain / avg_loss, np.inf)
                sector_data[f"{bound}_RSI_{prd}Days"] = np.where(
                    np.isinf(rs), 
                    100.0,
                    100 - (100 / (1 + rs))
                )

    return sector_separated_data

def calculate_sector_realized_volatility(sector_separated_data, periods) -> dict:
    """
    Inner Volatility DURING the day -> Intra Only
    """
    for sector in sector_separated_data.keys():
        sector_data = sector_separated_data[sector]

        for prd in periods:
            sector_data[f"RV_{prd}Days"] = (
                sector_data["IntraDay_Return(%)"]
                .rolling(window=prd)
                .apply(lambda x: np.sqrt(np.sum(x**2)), raw=True)
            )
            rv_current = sector_data[f"RV_{prd}Days"]
            rv_previous = sector_data[f"RV_{prd}Days"].shift(1)

            sector_data[f"RV_{prd}Days_1DayDelta(%)"] = np.where(
                rv_previous != 0,
                (rv_current - rv_previous) / rv_previous,
                0.0
            )
    return sector_separated_data

