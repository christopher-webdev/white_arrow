
import numpy as np
import pandas as pd

# Confusion matrix from meta classifier
conf_matrix = np.array([
 [466,   6,   2],
 [ 12, 128,   6],
 [ 18,  10, 420],
])

# R-multiples per class
rr_values = {
    0: 0,    # Reject (no trade)
    1: 1,    # 1:1
    2: 2,    # 1:2
    # 3: 3     # 1:3
}

# Simulate PnL
total_trades = 0
total_pnl = 0

for actual in range(3):
    for predicted in range(3):
        count = conf_matrix[actual, predicted]
        if predicted == 0:# or predicted == 1:
            continue  # No trade taken
        elif actual == predicted:
            total_pnl += count * rr_values[predicted]  # Correct prediction
        else:
            total_pnl += count * -1  # Wrong prediction
        total_trades += count

# Calculate average R
average_r = total_pnl / total_trades

# Create and show summary
pnl_summary = pd.DataFrame({
    "Total Trades Taken": [total_trades],
    "Total R": [total_pnl],
    "Average R per Trade": [average_r]
})
print(pnl_summary)