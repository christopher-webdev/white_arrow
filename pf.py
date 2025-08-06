
import numpy as np
import pandas as pd

# Confusion matrix from meta classifier
conf_matrix = np.array([
 [6999, 1660,  738,  880],
 [1091, 2131,  745,  345],
 [ 318,  448, 1358,  553],
 [ 326,  194,  460, 3739],
])

# R-multiples per class
rr_values = {
    0: 0,    # Reject (no trade)
    1: 1,    # 1:1
    2: 2,    # 1:2
    3: 3     # 1:3
}

# Simulate PnL
total_trades = 0
total_pnl = 0

for actual in range(4):
    for predicted in range(4):
        count = conf_matrix[actual, predicted]
        if predicted == 0 or predicted ==1:
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