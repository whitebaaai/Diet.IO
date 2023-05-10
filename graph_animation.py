import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

df2 = pd.read_excel('230303-17_MQ_CGM.xlsx', skiprows=1,
                    sheet_name='Scanned_inputs')  # 读取scanned input sheet
df2 = df2[["Device Timestamp", "Scan Glucose mg/dL", "Notes(Calories)", "Notes(Ingredients)"]]
df2.columns = ["time", "glu", "calories", "food"]
df2.head()

from matplotlib.animation import FuncAnimation
df = df2
df['time'] = pd.to_datetime(df['time'])

fig, ax = plt.subplots()

# Set x and y limits
ax.set_xlim(df['time'].iloc[0], df['time'].iloc[-1])
ax.set_ylim(0, 300)

# Create empty line object
line, = ax.plot([], [], lw=2)

# Create function to update the plot
def update(frame):
    # Get the data up to the current frame
    data = df.iloc[:frame+1]

    # Set x and y data for the line
    line.set_data(data['time'], data['glu'])

    # Set labels for each data point
    for _, row in data.iterrows():
        if not pd.isna(row['food']):
            ax.annotate(row['food'], xy=(row['time'], row['glu']), xytext=(5, 5), textcoords='offset points')

    return [line]

# Create animation
ani = FuncAnimation(fig, update, frames=len(df), interval=50)

# Show the plot
plt.show()