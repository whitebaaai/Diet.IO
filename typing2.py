import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
import pandas as pd

def json_handle():
    df2 = pd.read_excel('./230303-17_MQ_CGM.xlsx', skiprows=1,
                        sheet_name='Scanned_inputs')  # 读取scanned input sheet
    df2 = df2[["Device Timestamp", "Scan Glucose mg/dL", "Notes(Calories)", "Notes(Ingredients)"]]
    df2.columns = ["time", "glu", "calories", "ingredients"]

    temp = df2.dropna(subset=["ingredients", "calories"])
    temp["time_col"] = temp["time"].dt.strftime("%Y-%m-%d, %H:%M:%S")
    begin_date = pd.Timestamp('2023-03-03')
    end_date = pd.Timestamp('2023-03-17')

    temp = temp[temp['time'].between(begin_date, end_date)]
    return temp


df2 = pd.read_excel('230303-17_MQ_CGM.xlsx', skiprows=1,
                    sheet_name='Scanned_inputs')  # 读取scanned input sheet
df2 = df2[["Device Timestamp", "Scan Glucose mg/dL", "Notes(Calories)", "Notes(Ingredients)"]]
df2.columns = ["time", "glu", "calories", "food"]

from matplotlib.animation import FuncAnimation
df = df2
df['time'] = pd.to_datetime(df['time'])

# Create the figure and gridspec
fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(2, 1, height_ratios=[8,4])


text = ""
temp = json_handle()

begin_date = pd.Timestamp('2023-03-03')
date_range = pd.date_range(start=begin_date, periods=10, freq='D')
for day_to_filter in date_range:
    single_df = temp[temp['time'].dt.date == day_to_filter.date()]
    text += "day: {} \n".format(day_to_filter.date().strftime("%Y-%m-%d"))

    for _, row in single_df.iterrows():
        
        text += "time: {},  {} \n".format(row['time_col'], row['ingredients'])

    text += "\n"
# Create the plot
ax = fig.add_subplot(gs[1])
ax.set_axis_off()
txt = ax.text(0, 1.0, "", fontsize=8)

text_ax = fig.add_subplot(gs[0])
text_ax.set_xlim(df['time'].iloc[0], df['time'].iloc[-1])
text_ax.set_ylim(50, 210)
ax.set_clip_on(True)
# Create empty line object
line, = text_ax.plot([], [], lw=2)
# Create function to update the plot
def update_frame(frame):
    # Get the data up to the current frame
    data = df.iloc[:frame+1]

    # Set x and y data for the line
    line.set_data(data['time'], data['glu'])

    return [line]

# Define the update function
def update(i):
    txt.set_text(text[:i*(len(text)+1)//len(df)])
    return txt

# Create the animation
anim = FuncAnimation(fig, update, frames=len(df), interval=200)

secondanim = FuncAnimation(fig, update_frame, frames=len(df), interval=200)

plt.show()
