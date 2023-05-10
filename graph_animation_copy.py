import typer
import time

app = typer.Typer()
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


@app.command()
def demo_log():
    text = ""
    temp = json_handle()

    begin_date = pd.Timestamp('2023-03-03')
    date_range = pd.date_range(start=begin_date, periods=15, freq='D')
    for day_to_filter in date_range:
        single_df = temp[temp['time'].dt.date == day_to_filter.date()]
        text += "day: {} \n".format(day_to_filter.date().strftime("%Y-%m-%d"))

        for _, row in single_df.iterrows():
            text += "time: {},  {} \n".format(row['time_col'], row['ingredients'])

        text += "\n"

    for char in text:
        typer.echo(char, nl=False)
        time.sleep(0.02)
    typer.echo()

if __name__ == "__main__":
    app()