import numpy as np
from matplotlib import pyplot as plt
from urllib.request import urlopen
import pandas as pd 
# from transformers import pipeline
# from rembg import remove
from PIL import Image

import openai
openai.api_key = "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"

def load_selfie(image_url):
  img = Image.open(image_url)
  img = np.array(img)
  return img

# def remove_background(image):
#   output = remove(image)
#   output_RGB = cv2.cvtColor(output, cv2.COLOR_RGBA2RGB)
#   return output, output_RGB

def slopes():
  glu_df = pd.read_csv("output.csv")

  glu_df = glu_df.drop(columns=["gap_minute", "glu_diff"])
  glu_df = glu_df.fillna(0.0)
  glu_df["time"] = pd.to_datetime(glu_df["time"])
  glu_df["time"] = glu_df["time"].dt.strftime("%Y-%m-%d, %H:%M:%S")
  slope = glu_df[["time", "slope"]]

  slope = slope.fillna(0)
  slope['time'] = pd.to_datetime(slope['time'])
  group_slope = slope.groupby(slope["time"].dt.date)

  import numpy as np

  avg_slope = []
  for date, group in group_slope:
      group = np.array(group['slope'])
      avg_slope.append({"date": date.strftime("%Y-%m-%d"),
                        "slope": np.absolute(group).mean()})
      
  return avg_slope