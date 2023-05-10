import streamlit as st
from PIL import Image
import numpy as np
import app_engine
import pandas as pd
import time

# import torch
# import sys
# sys.path.append("./clipseg/models")
# from clipseg import CLIPDensePredT
# from clipseg.models.clipseg import CLIPDensePredT
from PIL import Image
# from torchvision import transforms

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.dates import DateFormatter

# from torch import autocast
# from matplotlib import pyplot as plt

#Helper functions
def resize_image(image, size=512):
    resized_image = image.resize((size, size))
    return np.array(resized_image)

# def mask_image(image):
#     model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
#     model.eval()
#     model.load_state_dict(torch.load('./weights/rd64-uni.pth', 
#                                      map_location=torch.device('cuda')), 
#                                      strict=False)
#     input_image = image
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         transforms.Resize((512, 512)),
#     ])
#     img = transform(input_image).unsqueeze(0)
    
#     prompts = ['area outside of face']
#     with torch.no_grad():
#         preds = model(img.repeat(len(prompts),1,1,1), prompts)[0]
    
#     preds = torch.sigmoid(preds[0][0])
#     array = preds.numpy()

#     # Scale the values in the array to the range 0-255
#     array = (array * 255).astype(np.uint8)
#     array = np.squeeze(array)

#     # Scale the values in the array to the range 0-255
#     preds = Image.fromarray(array)
#     return preds

def json_handle():
    df2 = pd.read_excel('./230303-17_MQ_CGM.xlsx', skiprows=1,
                        sheet_name='Scanned_inputs')  # 读取scanned input sheet
    df2 = df2[["Device Timestamp", "Scan Glucose mg/dL", "Notes(Calories)", "Notes(Ingredients)"]]
    df2.columns = ["time", "glu", "calories", "ingredients"]

    temp = df2.dropna(subset=["ingredients", "calories"])
    temp["time_col"] = temp["time"].dt.strftime("%Y-%m-%d, %H:%M:%S")
    begin_date = pd.Timestamp('2023-03-03')
    end_date = pd.Timestamp('2023-03-10')

    temp = temp[temp['time'].between(begin_date, end_date)]
    return temp

def home_page():
    st.title("DIET.IO")  # 把标题给改了
    
    st.write("Towards the Symbiotic Well-beings of Humans and the Environment")

    cover_image = Image.open("./cover.png")
    st.image(cover_image, caption=" ", use_column_width=True)
    st.write(" ")

    if st.button("Get Nudged "):
        # Set a flag to indicate the user wants to go to the contact page
        st.session_state.page = "entry"
        st.stop()
        # st.experimental_rerun()

def entry_page():
    st.title("DIET.IO")  
    st.subheader("Type in your info for our predictive calculations")

    col1, col2 = st.columns(2)
    # Add text input boxes to the first column
    with col1:
        name = st.text_input("Last_First_Name:  ")
        age = st.text_input("Age:  ")
        place = st.text_input("Place of Residency:  ")

    with col2:
        height = st.text_input("Height(cm): (For BMR calculations)")
        weight = st.text_input("Weight(kg): (For BMR calculationsg)")
        
    # Add text input boxes to the second Section

    st.subheader("Enter your Normal Eating Time ")
    c1, c2 = st.columns(2)
    with c1:

        breakfast_time = st.time_input("For Breakfast", value=None)
        dinner_time = st.time_input("For Dinner", value=None)

    with c2:
        lunch_time = st.time_input("For Lunch", value=None)

    st.write("")
    personal_info = {
            "name": "",
            "age": 0,
            "height": 0,
            "weight": 0, 
            "place": "Earth",
            "breakfast_time": None,
            "lunch_time": None,
            "dinner_time": None}
    selfie_image = None
    if name:
        st.write(f"Hello, {name}!")
        if age:
            try:
                age = int(age)
            except ValueError:
                st.write("Please enter a valid integer for your age.")
        if height:
            try:
                height = int(height)
            except ValueError:
                st.write("Please enter a valid integer for your height.") 
        if weight:
            try:
                weight = int(weight)
            except ValueError:
                st.write("Please enter a valid integer for your weight.")
        if age and height and weight:
            st.write("Personal Information gather complete")

        if place:
            st.write(f"Place of Residency: {place}, .")
        
        if age and height and weight and place \
            and breakfast_time and lunch_time and dinner_time:
            st.write("You normally have your daily meal at the following time:")
            st.write("BREAKFAST Time: ", breakfast_time.strftime("%I:%M %p"))
            st.write("LUNCH Time: ", lunch_time.strftime("%I:%M %p"))
            st.write("DINNER Time: ", dinner_time.strftime("%I:%M %p"))
            st.write("Thanks for your time. Now all the information has been fully gathered. ")

            personal_info["name"] = name
            personal_info["age"] = age
            personal_info["height"] = height
            personal_info["weight"] = weight
            personal_info["place"] = place
            personal_info["breakfast_time"] = breakfast_time
            personal_info["lunch_time"] = lunch_time
            personal_info["dinner_time"] = dinner_time

            st.title("Now, Take a Selfie! ")
            st.write("Please follow the instructions, and upload your image. Thanks for using our website.")

            uploaded_file = st.file_uploader("Upload an image in the format of ['jpg', 'jpeg', 'png']", 
                                            type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                st.write("For better post-processing, the image you uploaded will be resize to a square size.")
                image = Image.open(uploaded_file)
                selfie_image = resize_image(image)
                st.image(selfie_image, caption="Resized Selfie Image", use_column_width=True)

                st.session_state.selfie_img = selfie_image
                st.session_state.personal_info = personal_info

                if st.button("Go to your Health Monitor"):
                    # Set a flag to indicate the user wants to go to the contact page
                    st.session_state.page = "monitor"
                    # st.experimental_rerun()
                    st.stop()
    return

def health_monitor(personal_info, selfie_image):
    st.title("Health Monitor of DIET.IO")
    st.write("Welcome to your personal health monitor")
    # selfie_image = cv2.resize(selfie_image, (300, 350))
    # selfie_4ch, remove_bg = app_engine.remove_background(test_image)
    
    # selfie_image = Image.open("./steaks_remove_bg.png")
    st.write("Thanks for you information, now we will start to process your selfie image...")
    st.write("This is your selfie image: ")
    st.image(selfie_image, use_column_width=True)
    col1, col2 = st.columns(2)
    

    with col1:

        st.write("First, we will be extracting a outline of your face: ")
        progress_bar = st.progress(0)
        for i in range(20):
            # Update the progress bar with each iteration
            progress_bar.progress(5+i*5)
            time.sleep(0.1)
    
        edges = Image.open("./canny.png")
        st.image(edges, use_column_width=True)

    with col2:
        
        
        st.write("\n Second, we will be extracting mask to locate your central face area: ")
        with st.spinner('Running Mask Extraction function...'):
            progress_bar = st.progress(0)
            # mask = mask_image(selfie_image)
            mask = Image.open("./mask.png")
            time.sleep(2.0)
            progress_bar.progress(100)
        st.image(mask, use_column_width=True)
    
    st.markdown(" ** Now we will use Stable Diffusion Pipeline to create figure of you. **")
    prompt = "portrait + style A photo of a " + personal_info["name"] +" man, black hair, healthy-looking face, bright skins"
    st.write("Example: "  + prompt)
    prompt_input = st.text_input("Write a brief description of yourself: ")

    if prompt_input: 
        st.markdown('''
                **Now we will auto-generate a negative prompt which are the characteristics does not belong to you: ** 
                \n\n
                ''')
        
        st.write('''        
                honeycomb, blue eyes, white, grey, tatoo, dark skin, white shirt, \n
                hat, blurry, dead eyes, round nose, non-Asian, black-white,  \n
                open-mouth, mask, beard, multiple people, 3d, cgi, fake human, \n
                glasses, double_eye_yelid, a different person \n
                \n
                ''')
        
        st.markdown("**Now let the Stable Diffusion pipeline work: **")
        progress_bar = st.progress(0)

        for i in range(50):
            # Update the progress bar with each iteration
            progress_bar.progress(i*2+2)
            time.sleep(0.1)

        stable_result = Image.open("./stable.png")
        st.image(stable_result, use_column_width="True")

    c1, c2 = st.columns(2)
    with c1:
        pass
    with c2:  
        if st.button("Go to Next step"):
            # Set a flag to indicate the user wants to go to the contact page
            st.session_state.page = "glu_days"
            # st.experimental_rerun()
            st.stop()

def glu_days():
    st.header("Now we will start to monitor your glucose level at the backend")
    begin_date = pd.Timestamp('2023-03-03')
        
    temp = json_handle()
    date_range = pd.date_range(start=begin_date, periods=7, freq='D')
    for day_to_filter in date_range:
        single_df = temp[temp['time'].dt.date == day_to_filter.date()]
        st.subheader("day: {}".format(day_to_filter.date().strftime("%Y-%m-%d")))
        for _, row in single_df.iterrows():
            col1, col2, col3 = st.columns([5, 10, 2])
            with col1:
                st.write(row['time_col'])
            with col2:
                st.write(row['ingredients'])
            with col3:
                st.write(row["glu"])
            time.sleep(0.2)
    
    st.subheader("We collected the first 7 days of the user data, and now we can\
                  use further new data to render this user's predicted figure.")
    st.subheader("Wanna See our Prediction? click the button below.")

    if st.button("See our predictions"):
            # Set a flag to indicate the user wants to go to the contact page
        st.session_state.page = "prediction"
        # st.experimental_rerun()
        st.stop()
    
def predictions_page():
    st.title("First let's see you glucose flucatation of the past 7 days.")
    
    df2 = pd.read_excel('230303-17_MQ_CGM.xlsx', skiprows=1,
                    sheet_name='Scanned_inputs')  # 读取scanned input sheet
    df2 = df2[["Device Timestamp", "Scan Glucose mg/dL", "Notes(Calories)", "Notes(Ingredients)"]]
    df2.columns = ["time", "glu", "calories", "food"]
   
    df = df2
    df['time'] = pd.to_datetime(df['time'])
    data =  df[["time","glu"]]

    fig, ax = plt.subplots(figsize=(16,6))
    ax.plot(data['time'], data['glu'], marker='.')
    ax.set_xlabel('Time')
    ax.set_ylabel('Glucose Level')

    # Format x-axis
    date_formatter = DateFormatter('%m/%d/%y')
    ax.xaxis.set_major_formatter(date_formatter)
    fig.autofmt_xdate(rotation=45)

    # Display plot in Streamlit
    st.pyplot(fig)
    
    st.subheader("Glucose Summary: ")
    slopes = app_engine.slopes()

    for slope in slopes:
        st.write("Date: " , slope["date"], " Your Average change of Glucose Slope: ", slope["slope"])
        time.sleep(0.2)

    st.write("The percentage of unhealthy days are above 57.1 %")
    st.write("You need to control your diet !!!")
    time.sleep(4.0)

    st.subheader("Want to see what will happen if you keeping this diet: ")

    st.write("The system just noticed your last meal was a meal of steak: ")
    st.write("Does this looks familiar? ")
    time.sleep(0.3)

    
    st.image("./steaks.png")


    st.subheader("Not so dramatic? What about we place you in a different background? ")
    time.sleep(4.0)
    
    progress_bar = st.progress(0)
    for i in range(20):
        # Update the progress bar with each iteration
        progress_bar.progress(5+i*5)
        time.sleep(0.1)
    st.image("./final.png")

# def test():
#     predictions_page()


def main():
    if "page" not in st.session_state:
        st.session_state["page"] = "home"
    if "personal_info" not in st.session_state:
        st.session_state.personal_info = {}
    if "selfie_img" not in st.session_state:
        st.session_state.selfie_img = None

    # Display the appropriate page based on the session state
    pages = {
        "home": home_page,
        "entry": entry_page,
        "monitor": health_monitor,
        "glu_days": glu_days,
        "prediction": predictions_page,
    }

    if st.session_state.page == "monitor":
        health_monitor(st.session_state.personal_info, st.session_state.selfie_img)
    else:
        pages[st.session_state.page]()

if __name__ == "__main__":
    main()
    # test()