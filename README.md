# Lunar Terrain Segmentation Web Application

Hey, this repository will help you set up your webapp for lunar image segmentation.
* I have used FastAPI for building the API
* I have used Streamlit for the Frontend

## Setup
1. Clone or Download the Repository and open the project directory in your editor (VS Code)
2. Install the requirements
3. You can train your model using the [`train_model.ipynb`](https://github.com/SpartificialUdemy/lunar-segmentation-app/blob/main/train_model.ipynb) python notebook via [Kaggle](https://www.kaggle.com/)
4. Add your trained model in `models` and remove if there are other models present there
5. In command prompt first run your FastAPI app:- `uvicorn backend:app --reload`
6. Then again open command prompt and run the streamlit app:- `streamlit run frontend.py`

## About Trained Model used in this app
* This model is trained using UNET with VGG16 Backbone
* The data used for the training can be found on [Kaggle](https://www.kaggle.com/datasets/romainpessia/artificial-lunar-rocky-landscape-dataset)
* We used first 8000 images from `render` (artificially generted Moon terrain) & `clean` (respective masks) directories for training.
* We used all the other remaining images for validation except the last 4 which we used as test set.
* This model on Validation set gave 80% IOU on average.
* The model was trained as a part of training program at [Spartificial](https://spartificial.com/) where student's task was to improve this IOU score.


## Demo of the FastAPI (Post Request)
1. Passed this image:- `real_moon_kaggle_2.png`
![image](https://github.com/user-attachments/assets/cfde3c50-e017-4a06-afd9-799a82511d31)
2. Here is the segmentation Output:-
![image](https://github.com/user-attachments/assets/44e15df7-7041-451d-bce0-922ee9c230e3)

## Demo of Streamlit Webapp (Requesting API to segment the input image)
1. Initial Look
![image](https://github.com/user-attachments/assets/1621d843-6f21-41ed-88ca-0aa78838b3a3)
2. After Uploading image
![image](https://github.com/user-attachments/assets/4ec069d2-2cbb-42ef-bed2-18fe98dc39ea)
3. After clicking on `Segment Image` button (scroll down to see the output)
![image](https://github.com/user-attachments/assets/26a6f56c-c562-4878-9d4c-a36eb3e49db2)

## Video Demonstration
https://github.com/user-attachments/assets/a92ad839-9a76-473c-933b-ce9c2d098cb1



