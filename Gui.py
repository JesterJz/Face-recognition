import os
import cv2
import numpy as np
from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
from tkinter.font import Font
from tensorflow.keras import models

model = models.load_model('./model/model_colab_6.h5')

face_detector = cv2.CascadeClassifier(
    './haarcascades/haarcascade_frontalface_alt.xml')
# lst_resutl = {'dimaria':[1,0,0,0], 'neymar':[0,1,0,0], 'phuong_ly':[0,0,1,0], 'ronaldo':[0,0,0,1]}
lst_resutl = ['Dimaria', 'Neymar', 'Phuong Ly', 'Cristiano Ronaldo']


# create windows
root = Tk()
root.title("Face Recognition")
root.geometry("1200x820")
root.wm_resizable(width=True, height=True)
# root.configure(background='#80CBC4')
# root.maxsize(1920, 1080)

img_counter = 0
temp_img = None

pic_default = Image.open('./asset/image/icon_default.png')
img_def = ImageTk.PhotoImage(pic_default)

# Set Title Name App
name = Label(root, text="Face Recognition in video", fg="#000", bd=0)
name.config(font=("Engravers MT", 20))
name.grid(column=0, row=0, columnspan=10, pady=10)

image_box = Label(root, image=img_def, width=720,
                  height=640, bg="#00695C")
image_box.grid(column=0, row=1, columnspan=3, rowspan=10, padx=20, pady=5)


best_probabilities_txt = Label(
    root, text="Độ chính xác: ", fg="#004D40", bd=1, bg="#66BB6A")
best_probabilities_txt.config(font=("Arial", 20, "bold"))
best_probabilities_txt.grid(column=4, row=1, padx=10, pady=5)

best_probabilities = Label(root, text="None", fg="#004D40", bd=1, bg="#66BB6A")
best_probabilities.config(font=("Arial", 20, "bold"))
best_probabilities.grid(column=5, row=1, padx=10, pady=5)


bestname_txt = Label(root, text="Name: ", fg="#004D40", bd=1, bg="#66BB6A")
bestname_txt.config(font=("Arial", 20, "bold"))
bestname_txt.grid(column=4, row=2, padx=10, pady=5)

bestname = Label(root, text="None", fg="#004D40", bd=1, bg="#66BB6A")
bestname.config(font=("Arial", 20, "bold"))
bestname.grid(column=5, row=2, padx=10, pady=5)

# Recognition
btn_laplace_and_gau = Button(root, text="Recognition", font=(
    ("Arial"), 10, 'bold'), bg='#43A047', width=14, height=1, fg='#FFFFFF',  command=lambda: recog_image(temp_img, model))
btn_laplace_and_gau.grid(column=0, row=11, pady=5)


btn_laplace_and_gau = Button(root, text="Open Image", font=(
    ("Arial"), 10, 'bold'), bg='#43A047', width=14, height=1, fg='#FFFFFF', command=lambda: select_path())
btn_laplace_and_gau.grid(column=1, row=11, pady=5)


btn_laplace_and_gau = Button(root, text="Open Video", font=(
    ("Arial"), 10, 'bold'), bg='#43A047', width=14, height=1, fg='#FFFFFF', command=lambda: open_video())
btn_laplace_and_gau.grid(column=2, row=11, pady=5)


def select_path():
    global temp_img
    path = filedialog.askopenfilename()
    temp_img = path
    show_image(path)
    return 0


def show_image(path):

    if len(path) > 0:
        # load the image from disk
        img = cv2.imread(path)
        # Convert img to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # convert images to PIL format
        img = Image.fromarray(img)
        # resize Image
        resize_bf = img.resize((720, 576), Image.ANTIALIAS)
        # convert images to ImageTK format
        img_bf = ImageTk.PhotoImage(resize_bf)
        # set image to Label
        image_box.configure(image=img_bf)
        image_box.image = img_bf
    return 0


def recog_image(img_path, model):

    if(len(img_path)>0):

        general_result = []
        img = cv2.imread(img_path, 1)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        faces = face_detector.detectMultiScale(img_gray, 1.2, 2)
        for (x, y, w, h) in faces:
            roi = cv2.resize(img[y: y + h, x: x + w], (64, 64))
            predictions = model.predict(roi.reshape((-1, 64, 64, 3)))
            result = np.argmax(predictions)

            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(
                len(best_class_indices)), best_class_indices]

            general_result.append((result, str(best_class_probabilities[0])))

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
            # cv2.putText(img, lst_resutl[result], (x+15, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # cv2.putText(img, str(best_class_probabilities[0]), (x+15, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imwrite('./asset/image/recog_1.jpg', img)
        show_image('./asset/image/recog_1.jpg')
        show_properties(general_result)

    else:
        print("you can fck select image before recognition!!!, pls")
    return 0


def show_properties(general_result):

    if (len(general_result) < 2):
        best_probabilities.configure(text=lst_resutl[general_result[0][0]])
        bestname.configure(text=general_result[0][1])
    else:
        best_probabilities.configure(
            text = lst_resutl[general_result[0][0]] + ', ' + lst_resutl[general_result[1][0]])
        bestname.configure(text = general_result[0][1] + ', ' + general_result[1][1])

def open_video():

    path = filedialog.askopenfilename()
    video = cv2.VideoCapture(path)
    while True:
        _, frame = video.read()
        faces = face_detector.detectMultiScale(frame, 1.3, 4)
        for (x, y, w, h) in faces:
            roi = cv2.resize(frame[y: y + h, x: x + w], (64, 64))
            predictions = model.predict(roi.reshape((-1, 64, 64, 3)))
            result = np.argmax(predictions)
            # print(result)
            best_class_indices = np.argmax(predictions,axis=1)
            best_class_probabilities = predictions[
                np.arange(len(best_class_indices)), best_class_indices]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.putText(frame, lst_resutl[result], (x+15, y-15),
                    cv2.FONT_ITALIC, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, str(best_class_probabilities[0]), (x+15, y-40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('FRAME', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return 0


root.mainloop()
