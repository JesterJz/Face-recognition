import os
from tkinter import *
from tkinter import filedialog
from tkinter.font import Font

import cv2 
from PIL import Image, ImageTk
from tkinter import messagebox

root = Tk()
root.title("Face Recognition")
root.geometry("1280x720")
root.wm_resizable(width=True, height=True)
# root.configure(background='#80CBC4')
# root.maxsize(1920, 1080)

img_counter = 0
temp_img = None
# path = None
pic_default = Image.open('./asset/image/icon_default.png')
# convert images to ImageTK format
img_def = ImageTk.PhotoImage(pic_default)

# Set Title Name App
name = Label(root, text="Face Recognition video",
             fg="#000", bd=0)
name.config(font=("Engravers MT", 20))
name.grid(column=0, row=0, columnspan=10, pady=10)

image_box = Label(root, image=img_def, width=720,
                       height=576, bg="#00695C")
image_box.grid(column=0, row=1, columnspan=3, rowspan=10, padx=20, pady=5)


best_probabilities_txt = Label(root, text="Độ chính xác: ", fg="#004D40", bd=1, bg="#66BB6A")
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

# open cam
btn_laplace_and_gau = Button(root, text="Open Cam", font=(
    ("Arial"), 10, 'bold'), bg='#43A047', width=14, height=1, fg='#FFFFFF',  command=lambda: open_cam())
btn_laplace_and_gau.grid(column=0, row=11, pady=5)


btn_laplace_and_gau = Button(root, text="Open Image", font=(
    ("Arial"), 10, 'bold'), bg='#43A047', width=14, height=1, fg='#FFFFFF', command=lambda: select_path())
btn_laplace_and_gau.grid(column=1, row=11, pady=5)


btn_laplace_and_gau = Button(root, text="Open Video", font=(
    ("Arial"), 10, 'bold'), bg='#43A047', width=14, height=1, fg='#FFFFFF', command=lambda: select_path())
btn_laplace_and_gau.grid(column=2, row=11, pady=5)




def select_path():
    global temp_img
    path = filedialog.askopenfilename()

    if len(path) > 0:
        # load the image from disk
        img = cv2.imread(path)
        temp_img = path
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


def open_cam():

    cap = cv2.VideoCapture(0)
    while(True):


        _, frame = cap.read()
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Get the latest frame and convert into Image
        cv2image= cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        # Convert image to PhotoImage
        imgtk = ImageTk.PhotoImage(image = img)
        image_box.imgtk = imgtk
        image_box.configure(image=imgtk)
        # Repeat after an interval to capture continiously
        # image_box.after(20, open_cam)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    cv2.destroyAllWindows()
    return 0




root.mainloop()