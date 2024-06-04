import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
# import qrcode

# def create_qr_code(help_link):
#     qr = qrcode.QRCode(
#         version=1,
#         error_correction=qrcode.constants.ERROR_CORRECT_L,
#         # box_size=10,
#         box_size=5,
#         border=1,
#     )
#     qr.add_data(help_link)
#     qr.make(fit=True)
#     return qr.make_image(fill="black", back_color="white")

def remove_background(image_path, max_value, min_value):

    img = Image.open(image_path)
    img = img.convert("RGBA")
    datas = img.getdata()
    new_data = []
    for item in datas:
        if all([x > min_value and x < max_value for x in item]): 
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)

    img.putdata(new_data)
    img.save("我的签名图片-透明背景.png", "PNG")
    return "我的签名图片-透明背景.png"

def select_image():
    filename = filedialog.askopenfilename()
    if filename:
        img_path.set(filename)

def process():
    output_path = remove_background(img_path.get(), int(max_value.get()), int(min_value.get()))
    messagebox.showinfo("处理完成", f"图片已保存到： {output_path}")


root = tk.Tk()
root.title('签名图片透明化')
root.geometry('300x400+30+30')

img_path = tk.StringVar()
max_value = tk.StringVar()
min_value = tk.StringVar()


min_value.set(10) # 对于白底黑字签名，值越小越不容易有白边
max_value.set(256)
 
# help_link_image = create_qr_code('http://www.weiyoun.com')
# # help_link_photo = ImageTk.PhotoImage(help_link_image,height=10,width=10)
# help_link_photo = ImageTk.PhotoImage(help_link_image,size=(10,10))
# # help_link_image.grid()


filepath_label = tk.Label(root, text="路径").grid(row=5, column=1)
img_path_entry = tk.Entry(root, textvariable=img_path)
img_path_entry.insert(0,r'签名.png')
img_path_entry.grid(row=5, column=2)
button = tk.Button(root, text="选择图片", command=select_image)
button.grid(row=5, column=3)

min_value_label = tk.Label(root, text="最小值").grid(row=10, column=1)
min_value_input = tk.Entry(root, textvariable=min_value).grid(row=10, column=2)
max_value_label = tk.Label(root, text="最大值").grid(row=12, column=1)
max_value_input = tk.Entry(root, textvariable=max_value).grid(row=12, column=2)
button_process = tk.Button(root, text="开始处理", command=process,background='green')
button_process.grid(row=15, column=3)

# hlep_label = tk.Label(root, text="扫码获取帮助").grid(row=20, column=1,columnspan=2)
# img_label = tk.Label(root, image=help_link_photo)
# img_label.grid(row=30, column=1,columnspan=2)


root.mainloop()