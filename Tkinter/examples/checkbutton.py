from Tkinter import *
root = Tk()
root.title("Checkbutton Test")

var1 = StringVar()
var2 = IntVar()

def callCheckbutton():
    print var1.get()
Checkbutton(root, text="Disable",state="disable",width=30).pack()
Checkbutton(root, text="setStatus", variable=var1,
            onvalue="OK", offvalue="NO",command=callCheckbutton,
            height=2,width=20,relief="groove").pack()
Checkbutton(root, text="set solid,anchor", variable=var2,
                height=3,width=30,relief="solid",anchor="sw").pack()


root.mainloop()