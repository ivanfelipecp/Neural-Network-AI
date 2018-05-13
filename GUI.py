from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
from Controller import Controller
from pygame import mixer # Load the required library

class GUI(object):

    def __init__(self):
        # Variables de la clase
        self.root = Tk()
        self.root.config(bg="black")
        self.controller = Controller()
        self.images_extensions = (("Image files", ("*.jpg","*.png")),)
        self.guitar = "guitar.mp3"


        # Frame de carga
        width = 100
        height = 100
        choose_color = "blue"
        self.choose_frame = Frame(self.root, bg=choose_color,width=width, height=height)
        self.choose_frame.grid(row=0, column=0)

        self.choose_text = "Seleccione un archivo para cargar los pesos"
        self.choose_error = "Eso no es un archivo binario"
        self.choose_label = Label(self.choose_frame,text=self.choose_text,bg=choose_color,fg="white")
        self.choose_label.grid(row=0,column=0)
        self.choose_button = Button(self.choose_frame,text="Browse",command=self.load_file,width=30,bg="white")
        self.choose_button.grid(row=1,column=0)


        # Frame de clasificación
        classify_color = "red"
        self.classify_frame = Frame(self.root, bg="red",width=width, height=height)
        self.classify_frame.grid(row=1, column=0)

        # Mensajes
        self.classify_text = "Seleccione una imágen para clasificar"
        self.classify_error = "Eso no es un archivo binario"

        #Label
        self.classify_label = Label(self.classify_frame,text=self.choose_text,bg=classify_color,fg="white")
        self.classify_label.grid(row=0,column=0)
        # Boton
        self.classify_button = Button(self.classify_frame,text="Cargar y clasificar",command=self.load_and_classify,width=30,bg="white")
        self.classify_button.grid(row=1,column=0)
        # Label
        self.classify_msg = Label(self.classify_frame,text="Es un 69",bg=classify_color,fg="white")
        self.classify_msg.grid(row=2,column=0)


        # Frame de información de la NN
        nn_color = "green"
        self.nn_frame = Frame(self.root,bg=nn_color,width=width, height=height)
        self.nn_frame.grid(row=2, column=0)

        # Labels
        prueba = ""
        self.nn_label = Label(self.nn_frame,text="No se ha cargado la red",bg=nn_color,fg="white")
        self.nn_label.grid(row=0,column=0)
        self.nn_input = Label(self.nn_frame,text=prueba,bg=nn_color,fg="white")
        self.nn_input.grid(row=1,column=0)
        self.nn_output = Label(self.nn_frame,text=prueba,bg=nn_color,fg="white")
        self.nn_output.grid(row=2,column=0)
        self.nn_batch = Label(self.nn_frame,text=prueba,bg=nn_color,fg="white")
        self.nn_batch.grid(row=3,column=0)
        self.nn_learning = Label(self.nn_frame,text=prueba,bg=nn_color,fg="white")
        self.nn_learning.grid(row=4,column=0)
        self.nn_loss = Label(self.nn_frame,text=prueba,bg=nn_color,fg="white")
        self.nn_loss.grid(row=5,column=0)
        self.nn_accuracy = Label(self.nn_frame,text=prueba,bg=nn_color,fg="white")
        self.nn_accuracy.grid(row=6,column=0)

        # Label del titulo
        #self.title = Label(self.root,text="Seleccione el archivo pickle para cargarlo")
        #self.title.grid(row=0,column=0)

        """ funca
        # Boton de elegir pickle
        self.choose = Button(self.root,text="Browse",command=self.load_file,width=30)
        self.choose.grid(row=1,column=0)

        # Mensaje de seleccionado
        self.choose_text = "Seleccione un archivo binario"
        self.choose_error = "Eso no es un archivo binario"
        self.choose_label = Label(self.root,text=self.choose_text)
        self.choose_label.grid(row=1,column=1)

        # Boton de clasificar
        self.classify = Button(self.root,text="Seleccionar imágen y clasificar",command=self.load_image,width=30)
        self.classify.grid(row=2,column=0)


        # Mensaje de clasificado
        self.classify_text = "Seleccione una imágen"
        self.classify_wait = "Clasificando imágen..."
        self.classify_msg = Label(self.root,text=self.classify_text)
        self.classify_msg.grid(row=2,column=1)
        """

    def start(self):
        self.root.mainloop()

    def classify(self,image):
        return self.controller.classify(image)

    def set_file(self,file):
        return self.controller.set_file(file)
        
    def load_file(self):
        file = askopenfilename(filetypes=(("Binary files", "*.*"),))

        if file:
            extension = file.split("/")[-1]
            if not "." in extension:
                # selecciono un binary
                # todo
                text = self.set_file(file)
                self.choose_label.config(text=text)
            else:
                # selecciono uno con extension
                self.choose_label.config(text=self.choose_error)
        else:
            self.choose_label.config(text=self.choose_error)



    def load_and_classify(self):
        image = askopenfilename(filetypes=self.images_extensions)

        if image:
            # selecciono una img
            # todo
            self.classify_msg.config(text=self.classify_wait)
            text = self.classification(image)
            self.classify_msg.config(text=text)
            self.play(self.guitar)
        else:
            # selecciono uno con extension
            self.classify_msg.config(text=self.classify_text)

    def play(self,sound):
        mixer.init()
        mixer.music.load(sound)
        mixer.music.play()

a = GUI()
a.start()