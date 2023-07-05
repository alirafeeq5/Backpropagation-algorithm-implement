import tkinter as tk
from tkinter import *
import pandas as pd
import util
from plots import plot_with_line, pre_plots

def run_gui(run):
    root = Tk()
    root.title('Penguin classification')
    root.geometry("900x500")


    def upd1():
        c=0
        if(c4_v.get()==1):c=c+1
        if(c5_v.get()==1):c=c+1    
        if(c6_v.get()==1):c=c+1
        if(c7_v.get()==1):c=c+1
        if(c8_v.get()==1):c=c+1
        if(c>=2):
            if(c4_v.get()!=1):c4.config(state='disabled')
            if(c5_v.get()!=1):c5.config(state='disabled')
            if(c6_v.get()!=1):c6.config(state='disabled')
            if(c7_v.get()!=1):c7.config(state='disabled')
            if(c8_v.get()!=1):c8.config(state='disabled')
        else:
            c4.config(state='normal')
            c5.config(state='normal')
            c6.config(state='normal')
            c7.config(state='normal')
            c8.config(state='normal')
            
    myLabel= Label(root, text = "please select the two features")
    myLabel.grid(column = 0, row = 0, sticky=W)

    c4_v=tk.IntVar(root)
    c4 = tk.Checkbutton(root, text='bill length', command=upd1, variable = c4_v, onvalue= 1)
    c4.grid(column = 1, row = 1)
    c5_v=tk.IntVar(root)
    c5 = tk.Checkbutton(root, text='bill depth', command=upd1, variable = c5_v, onvalue= 1)
    c5.grid(column = 2, row = 1)
    c6_v=tk.IntVar(root)
    c6 = tk.Checkbutton(root, text='flipper length',command=upd1, variable = c6_v, onvalue= 1)
    c6.grid(column = 3, row = 1)
    c7_v=tk.IntVar(root)
    c7 = tk.Checkbutton(root, text='gender', command=upd1, variable = c7_v, onvalue= 1)
    c7.grid(column = 4, row = 1)
    c8_v=tk.IntVar(root)
    c8 = tk.Checkbutton(root, text='body mass', command=upd1, variable = c8_v, onvalue= 1)
    c8.grid(column = 5, row = 1)

    def upd():
        i=0
        if(c1_v.get()==1):i=i+1
        if(c2_v.get()==1):i=i+1    
        if(c3_v.get()==1):i=i+1
        if(i>=2):
            if(c1_v.get()!=1):c1.config(state='disabled')
            if(c2_v.get()!=1):c2.config(state='disabled')
            if(c3_v.get()!=1):c3.config(state='disabled')
        else:
            c1.config(state='normal')
            c2.config(state='normal')
            c3.config(state='normal')
            
    myLabelC= Label(root, text = "please select the two classes")
    myLabelC.grid(column = 0, row = 2, sticky=W)

    c1_v=tk.IntVar(root)
    c1 = tk.Checkbutton(root, text='Adelie', command=upd, variable = c1_v, onvalue= 1)
    c1.grid(column = 1, row = 3)
    c2_v=tk.IntVar(root)
    c2 = tk.Checkbutton(root, text='Gentoo', command=upd, variable = c2_v, onvalue= 1 )
    c2.grid(column = 2, row = 3)
    c3_v=tk.IntVar(root)
    c3 = tk.Checkbutton(root, text='Chinstrap', command=upd, variable = c3_v, onvalue= 1)
    c3.grid(column = 3, row = 3)

    mx= tk.StringVar(root)


    def Listing():
        featureList=[]
        speciesList=[]
        biasStatus=0
        if(c4_v.get()==1): featureList.append('bill_length_mm')
        if(c5_v.get()==1): featureList.append('bill_depth_mm')
        if(c6_v.get()==1): featureList.append('flipper_length_mm')
        if(c7_v.get()==1): featureList.append('gender')
        if(c8_v.get()==1): featureList.append('body_mass_g')
        if(c1_v.get()==1): speciesList.append('Adelie')
        if(c2_v.get()==1): speciesList.append('Gentoo')
        if(c3_v.get()==1): speciesList.append('Chinstrap')
        
        if(c9_v.get()==1): biasStatus= 1
        
        epochNo= int(T2.get(1.0, "end-1c"))
        learningRate= float(T1.get(1.0, "end-1c"))
        MseThreshold= float(T3.get(1.0, "end-1c"))
        
        print(featureList, speciesList, epochNo, learningRate, biasStatus)
        data= pd.read_csv('penguins.csv')
        util.preprocess(data)
        model = run(data, featureList, speciesList, learningRate, epochNo, biasStatus, MseThreshold)
        mx.set("confusion matrix: " + "\n" f"{model.confusion_matrix[0]}\n{model.confusion_matrix[1]}")
        plot_with_line(model, featureList, speciesList)
        
    def visualiser():
        data= pd.read_csv('penguins.csv')
        util.preprocess(data)
        pre_plots(data)

    myLabelT1 = Label(root, text = "Enter learning rate: ")
    myLabelT1.grid(column = 0, row = 5, sticky=W)
    T1 = Text(root, height = 1, width = 10)
    T1.grid(column = 1, row = 5)

    myLabelT2 = Label(root, text = "Enter number of epochs: ")
    myLabelT2.grid(column = 0, row = 6, sticky=W)
    T2 = Text(root, height = 1, width = 10)
    T2.grid(column = 1, row = 6)

    myLabelT1 = Label(root, text = "Enter MSE threshold: ")
    myLabelT1.grid(column = 0, row = 7, sticky=W)
    T3 = Text(root, height = 1, width = 10)
    T3.grid(column = 1, row = 7)

    myLabel2 = Label(root, textvariable= mx)
    myLabel2.grid(column = 1, row = 12)


    c9_v=tk.IntVar(root)
    c9 = tk.Checkbutton(root, text='Bias', onvalue=1, variable= c9_v)
    c9.grid(column = 3, row = 6)

    myButton = Button( root, text = "run", height= 2,width = 10, command = Listing )
    myButton.grid(column = 1, row = 10)

    myButton1 = Button( root, text = "visualize", height= 2,width = 10, command = visualiser )
    myButton1.grid(column = 3, row = 10)

    root.mainloop()
