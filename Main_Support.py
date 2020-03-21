try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

try:
    import ttk

    py3 = False
except ImportError:
    import tkinter.ttk as ttk

    py3 = True


def set_Tk_var():
    global lfeed_Button
    lfeed_Button = tk.StringVar()
    global file_Button
    file_Button = tk.StringVar()
    global Selcet_Function
    Selcet_Function = tk.StringVar()
    global Set_Threshold
    Set_Threshold = tk.StringVar()
    global selectedButton
    selectedButton = tk.StringVar()


def init(top, gui, *args, **kwargs):
    global w, top_level, root
    w = gui
    top_level = top
    root = top


def destroy_window():
    # Function which closes the window.
    global top_level
    top_level.destroy()
    top_level = None


if __name__ == '__main__':
    import Main

    Main.vp_start_gui()
