from PyQt5.QtWidgets import QHBoxLayout, QLineEdit, QMainWindow, QApplication, QLabel, QGridLayout, QWidget, QVBoxLayout, QPushButton, QColorDialog, QSlider, QMessageBox
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCursor, QPixmap
import matplotlib.pyplot as plt
import numpy as np
import sys

class cnn():
    
    def __init__(self):
        # init weights, biases and lowest loss
        try:
            self.W1 = np.load("training/W1.npy")
            self.B1 = np.load("training/B1.npy")
            self.W2 = np.load("training/W2.npy")
            self.B2 = np.load("training/B2.npy")
            self.W3 = np.load("training/W3.npy")
            self.B3 = np.load("training/B3.npy")
            self.L = np.load("training/L.npy")

        except FileNotFoundError:
            self.W1 = np.random.randn(32, 1, 3, 3) * np.sqrt(2/(1*3*3))
            self.B1 = np.zeros(32)
            self.W2 = np.random.randn(64, 32, 3, 3) * np.sqrt(2/(32*3*3))
            self.B2 = np.zeros(64)
            self.W3 = np.random.randn(10, 3136) * np.sqrt(2 / 3136)
            self.B3 = np.zeros(10)
            self.L = np.inf

    def train(self, image_filepath, label_filepath, epochs, epoch_size, batch_size, learning_rate, adam_parameter1, adam_parameter2, reinit_values):
        # init training data
        with open(image_filepath, "rb") as f:
            f.read(16)
            image_data = np.frombuffer(f.read(), dtype=np.uint8)
            image_data = image_data.reshape(-1, 28, 28)

        # init training datas labels
        with open(label_filepath, "rb") as f:
            f.read(8)
            label_data = np.frombuffer(f.read(), dtype=np.uint8)
        
        # if epoch size is supposed to be bigger than data, return
        if epoch_size > label_data.shape[0]:
            print(f"Epoch size ({epoch_size}) can't be bigger than data size ({label_data.shape[0]})!")

        # normalize data (values between 0 and 1)
        image_data = image_data / 255.0

        # init convolutions and biases
        if not reinit_values:
            W1 = self.W1
            B1 = self.B1
            W2 = self.W2
            B2 = self.B2
            W3 = self.W3
            B3 = self.B3

        # if reinit_values == True, get random weights and biases (to not overfit model)
        else:
            W1 = np.random.randn(32, 1, 3, 3) * np.sqrt(2/(1*3*3))
            B1 = np.zeros(32)
            W2 = np.random.randn(64, 32, 3, 3) * np.sqrt(2/(32*3*3))
            B2 = np.zeros(64)
            W3 = np.random.randn(10, 3136) * np.sqrt(2 / 3136)
            B3 = np.zeros(10)

        # init momentum values for adam optimizer
        W1_m_prev, W1_v_prev = (np.zeros_like(W1),) * 2
        B1_m_prev, B1_v_prev = (np.zeros_like(B1),) * 2
        W2_m_prev, W2_v_prev = (np.zeros_like(W2),) * 2
        B2_m_prev, B2_v_prev = (np.zeros_like(B2),) * 2
        W3_m_prev, W3_v_prev = (np.zeros_like(W3),) * 2
        B3_m_prev, B3_v_prev = (np.zeros_like(B3),) * 2

        # init lowest cross entropy
        lowest_cross_entropy = self.L

        # training epochs
        for epoch in range(epochs):

            # print statement to keep track of epoch
            print(f"Epoch {epoch} started")

            # shuffle data
            rand_idx = np.random.permutation(len(image_data))
            image_data = image_data[rand_idx]
            label_data = label_data[rand_idx]
            
            # get amount of possible batches from shuffled data
            possible_batches = epoch_size//batch_size

            # assign data to batches
            image_batches = []
            label_batches = []
            for batch_idx in range(possible_batches):
                image_batches.append(image_data[batch_idx*batch_size:batch_size*(batch_idx+1)])
                label_batches.append(label_data[batch_idx*batch_size:batch_size*(batch_idx+1)])

            # train through every batch 
            for t in range(len(label_batches)):

                # forward
                Y = self.one_hot(label_batches[t], batch_size)
                A0 = image_batches[t][:, np.newaxis, :, :]
                Z1 = self.apply_conv_layer(A0, W1, B1)
                A1 = self.ReLU(Z1)
                P1, P1_info = self.max_pool(A1)
                Z2 = self.apply_conv_layer(P1, W2, B2)
                A2 = self.ReLU(Z2)
                P2, P2_info = self.max_pool(A2)
                F2 = self.flatten_batch(P2)
                Z3 = self.apply_dense_layer(F2, W3, B3)
                A3 = self.softmax(Z3)
                
                # save values if new lowest cross entropy
                L = self.cross_entropy(A3, Y)
                if L < lowest_cross_entropy:
                    lowest_cross_entropy = L
                    np.save("training/L.npy",L)
                    np.save("training/W1.npy",W1)
                    np.save("training/B1.npy",B1)
                    np.save("training/W2.npy",W2)
                    np.save("training/B2.npy",B2)
                    np.save("training/W3.npy",W3)
                    np.save("training/B3.npy",B3)
                    print("saved new w&b")

                # print statement to keep track of iteration
                print(f"iteration: {t+1}, loss: {L}")

                # backprop
                dZ3 = A3 - Y
                dF2 = dZ3 @ W3
                dW3 = dZ3.T @ F2
                dB3 = np.sum(dZ3, axis=0)
                dP2 = dF2.reshape(*P2.shape) 
                dA2 = self.revert_max_pooling(dP2, P2_info)
                dZ2 = dA2 * self.ReLU_prime(Z2)
                dP1 = self.backprop_convolution(dZ2, W2)
                dW2 = self.backprop_weights(dP1, dZ2)
                dB2 = np.sum(dZ2, axis=(0, 2, 3))
                dA1 = self.revert_max_pooling(dP1, P1_info)
                dZ1 = dA1 * self.ReLU_prime(Z1)
                dA0 = self.backprop_convolution(dZ1, W1)
                dW1 = self.backprop_weights(dA0, dZ1)
                dB1 = np.sum(dZ1, axis=(0, 2, 3))
                
                # adjust weights and biases
                W1, W1_m_prev, W1_v_prev = self.adam(W1, dW1, W1_m_prev, W1_v_prev, learning_rate, adam_parameter1, adam_parameter2, t+1)
                B1, B1_m_prev, B1_v_prev = self.adam(B1, dB1, B1_m_prev, B1_v_prev, learning_rate, adam_parameter1, adam_parameter2, t+1)
                W2, W2_m_prev, W2_v_prev = self.adam(W2, dW2, W2_m_prev, W2_v_prev, learning_rate, adam_parameter1, adam_parameter2, t+1)
                B2, B2_m_prev, B2_v_prev = self.adam(B2, dB2, B2_m_prev, B2_v_prev, learning_rate, adam_parameter1, adam_parameter2, t+1)
                W3, W3_m_prev, W3_v_prev = self.adam(W3, dW3, W3_m_prev, W3_v_prev, learning_rate, adam_parameter1, adam_parameter2, t+1)
                B3, B3_m_prev, B3_v_prev = self.adam(B3, dB3, B3_m_prev, B3_v_prev, learning_rate, adam_parameter1, adam_parameter2, t+1)

    def predict(self, matrix):
        # use training data to predict number
        A0 = matrix[np.newaxis, np.newaxis, :, :]
        Z1 = self.apply_conv_layer(A0, self.W1, self.B1)
        A1 = self.ReLU(Z1)
        P1, _ = self.max_pool(A1)
        Z2 = self.apply_conv_layer(P1, self.W2, self.B2)
        A2 = self.ReLU(Z2)
        P2, _ = self.max_pool(A2)
        F2 = self.flatten_batch(P2)
        Z3 = self.apply_dense_layer(F2, self.W3, self.B3)
        A3 = self.softmax(Z3)
        return np.argmax(A3, axis=1)[0]

    def adam(self, param, dparam, m_prev, v_prev, learning_rate, par1, par2, t):
        # use adam optimizer to adjust eights and biases
        m = par1*m_prev+(1-par1)*dparam
        v = par2*v_prev+(1-par2)*(dparam**2)
        m_hat = m/(1-par1**t)
        v_hat = v/(1-par2**t)
        return param - learning_rate*(m_hat/(np.sqrt(v_hat)+1e-8)), m, v

    def revert_max_pooling(self, A, mask):
        # undo max pooling
        N, C, H, W = A.shape
        out_H = H*2
        out_W = W*2
        output = np.zeros((N,C,out_H,out_W))
        sN, sC, sH, sW = output.strides
        patches = np.lib.stride_tricks.as_strided(
                output,
                shape = (N, C, H, W, 2, 2),
                strides = (sN, sC, sH * 2, sW *2, sH, sW)
        )
        patches += mask * A[..., None, None]
        return output

    
    def cross_entropy(self, pred, onehot):
        # get cross entropy loss
        pred = np.clip(pred, 1e-12, 1 - 1e-12)
        return -np.mean(np.sum(onehot * np.log(pred), axis=1))


    def one_hot(self, labels, batch_size, num_classes=10):
        # turn 1d array into 2d onehot
        output = np.zeros((batch_size, num_classes))
        for i in range(batch_size):
            output[i, labels[i]] = 1
        return output

    def flatten_batch(self, A):
        # flatten batch while maintaining batch dim
        return A.reshape(A.shape[0], -1)

    def softmax(self, A):
        # returns a percentage
        exp_A = np.exp(A)
        return exp_A / np.sum(exp_A, axis=1, keepdims=True)

    def display_image(self, matrix):
        # display matrix grayscale using matplotlib
        plt.imshow(matrix, cmap="gray")
        plt.show()

    def apply_dense_layer(self, A, W, B):
        # matrix multiplication and bias addition for flattened layer
        return A @ W.T + B

    def apply_conv_layer(self, A, W, B):
        # use im2col and apply convolutions    
        N, C, in_H, in_W = A.shape
        oC, _, k_size, _ = W.shape
        p = (k_size-1)//2
        padded_A = np.pad(A, ((0,0),(0,0),(p,p),(p,p)), mode="constant", constant_values=0)
        sN, sC, sH, sW = padded_A.strides
        patches = np.lib.stride_tricks.as_strided(
                padded_A,
                (N, C, in_H, in_W, k_size, k_size),
                (sN, sC, sH, sW, sH, sW)
        )
        patches = patches.transpose(0, 2, 3, 1, 4, 5).reshape(N, in_H * in_W, C * k_size *k_size)
        flattened_ker = W.reshape(oC, C*k_size*k_size)
        applied_convolution = np.dot(patches, flattened_ker.T).transpose(0, 2, 1).reshape(N, oC, in_H, in_W)
        return applied_convolution + B[:, None, None]

    def backprop_convolution(self, A, W):
        # backpropagate convolutional layer using im2col
        W_rot = np.rot90(W, 2).transpose(1,0,2,3)
        N, C, in_H, in_W = A.shape
        oC, _, k_size, _ = W_rot.shape
        p = (k_size-1)//2
        padded_A = np.pad(A, ((0,0),(0,0),(p,p),(p,p)), mode="constant", constant_values=0)
        sN, sC, sH, sW = padded_A.strides
        patches = np.lib.stride_tricks.as_strided(
                padded_A,
                (N, C, in_H, in_W, k_size, k_size),
                (sN, sC, sH, sW, sH, sW)
        )
        patches = patches.transpose(0, 2, 3, 1, 4, 5).reshape(N, in_H * in_W, C * k_size *k_size)
        flattened_ker = W_rot.reshape(oC, C*k_size*k_size)
        return np.dot(patches, flattened_ker.T).transpose(0, 2, 1).reshape(N, oC, in_H, in_W)
    
    def backprop_weights(self, A_prev, dZ, k=3):
        # backpropagate weights using im2col
        N, C_in, H, W = A_prev.shape
        N, C_out, H_out, W_out = dZ.shape
        p = (k-1)//2
        padded_A = np.pad(A_prev, ((0,0),(0,0),(p,p),(p,p)), mode="constant", constant_values=0)
        sN, sC, sH, sW = padded_A.strides
        patches = np.lib.stride_tricks.as_strided(
                padded_A,
                (N, C_in, H, W, k, k),
                (sN, sC, sH, sW, sH, sW)
        )
        patches = patches.transpose(0, 2, 3, 1, 4, 5).reshape(N * H_out * W_out, -1)
        dZ_flat = dZ.transpose(0, 2, 3, 1).reshape(-1, C_out)
        return np.dot(dZ_flat.T, patches).reshape(C_out, C_in, k, k)

    def max_pool(self, A, k_size=2):
        # pool patches out of batch 
        sN, sC, sH, sW = A.strides
        N, C, H, W = A.shape
        out_H = H//k_size
        out_W = W//k_size
        patches = np.lib.stride_tricks.as_strided(
                A,
                (N, C, out_H, out_W, k_size, k_size),
                (sN, sC, sH * k_size, sW * k_size, sH, sW)
        )
        max_vals = np.max(patches, axis=(-2,-1))
        max_mask = (patches == max_vals[..., None, None])
        return max_vals, max_mask
        
    def ReLU(self, input):
        # ReLU activation function
        return np.maximum(0, input)

    def ReLU_prime(self, input):
        # Derivative of ReLU function
        return (input > 0).astype(float)

    def test(self, image_filepath, label_filepath):
        # test accuracy of fit
        with open(image_filepath, "rb") as f:
            f.read(16)
            image_data = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28) / 255.0

        with open(label_filepath, "rb") as f:
            f.read(8)
            label_data = np.frombuffer(f.read(), dtype=np.uint8)
        
        A0 = image_data[:, np.newaxis, :, :]
        Z1 = self.apply_conv_layer(A0, self.W1, self.B1)
        A1 = self.ReLU(Z1)
        P1, _ = self.max_pool(A1)
        Z2 = self.apply_conv_layer(P1, self.W2, self.B2)
        A2 = self.ReLU(Z2)
        P2, _ = self.max_pool(A2)
        F2 = self.flatten_batch(P2)
        Z3 = self.apply_dense_layer(F2, self.W3, self.B3)
        A3 = self.softmax(Z3)

        guesses = np.argmax(A3, axis=1)

        return np.mean((label_data == guesses).astype(int))


class MainWindow(QMainWindow):
    
    def __init__(self, resolution, brush_size, frame_size):
        super().__init__()
        
        self.cnn = cnn()

        #! Changeable Variables
        self.pixels = (2**(resolution-1))*28 # makes the amount of pixels fit the cnn model so whatever the resolution is, it can maxpool until its 28x28
        self.resolution = resolution
        self.brush_size = brush_size 
        self.frame_size = frame_size
        self.color = "gray" #standard color
        
        #*init variables
        self.Settingswindow_isShown = False
        self.mousePressed = False
        self.method_isLoading = False
        self.brush_activated = True
        self.pressed_keys = set()
        self.undo_version = 1
        self.undo_stack = []
        self.pixel_labels = {}
        self.last_save = {}
        
        self.app = app #make it usuable for everything in MainWindow 
        self.initUI()     
            
    def initUI(self):
        self.setWindowTitle("Draw")
        
        #*init layout
        self.setFixedSize(self.frame_size, self.frame_size)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.grid_layout = QGridLayout()
        self.grid_layout.setContentsMargins(0,0,0,0)
        self.grid_layout.setSpacing(0)
        self.central_widget.setLayout(self.grid_layout)
        
        #*init pixels
        for i in range(self.pixels):
            for j in range(self.pixels):
                label = QLabel(self)
                label.setStyleSheet("background-color: white;")
                self.grid_layout.addWidget(label, i, j)
                self.pixel_labels[(i, j)] = label
                self.last_save[(i, j, self.undo_version)] = label.styleSheet() 
                
                
        #*init settingswindow
        self.Settingswindow = QMainWindow()
        self.Settingswindow.setFixedSize(500,300)
        self.Settingswindow.setWindowFlag(Qt.FramelessWindowHint)
        self.central_widget_Settingswindow = QWidget() #layout
        self.Settingswindow.setCentralWidget(self.central_widget_Settingswindow)
        self.layout_Settingswindow = QVBoxLayout()
        self.central_widget_Settingswindow.setLayout(self.layout_Settingswindow)
        
        #*init savewindow
        self.Savewindow = QMainWindow()
        self.Savewindow.setFixedSize(500,300)
        self.central_widget_Savewindow = QWidget() #layout
        self.Savewindow.setCentralWidget(self.central_widget_Savewindow)
        self.layout_Savewindow = QHBoxLayout()
        self.central_widget_Savewindow.setLayout(self.layout_Savewindow)
        
        #*submit button for savewindow
        self.close_Savewindow_button = QPushButton("Save")
        self.layout_Savewindow.addWidget(self.close_Savewindow_button)
        self.close_Savewindow_button.setStyleSheet("background-color: green;font-family: Times New Roman;border-radius: 5px;")
        self.close_Savewindow_button.clicked.connect(self.check_file_name)
        self.close_Savewindow_button.setFixedSize(50,50)
        
        #*savewindow info
        self.savewindow_info = QLabel("Set file name ->")
        self.layout_Savewindow.addWidget(self.savewindow_info)
        self.savewindow_info.setStyleSheet("color: gray;font-family: Times New Roman;font-size: 25px;")
        
        #*Lineedit for savewindow
        self.saveDirectory_lineedit = QLineEdit()
        self.layout_Savewindow.addWidget(self.saveDirectory_lineedit)
        self.saveDirectory_lineedit.setFixedSize(200,50)
        self.saveDirectory_lineedit.setStyleSheet("font-family: Times New Roman;font-size: 25px;")
        self.saveDirectory_lineedit.setAlignment(Qt.AlignCenter)
        
        #*savewindow info for the fileending = .png
        self.savewindow_info_fileending = QLabel(".png")
        self.layout_Savewindow.addWidget(self.savewindow_info_fileending)
        self.savewindow_info_fileending.setFixedSize(50,50)
        self.savewindow_info_fileending.setStyleSheet("color: gray;font-family: Times New Roman;font-size: 25px;")
        
        #*close button for settingswindow
        self.close_Settingswindow_button = QPushButton("X")
        self.layout_Settingswindow.addWidget(self.close_Settingswindow_button)
        self.close_Settingswindow_button.setStyleSheet("background-color: red;font-family: Times New Roman;")
        self.close_Settingswindow_button.clicked.connect(self.close_Settingswindow)
        self.close_Settingswindow_button.setFixedSize(50,50)
        
        #*open color picker button
        self.open_color_picker = QPushButton("Change brush-color")
        self.layout_Settingswindow.addWidget(self.open_color_picker)
        self.open_color_picker.setStyleSheet(f"background-color: {self.color};font-family: Times New Roman;font-size: 35px;border-radius: 15px;border: 2px solid black;") 
        self.open_color_picker.setFixedSize(450,75)
        self.open_color_picker.clicked.connect(self.change_brush_color)
        
        #*brush size info
        self.brush_size_info = QLabel("Brush-size: 5")
        self.layout_Settingswindow.addWidget(self.brush_size_info)
        self.brush_size_info.setStyleSheet("color: black;font-family: Times New Roman;font-size: 35px;")
        
        #*brush size slider
        self.brush_size_slider = QSlider(Qt.Horizontal)
        self.brush_size_slider.setMinimum(1)
        self.brush_size_slider.setMaximum(25)
        self.brush_size_slider.setValue(5)
        self.brush_size_slider.valueChanged.connect(self.update_brush_size)
        self.layout_Settingswindow.addWidget(self.brush_size_slider)
        self.brush_size_slider.setFixedSize(450,50)
        self.brush_size_slider.setStyleSheet("border-radius: 5px;")
    
    def show_Savewindow(self):
            self.pressed_keys.remove(83)       # ↓
            self.pressed_keys.remove(16777249) #somehow needed due to keys not getting removed after this functtion
            self.Savewindow.hide()
            self.Savewindow.show()
        
    def change_brush_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.change_color_picker_button_color(color.name()) #! ↓ Function must be called before ↓
            self.color = color.name() #get string so it can be used for style sheet
            
    def change_color_picker_button_color(self, color):
        #*Function needed so styleSheet doesn't get too long
        
        oldBackground = self.open_color_picker.styleSheet().find(f"background-color: {self.color};")
        #saves new color + everything in style sheet except old color
        newStyleSheet =  self.open_color_picker.styleSheet()[:oldBackground] + self.open_color_picker.styleSheet()[(oldBackground + len(f"background-color: {self.color};")):] + f"background-color: {color};"
        self.open_color_picker.setStyleSheet(newStyleSheet)
        
    def update_brush_size(self, val):
        self.brush_size = val
        self.brush_size_info.setText(f"Brush-size: {val}")
    
    def close_Settingswindow(self):
        self.Settingswindow.hide()
        self.Settingswindow_isShown = False
        
    def show_Settingswindow(self):
        #*Show settingswindow
        if not self.Settingswindow_isShown:
            self.Settingswindow.show()
            self.Settingswindow_isShown = True
    
    def closeEvent(self, event):
        #* Close everything if MainWindow is closed
        self.app.quit()
        self.Settingswindow.close()
    
    def keyPressEvent(self, event):
        key = event.key()
        if not key == Qt.Key_G:
            self.pressed_keys.add(key)
        
        if key == Qt.Key_T:
            if not self.Settingswindow_isShown:
                self.show_Settingswindow()
            else:
                self.close_Settingswindow() #close and open to put it on top of screen again
                self.show_Settingswindow()
                
        elif Qt.Key_Z in self.pressed_keys and Qt.Key_Control in self.pressed_keys:
            if not self.method_isLoading:
                self.method_isLoading = True
                self.undo()
                self.method_isLoading = False
              
        elif Qt.Key_Q in self.pressed_keys and Qt.Key_Control in self.pressed_keys:
            if not self.method_isLoading:
                self.method_isLoading = True
                self.empty_canva()
                self.method_isLoading = False
            
        elif Qt.Key_S in self.pressed_keys and Qt.Key_Control in self.pressed_keys:
            if not self.method_isLoading:
                self.method_isLoading = True
                self.show_Savewindow()
                self.method_isLoading = False
            
        elif Qt.Key_R in self.pressed_keys and Qt.Key_Control in self.pressed_keys:
            if not self.method_isLoading:
                self.method_isLoading = True
                self.rotateImg()
                self.method_isLoading = False
                
        elif Qt.Key_B in self.pressed_keys:
            if not self.brush_activated:
                self.brush_activated = True
        
        elif Qt.Key_E in self.pressed_keys:
            if self.brush_activated:
                self.brush_activated = False
            else:
                self.brush_activated = True

        if Qt.Key_G == key:
            if not self.method_isLoading:
                self.method_isLoading = True

                cols = []
                for i in range(self.pixels):
                    row = []
                    for j in range(self.pixels):
                        row.append(0 if self.pixel_labels[i,j].styleSheet() == "background-color: white;"  else 1)
                    cols.append(row)
                matrix = np.array(cols)[np.newaxis, np.newaxis, :, :]
                for i in range(self.resolution-1):
                    matrix, _ = self.cnn.max_pool(matrix)
                prediction = self.cnn.predict(np.squeeze(matrix))
                msg = QMessageBox()
                msg.setWindowTitle("Prediction")
                msg.setText(f"{prediction}")
                msg.setIcon(QMessageBox.Information)
                msg.exec_()
                self.method_isLoading = False
        
    def save(self, filename):
        saved_image = QPixmap(self.size())
        self.render(saved_image)
        saved_image.save(f"{filename}.png")
        
        self.Savewindow.hide()
    
    def check_file_name(self):
        filename = self.saveDirectory_lineedit.text()
        if len(filename) < 1:
            self.warn_invalid_file_name("null")
            return #if theres no character return
        
        if len(filename) > 255: 
            self.warn_invalid_file_name("length")
            return #if file name is longer than 255 characters return
         
        invalid_characters = ["\\", "/", ":", "*", "?", "\"", "<", ">", "|", " "]   
        for character in filename: #look for invalid characters
            if character in invalid_characters:
                self.warn_invalid_file_name("invalid", character)
                return
                
        self.save(filename) #if file name is ok, save file
                  
    def warn_invalid_file_name(self, reason, char=""):
        if reason == "length":
            self.savewindow_info.setText("Too long")
        elif reason == "invalid":
            self.savewindow_info.setText(f"Cant contain {char}")
        else: #else is called if the length of the file name is lower than 1
            self.savewindow_info.setText(f"Atleast 1 char")
      
    def empty_canva(self):
        self.save_current_state()
        for (row, col), pixels in self.pixel_labels.items():
            self.pixel_labels[(row,col)].setStyleSheet("background-color: white;")    
                
    def undo(self):
        #* Undo last thing drawn
        if self.undo_stack:
            last_state = self.undo_stack.pop()
            for (row, col), style in last_state.items():
                self.pixel_labels[(row, col)].setStyleSheet(style)
                
    def save_current_state(self):
        #* Save current drawing
        current_state = {key: pixel.styleSheet() for key, pixel in self.pixel_labels.items()}
        self.undo_stack.append(current_state)
        
    def keyReleaseEvent(self, event):
        self.pressed_keys.discard(event.key())
    
    def mousePressEvent(self, event):
        if not self.mousePressed:
            self.save_current_state()        
            self.draw()
            self.startDrawing = QtCore.QTimer()
            self.startDrawing.timeout.connect(self.draw)
            self.startDrawing.start(0)
            self.mousePressed = True
        
    def mouseReleaseEvent(self, event):
        if self.mousePressed:
            self.startDrawing.stop()
            self.mousePressed = False
        
    def draw(self):
        local_pos = self.central_widget.mapFromGlobal(QCursor.pos())
        pixel_size_x = self.width() // self.pixels
        pixel_size_y = self.height() // self.pixels
        
        row = local_pos.y() // pixel_size_y
        col = local_pos.x() // pixel_size_x
        
        if self.brush_activated:
            color = self.color
        else:
            color = "white"
        
        #*right pixels
        for i in range(self.brush_size-1): 
            y_coordinate = col + (self.brush_size-1 - i) 
            needed_pixels = i * 2 + 1
            side_pixels = (needed_pixels - 1) // 2

            for j in range(side_pixels):
                if 0 <= row - side_pixels + j < self.pixels and 0 <= y_coordinate < self.pixels:
                    self.pixel_labels[(row - side_pixels + j, y_coordinate)].setStyleSheet(f"background-color: {color};")
                if 0 <= row + 1 + j < self.pixels and 0 <= y_coordinate < self.pixels:
                    self.pixel_labels[(row + 1 + j, y_coordinate)].setStyleSheet(f"background-color: {color};")

            if 0 <= row < self.pixels and 0 <= y_coordinate < self.pixels:
                self.pixel_labels[(row, y_coordinate)].setStyleSheet(f"background-color: {color};")

        #*middle pixels
        for i in range((self.brush_size-1)*2+1):
            if 0 <= row - (self.brush_size-1) + i < self.pixels and 0 <= col < self.pixels:
                self.pixel_labels[(row - (self.brush_size-1) + i, col)].setStyleSheet(f"background-color: {color};")

        #*left pixels
        for i in range(self.brush_size-1):
            y_coordinate = col - (self.brush_size-1 - i) 
            needed_pixels = i * 2 + 1
            side_pixels = (needed_pixels - 1) // 2

            for j in range(side_pixels):
                if 0 <= row - side_pixels + j < self.pixels and 0 <= y_coordinate < self.pixels:
                    self.pixel_labels[(row - side_pixels + j, y_coordinate)].setStyleSheet(f"background-color: {color};")
                if 0 <= row + 1 + j < self.pixels and 0 <= y_coordinate < self.pixels:
                    self.pixel_labels[(row + 1 + j, y_coordinate)].setStyleSheet(f"background-color: {color};")

            if 0 <= row < self.pixels and 0 <= y_coordinate < self.pixels:
                self.pixel_labels[(row, y_coordinate)].setStyleSheet(f"background-color: {color};")

    def rotateImg(self):
        self.save_current_state()
        
        #*Make a copy of current image
        tmp = {key: pixel.styleSheet() for key, pixel in self.pixel_labels.items()}
        self.undo_stack.append(tmp)
        
        size = self.pixels #sidelenght of matrix
        
        #*Make tmp be equal to the rotated image
        for col in range(size):
            for row in range(size):
                tmp[row,col] = self.pixel_labels[size-col-1,row].styleSheet()
        
        #*update image to tmp
        for col in range(size):
            for row in range(size):
                self.pixel_labels[row,col].setStyleSheet(tmp[row,col])
                

if __name__ == "__main__":
    print("")
    model = cnn()

    model.train(
            image_filepath="./train-images.idx3-ubyte",
            label_filepath="./train-labels.idx1-ubyte",
            epochs=3,
            epoch_size=30000,
            batch_size=128,
            learning_rate=0.0001,
            adam_parameter1=0.9,
            adam_parameter2=0.999,
            reinit_values=True
    )

    # model_accuracy = model.test(
    #         "t10k-images.idx3-ubyte",
    #         "t10k-labels.idx1-ubyte"
    # )
    # print(model_accuracy)

    # app = QApplication(sys.argv)
    # Mainwindow = MainWindow(
    #         resolution=2,
    #         brush_size=3,
    #         frame_size=784)
    # Mainwindow.show()
    # sys.exit(app.exec())

    
