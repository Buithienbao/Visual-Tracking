# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!

# Adopt from https://github.com/prateekroy/Computer-Vision/blob/master/HW3/detection_tracking.py

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import cv2
import numpy as np 
from visual_tracking import *

class Ui_MainWindow(object):

    file_path = ''
    global cap
    global frame, frame1, frame2, frame3
    global particle
    global kalman
    target_model_window = "Target Model Window"
    hsv_window = "HSV Histogram"
    roi_box = []
    cur_pos = np.ones(2, dtype=np.int32)
    end_vid = False
    isROI = False
    s_low = 60
    s_high = 255
    v_low = 32
    v_high = 255
    initParticles = True
    initKalman = True
    gotROI = False
    # Spread 200 - 400 Random Particles near to the Object to Track
    frame_count = 0
    pred = np.zeros((2,1), np.float32)
    
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(10, 0, 800, 600))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.loadVidButton = QtWidgets.QPushButton(self.frame)
        self.loadVidButton.setGeometry(QtCore.QRect(160, 40, 89, 25))
        self.loadVidButton.setObjectName("loadVidButton")
        self.runButton = QtWidgets.QPushButton(self.frame)
        self.runButton.setGeometry(QtCore.QRect(370, 40, 89, 25))
        self.runButton.setObjectName("runButton")
        self.stopButton = QtWidgets.QPushButton(self.frame)
        self.stopButton.setGeometry(QtCore.QRect(590, 40, 89, 25))
        self.stopButton.setObjectName("stopButton")
        self.kalLabel = QtWidgets.QLabel(self.frame)
        self.kalLabel.setGeometry(QtCore.QRect(40, 200, 200, 200))
        self.kalLabel.setText("")
        self.kalLabel.setObjectName("kalLabel")
        self.parLabel = QtWidgets.QLabel(self.frame)
        self.parLabel.setGeometry(QtCore.QRect(310, 200, 200, 200))
        self.parLabel.setText("")
        self.parLabel.setObjectName("parLabel")
        self.kalparLabel = QtWidgets.QLabel(self.frame)
        self.kalparLabel.setGeometry(QtCore.QRect(560, 200, 200, 200))
        self.kalparLabel.setText("")
        self.kalparLabel.setObjectName("kalparLabel")
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setGeometry(QtCore.QRect(90, 450, 91, 17))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setGeometry(QtCore.QRect(360, 450, 101, 17))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.frame)
        self.label_3.setGeometry(QtCore.QRect(610, 450, 111, 17))
        self.label_3.setObjectName("label_3")
        self.filePathLabel = QtWidgets.QLineEdit(self.frame)
        self.filePathLabel.setGeometry(QtCore.QRect(160, 90, 521, 25))
        self.filePathLabel.setStyleSheet("background-color:\"transparent\"")
        self.filePathLabel.setFrame(False)
        self.filePathLabel.setReadOnly(True)
        self.filePathLabel.setObjectName("filePathLabel")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.filePathLabel.textChanged.connect(self.getVideo)
        self.runButton.clicked.connect(self.onRunButtonPressed)
        self.stopButton.clicked.connect(self.onStopButtonPressed)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.loadVidButton.setText(_translate("MainWindow", "Load Video"))
        self.runButton.setText(_translate("MainWindow", "Run"))
        self.stopButton.setText(_translate("MainWindow", "Stop"))
        self.label.setText(_translate("MainWindow", "Kalman Filter"))
        self.label_2.setText(_translate("MainWindow", "Particle Filter"))
        self.label_3.setText(_translate("MainWindow", "Kal + Par Filter"))

    
    def setFilePathLabel(self, file_path):
        
        if file_path:
            self.file_path = file_path
            self.filePathLabel.setText(self.file_path)
        else:
            print('Invalid file path')

    
    def getVideo(self, file_path):

        #Load video from file_path to window

        if file_path:

            self.cap = cv2.VideoCapture(file_path)

        else:
            print("File path is empty")
            return
    
    def onStopButtonPressed(self):
        
        self.end_vid = True

    def onMousedragged(self,event, x, y, flags, params):
        # Get the Coordinates
        self.cur_pos[0] = x
        self.cur_pos[1] = y

        if event == cv2.EVENT_LBUTTONDOWN:
            # If Left Mouse Button is Pressed
            self.roi_box = []
            # Start Mouse Position
            start = [x, y]
            self.roi_box.append(start)

        elif event == cv2.EVENT_LBUTTONUP:
            # If Left Mouse Button is Released
            # End Mouse Position
            end = [x, y]
            self.roi_box.append(end)
            top_left = (self.roi_box[0][0], self.roi_box[0][1])
            bottom_right = (self.cur_pos[0], self.cur_pos[1])
            cv2.rectangle(self.frame,top_left, bottom_right, (0,255,0), 2)  
            cv2.imshow(self.target_model_window,self.frame)


    def onRunButtonPressed(self):


        cv2.namedWindow(self.target_model_window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.target_model_window, self.onMousedragged, 0)

        cv2.namedWindow(self.hsv_window,cv2.WINDOW_NORMAL)
        cv2.createTrackbar("S low", self.hsv_window, self.s_low, 255, ignore)
        cv2.createTrackbar("S high", self.hsv_window, self.s_high, 255, ignore)
        cv2.createTrackbar("V low", self.hsv_window, self.v_low, 255, ignore)
        cv2.createTrackbar("V high", self.hsv_window, self.v_high, 255, ignore)

        while not self.end_vid:
            
            if(self.cap.isOpened()):

                ret, self.frame = self.cap.read()
            
                cv2.imshow(self.target_model_window,self.frame)

                key = cv2.waitKey(30) & 0xFF
                
                if key == 32:
                    
                    cv2.waitKey()

                if((len(self.roi_box) > 1) and (self.roi_box[0][1] < self.roi_box[1][1]) and (self.roi_box[0][0] < self.roi_box[1][0])):
                    
                    self.iniTrackingFrame()
                    self.windowHSV()
                    if not self.gotROI:
                        hsv_hist, track_window = self.getROI()
                        self.gotROI = True

                    # Meanshift
                    tracked_window, backproject_img = self.applyMeanShift(hsv_hist,track_window)

                    # Kalman
                    self.applyKalman(tracked_window)

                    # Particle

                    X,Y,X_W,Y_H = self.applyParticle(tracked_window, backproject_img)

                    # Kal-Par
                    self.applyKalPar(X,Y,X_W,Y_H,tracked_window)
                    
                    self.drawKalman()

                    self.drawParticle()
                    
                    self.drawKalPar()
                
                else:
                    
                    hsv_img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)

                    mask = cv2.inRange(hsv_img, np.array((0., float(self.s_low),float(self.v_low))), np.array((180.,float(self.s_high),float(self.v_high))))

                    cv2.imshow(self.hsv_window,mask)



        self.cap.release()
        cv2.destroyAllWindows()

    def drawKalman(self):
        
        kres = cv2.resize(self.frame1, (200, 200))
        kres_rgb = cv2.cvtColor(kres, cv2.COLOR_BGR2RGB)
        h, w, c = kres_rgb.shape
        bytesPerLine = c * w
        kres_qt = QtGui.QImage(kres_rgb, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap = QPixmap(kres_qt)
        self.kalLabel.setPixmap(pixmap)

    def drawParticle(self):

        pres = cv2.resize(self.frame2, (200, 200))
        pres_rgb = cv2.cvtColor(pres, cv2.COLOR_BGR2RGB)
        h, w, c = pres_rgb.shape
        bytesPerLine = c * w
        pres_qt = QtGui.QImage(pres_rgb, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap = QPixmap(pres_qt)
        self.parLabel.setPixmap(pixmap)
    
    def drawKalPar(self):

        kpres = cv2.resize(self.frame3, (200, 200))
        kpres_rgb = cv2.cvtColor(kpres, cv2.COLOR_BGR2RGB)
        h, w, c = kpres_rgb.shape
        bytesPerLine = c * w
        kpres_qt = QtGui.QImage(kpres_rgb, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap = QPixmap(kpres_qt)
        self.kalparLabel.setPixmap(pixmap)

    def applyKalman(self,track_window):
        
        if self.initKalman:

            # Initiate Kalman Filter Object
            self.kalman = cv2.KalmanFilter(4,2)
            self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)

            self.kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)

            self.kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.025

            self.initKalman = False

        x,y,w,h = track_window

        pts = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])

        # Use Extracted Centre for Kalman Correction
        self.kalman.correct(center(pts))

        # Get New Kalman Filter Prediction
        self.pred = self.kalman.predict()

        # Draw Bounding Box over the Predicted Region
        self.frame1 = cv2.rectangle(self.frame1, (self.pred[0]-(0.5*w),self.pred[1]-(0.5*h)), (self.pred[0]+(0.5*w),self.pred[1]+(0.5*h)), (255,0,0),2)


    def applyParticle(self,track_window,backproject_img):

        x,y,w,h = track_window
        
        particle_num = 250
        
        # Track Point for the Initial Frame
        if(self.initParticles):
            
            self.frame_count = self.frame_count + 1

            # Initial pose
            init_pos = np.array([x + w/2.0,y + h/2.0], int)
            
            # Init Particles to init position
            self.particle = np.ones((particle_num, 2), int) * init_pos 
            
            # Weights are uniform at first
            weight = np.ones(particle_num)/particle_num   
            
            self.initParticles = False
        
        # Number of frames
        stepsize = 18;

        # Particle Motion Model: Uniform step
        np.add(self.particle, np.random.uniform(-stepsize, stepsize, self.particle.shape), out=self.particle, casting="unsafe")

        # Clip Out-of-bounds Particles
        self.particle = self.particle.clip(np.zeros(2), np.array((self.frame2.shape[1],self.frame2.shape[0]))-1).astype(int)

        f = particleEvaluator(backproject_img, self.particle.T)

        weight = np.float32(f.clip(1))
        
        # Normalize Weights
        weight /= np.sum(weight) 
        
        # Expected Position: Weighted Average of Particles
        pos = np.sum(self.particle.T * weight, axis=1).astype(int) 

        # Center of ROI
        a = np.int32(pos[0])
        b = np.int32(pos[1]) 

        # Coordinates of the Predicted ROI
        X = np.int32(a-w/2) 
        Y = np.int32(b-h/2) 
        X_W = np.int32(a+w/2) 
        Y_H = np.int32(b+h/2) 

        if 1./np.sum(weight**2) < particle_num/2.:
            
            self.particle = self.particle[resample(weight),:]
    
        self.frame_count = self.frame_count + 1

        # Draw bounding box over ROI predicted by Particle Filter
        self.frame2 = cv2.rectangle(self.frame2, (X,Y), (X_W,Y_H), (0,0,255),2)

        return X,Y,X_W,Y_H
    
    def applyKalPar(self,X,Y,X_W,Y_H,track_window):

        x,y,w,h = track_window

        # Get centre of ROI from Particle Filter
        pts = np.array([[X,Y],[X_W,Y],[X_W,Y_H],[X,Y_H]])

        # Apply Kalman
        self.kalman.correct(center(pts))
        pred_PKF = self.kalman.predict()

        # Draw bounding box over ROI predicted by Particle-Kalman Filter
        self.frame3 = cv2.rectangle(self.frame3, (pred_PKF[0]-(0.5*w),pred_PKF[1]-(0.5*h)), (pred_PKF[0]+(0.5*w),pred_PKF[1]+(0.5*h)), (0,255,0),2)
    
    def applyMeanShift(self,hsv_hist,track_window):

        hsv_img = cv2.cvtColor(self.frame1, cv2.COLOR_BGR2HSV)
        backproject_img = cv2.calcBackProject([hsv_img],[0,1],hsv_hist,[0,180,0,255],1)

        cv2.imshow(self.hsv_window, backproject_img)

        criteria = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
        ret1, track_window = cv2.meanShift(backproject_img, track_window, criteria)

        # Draw bounding box over ROI
        x,y,w,h = track_window
        self.frame1 = cv2.rectangle(self.frame1, (x,y), (x+w,y+h), (255,255,255),2)
        self.frame2 = cv2.rectangle(self.frame2, (x,y), (x+w,y+h), (255,255,255),2)
        self.frame3 = cv2.rectangle(self.frame3, (x,y), (x+w,y+h), (255,255,255),2)
        
        return track_window, backproject_img

    def iniTrackingFrame(self):
        
        _, self.frame1 = self.cap.read()
        _, self.frame2 = self.cap.read()
        _, self.frame3 = self.cap.read()

    def windowHSV(self):

        self.s_low = cv2.getTrackbarPos("S low", self.hsv_window)
        self.s_high = cv2.getTrackbarPos("S high", self.hsv_window)
        self.v_low = cv2.getTrackbarPos("V low", self.hsv_window)
        self.v_high = cv2.getTrackbarPos("V high", self.hsv_window)
        
    def getROI(self):

        crop = self.frame[self.roi_box[0][1]:self.roi_box[1][1],self.roi_box[0][0]:self.roi_box[1][0]].copy()
        
        h, w, c = crop.shape
        
        if (h > 0) and (w > 0):

            # convert from RGB to HSV
            hsvROI =  cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

            # select all Hue from 0 to 180 and Saturation values which is not too low
            mask = cv2.inRange(hsvROI, np.array((0., float(self.s_low),float(self.v_low))), np.array((180.,float(self.s_high),float(self.v_high))))

            # get HSV histogram
            hsv_hist = cv2.calcHist([hsvROI],[0, 1],mask,[180, 255],[0,180, 0, 255])
            # normalize histogram
            cv2.normalize(hsv_hist,hsv_hist,0,255,cv2.NORM_MINMAX)

            # set intial position to track
            track_window = (self.roi_box[0][0],self.roi_box[0][1],self.roi_box[1][0] - self.roi_box[0][0],self.roi_box[1][1] - self.roi_box[0][1])

        return hsv_hist, track_window










