import os
import cv2
from base_camera import BaseCamera
import numpy as np
import switch
import datetime
import Kalman_filter
import PID
import time
import threading
import imutils
import robotLight
import libcamera
import SpiderG
from picamera2 import Picamera2

light = robotLight.RobotLight()

pid = PID.PID()
pid.SetKp(0.5)
pid.SetKd(0)
pid.SetKi(0)

CVRun = 1
linePos_1 = 440
linePos_2 = 380

# Black line = 0
lineColorSet = 0

frameRender = 1
findLineError = 20

colorUpper = np.array([44, 255, 255])
colorLower = np.array([24, 100, 100])


# ============================================================
# PARAM TUNING - AUTONOMOUS DRIVING
# ============================================================

# PARAM TUNING - Yellow Sign Area
# Increase this value to turn closer to the yellow sign.
# Decrease this value to turn earlier.
YELLOW_TRIGGER_AREA = 18000


# PARAM TUNING - Right Turn Time
# Increase this value for a larger right turn.
# Decrease this value for a smaller right turn.
RIGHT_TURN_TIME = 2.0


# Number of consecutive frames required to confirm the yellow sign.
YELLOW_CONFIRM_FRAMES = 3

# The detector is armed again after the yellow area becomes smaller
# than this value.
YELLOW_RELEASE_AREA = 3000


# Yellow HSV range.
# Normally, students do not need to change these values.
YELLOW_LOWER = np.array([20, 100, 100])
YELLOW_UPPER = np.array([40, 255, 255])

def get_rpi_os_version():
    try:
        with open('/etc/os-release', 'r') as f:
            content = f.read()
            if 'bookworm' in content.lower():
                return 'bookworm'
            elif 'bullseye' in content.lower():
                return 'bullseye'
            else:
                return 'unknown'
    except Exception as e:
        print(f"Error reading OS version: {e}")
        return 'unknown'

class CVThread(threading.Thread):
    font = cv2.FONT_HERSHEY_SIMPLEX

    kalman_filter_X =  Kalman_filter.Kalman_filter(0.01,0.1)
    kalman_filter_Y =  Kalman_filter.Kalman_filter(0.01,0.1)
    P_direction = 1
    T_direction = 1
    P_servo = 0
    T_servo = 4
    P_anglePos = 0
    T_anglePos = 0
    cameraDiagonalW = 64
    cameraDiagonalH = 48
    videoW = 640
    videoH = 480
    Y_lock = 0
    X_lock = 0
    tor = 17

    switch.switchSetup()

    def __init__(self, *args, **kwargs):
        self.CVThreading = 0
        self.CVMode = 'none'
        self.imgCV = None

        self.mov_x = None
        self.mov_y = None
        self.mov_w = None
        self.mov_h = None

        self.radius = 0
        self.box_x = None
        self.box_y = None
        self.drawing = 0

        self.findColorDetection = 0

        self.left_Pos1 = None
        self.right_Pos1 = None
        self.center_Pos1 = None

        self.left_Pos2 = None
        self.right_Pos2 = None
        self.center_Pos2 = None

        self.center = None

        # Yellow sign detection status
        self.yellow_area = 0.0
        self.yellow_x = None
        self.yellow_y = None
        self.yellow_w = None
        self.yellow_h = None

        self.yellow_confirm_count = 0
        self.yellow_latched = False
        self.yellow_status = 'SEARCHING'

        # Current walking command
        self.drive_command = None

        super(CVThread, self).__init__(*args, **kwargs)
        
        self.__flag = threading.Event()
        self.__flag.clear()

        self.avg = None
        self.motionCounter = 0
        self.lastMovtionCaptured = datetime.datetime.now()
        self.frameDelta = None
        self.thresh = None
        self.cnts = None

    def mode(self, invar, imgInput):
        # Stop walking when leaving autonomous line-following mode.
        if self.CVMode == 'findlineCV' and invar != 'findlineCV':
            self._setWalkCommand('stop')

        # Reset autonomous-driving status when entering line-following mode.
        if invar == 'findlineCV' and self.CVMode != 'findlineCV':
            self.yellow_area = 0.0
            self.yellow_x = None
            self.yellow_y = None
            self.yellow_w = None
            self.yellow_h = None

            self.yellow_confirm_count = 0
            self.yellow_latched = False
            self.yellow_status = 'SEARCHING'

            self.drive_command = None

        self.CVMode = invar
        self.imgCV = imgInput
        self.resume()

    def elementDraw(self,imgInput):
        if self.CVMode == 'none':
            pass

        elif self.CVMode == 'findColor':
            if self.findColorDetection:
                cv2.putText(imgInput,'Target Detected',(40,60), CVThread.font, 0.5,(255,255,255),1,cv2.LINE_AA)
                self.drawing = 1
            else:
                cv2.putText(imgInput,'Target Detecting',(40,60), CVThread.font, 0.5,(255,255,255),1,cv2.LINE_AA)
                self.drawing = 0

            if self.radius > 10 and self.drawing:
                cv2.rectangle(imgInput,(int(self.box_x-self.radius),int(self.box_y+self.radius)),(int(self.box_x+self.radius),int(self.box_y-self.radius)),(255,255,255),1)

        elif self.CVMode == 'findlineCV':
            if frameRender:
                gray = cv2.cvtColor(imgInput, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(
                    gray,
                    0,
                    255,
                    cv2.THRESH_OTSU
                )
                binary = cv2.erode(binary, None, iterations=6)

                # Convert back to BGR so colored debug information
                # can be displayed on the web video.
                imgInput = cv2.cvtColor(
                    binary,
                    cv2.COLOR_GRAY2BGR
                )

            if lineColorSet == 255:
                cv2.putText(
                    imgInput,
                    'Following White Line',
                    (30, 50),
                    CVThread.font,
                    0.5,
                    (128, 255, 128),
                    1,
                    cv2.LINE_AA
                )
            else:
                cv2.putText(
                    imgInput,
                    'Following Black Line',
                    (30, 50),
                    CVThread.font,
                    0.5,
                    (128, 255, 128),
                    1,
                    cv2.LINE_AA
                )

            try:
                if self.left_Pos1 is not None and self.right_Pos1 is not None:
                    cv2.line(
                        imgInput,
                        (self.left_Pos1, linePos_1 + 30),
                        (self.left_Pos1, linePos_1 - 30),
                        (255, 128, 64),
                        1
                    )

                    cv2.line(
                        imgInput,
                        (self.right_Pos1, linePos_1 + 30),
                        (self.right_Pos1, linePos_1 - 30),
                        (64, 128, 255),
                        1
                    )

                if self.left_Pos2 is not None and self.right_Pos2 is not None:
                    cv2.line(
                        imgInput,
                        (self.left_Pos2, linePos_2 + 30),
                        (self.left_Pos2, linePos_2 - 30),
                        (255, 128, 64),
                        1
                    )

                    cv2.line(
                        imgInput,
                        (self.right_Pos2, linePos_2 + 30),
                        (self.right_Pos2, linePos_2 - 30),
                        (64, 128, 255),
                        1
                    )

                cv2.line(
                    imgInput,
                    (0, linePos_1),
                    (640, linePos_1),
                    (255, 255, 64),
                    1
                )

                cv2.line(
                    imgInput,
                    (0, linePos_2),
                    (640, linePos_2),
                    (255, 255, 64),
                    1
                )

                if self.center is not None:
                    center_y = int((linePos_1 + linePos_2) / 2)

                    cv2.line(
                        imgInput,
                        (self.center - 20, center_y),
                        (self.center + 20, center_y),
                        (0, 0, 255),
                        2
                    )

                    cv2.line(
                        imgInput,
                        (self.center, center_y - 20),
                        (self.center, center_y + 20),
                        (0, 0, 255),
                        2
                    )

            except Exception:
                pass

            # Yellow sign bounding box
            if self.yellow_w is not None and self.yellow_h is not None:
                cv2.rectangle(
                    imgInput,
                    (self.yellow_x, self.yellow_y),
                    (
                        self.yellow_x + self.yellow_w,
                        self.yellow_y + self.yellow_h
                    ),
                    (0, 255, 255),
                    2
                )

            # Yellow sign debugging information
            cv2.putText(
                imgInput,
                'Yellow Area: %d' % int(self.yellow_area),
                (30, 75),
                CVThread.font,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )

            cv2.putText(
                imgInput,
                'Trigger Area: %d' % YELLOW_TRIGGER_AREA,
                (30, 95),
                CVThread.font,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )

            cv2.putText(
                imgInput,
                'Yellow: %s' % self.yellow_status,
                (30, 115),
                CVThread.font,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )

        elif self.CVMode == 'watchDog':
            if self.drawing:
                cv2.rectangle(imgInput, (self.mov_x, self.mov_y), (self.mov_x + self.mov_w, self.mov_y + self.mov_h), (128, 255, 0), 1)

        return imgInput


    def watchDog(self, imgInput):
        timestamp = datetime.datetime.now()
        gray = cv2.cvtColor(imgInput, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.avg is None:
            print("[INFO] starting background model...")
            self.avg = gray.copy().astype("float")
            return 'background model'

        cv2.accumulateWeighted(gray, self.avg, 0.5)
        self.frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(self.avg))

        # threshold the delta image, dilate the thresholded image to fill
        # in holes, then find contours on thresholded image
        self.thresh = cv2.threshold(self.frameDelta, 5, 255,
            cv2.THRESH_BINARY)[1]
        self.thresh = cv2.dilate(self.thresh, None, iterations=2)
        self.cnts = cv2.findContours(self.thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        self.cnts = imutils.grab_contours(self.cnts)
        # print('x')
        # loop over the contours
        for c in self.cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < 5000:
                continue
     
            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (self.mov_x, self.mov_y, self.mov_w, self.mov_h) = cv2.boundingRect(c)
            self.drawing = 1
            
            self.motionCounter += 1
            #print(motionCounter)
            #print(text)
            self.lastMovtionCaptured = timestamp
            switch.switch(1,1)
            switch.switch(2,1)
            switch.switch(3,1)
            light.setColor(255,64,0)

        if (timestamp - self.lastMovtionCaptured).seconds >= 0.5:
            self.drawing = 0
            switch.switch(1,0)
            switch.switch(2,0)
            switch.switch(3,0)
            light.setColor(0,64,255)
        self.pause()


    def _setWalkCommand(self, command):
        # Do not send the same walking command repeatedly.
        if command == self.drive_command:
            return

        if command == 'stop':
            SpiderG.servoStop()
        else:
            SpiderG.walk(command)

        self.drive_command = command


    def findLineCtrl(self, posInput, setCenter):
        # Stop if autonomous driving is disabled
        # or if the black line cannot be found.
        if not CVRun or posInput is None:
            self._setWalkCommand('stop')
            return

        # Black line is on the right side.
        if posInput > (setCenter + findLineError):
            self._setWalkCommand('turnright')

        # Black line is on the left side.
        elif posInput < (setCenter - findLineError):
            self._setWalkCommand('turnleft')

        # Black line is near the center.
        else:
            self._setWalkCommand('forward')


    def _checkYellowSign(self, frame_image):
        hsv = cv2.cvtColor(
            frame_image,
            cv2.COLOR_BGR2HSV
        )

        mask = cv2.inRange(
            hsv,
            YELLOW_LOWER,
            YELLOW_UPPER
        )

        mask = cv2.erode(
            mask,
            None,
            iterations=2
        )

        mask = cv2.dilate(
            mask,
            None,
            iterations=2
        )

        cnts = cv2.findContours(
            mask.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )[-2]

        # Reset current-frame information.
        self.yellow_area = 0.0
        self.yellow_x = None
        self.yellow_y = None
        self.yellow_w = None
        self.yellow_h = None

        # No yellow object detected.
        if len(cnts) == 0:
            self.yellow_confirm_count = 0

            if self.yellow_latched:
                self.yellow_latched = False
                print('[YELLOW] Sign cleared - detector armed')

            self.yellow_status = 'SEARCHING'
            return False

        # Use only the largest yellow contour.
        c = max(
            cnts,
            key=cv2.contourArea
        )

        self.yellow_area = float(
            cv2.contourArea(c)
        )

        (
            self.yellow_x,
            self.yellow_y,
            self.yellow_w,
            self.yellow_h
        ) = cv2.boundingRect(c)

        # ----------------------------------------------------
        # Duplicate detection prevention
        # ----------------------------------------------------
        if self.yellow_latched:
            self.yellow_confirm_count = 0

            # Arm the detector again only after
            # the previous yellow sign becomes small enough.
            if self.yellow_area < YELLOW_RELEASE_AREA:
                self.yellow_latched = False
                self.yellow_status = 'SEARCHING'

                print(
                    '[YELLOW] Sign cleared - detector armed'
                )

            else:
                self.yellow_status = 'LOCKED'

            return False

        # ----------------------------------------------------
        # Yellow sign trigger
        # ----------------------------------------------------
        if self.yellow_area >= YELLOW_TRIGGER_AREA:
            self.yellow_confirm_count += 1

            self.yellow_status = (
                'CANDIDATE %d/%d'
                % (
                    self.yellow_confirm_count,
                    YELLOW_CONFIRM_FRAMES
                )
            )

            print(
                '[YELLOW] Candidate %d/%d - Area: %d'
                % (
                    self.yellow_confirm_count,
                    YELLOW_CONFIRM_FRAMES,
                    int(self.yellow_area)
                )
            )

            # Confirm only after consecutive detections.
            if self.yellow_confirm_count >= YELLOW_CONFIRM_FRAMES:
                self.yellow_latched = True
                self.yellow_confirm_count = 0
                self.yellow_status = 'TURNING RIGHT'

                print('[YELLOW] Yellow sign detected!')
                print(
                    '[YELLOW] Area: %d'
                    % int(self.yellow_area)
                )
                print(
                    '[YELLOW] Trigger Area: %d'
                    % YELLOW_TRIGGER_AREA
                )

                print('[ACTION] RIGHT TURN START')
                print(
                    '[ACTION] Turn Time: %.2f sec'
                    % RIGHT_TURN_TIME
                )

                # Turn right.
                self._setWalkCommand('turnright')

                # ====================================================
                # PARAM TUNING - Right Turn Time
                # Increase RIGHT_TURN_TIME for a larger right turn.
                # Decrease RIGHT_TURN_TIME for a smaller right turn.
                # ====================================================
                time.sleep(RIGHT_TURN_TIME)

                # Stop the right-turn command exactly after
                # RIGHT_TURN_TIME.
                self._setWalkCommand('stop')

                print('[ACTION] RIGHT TURN END')
                print('[AUTO] Line tracking resumed')

                return True

        else:
            self.yellow_confirm_count = 0
            self.yellow_status = 'APPROACHING'

        return False


    def findlineCV(self, frame_image):

        # ============================================================
        # 1. Check the yellow sign first.
        # ============================================================
        #
        # PARAM TUNING - Yellow Sign Area
        #
        # Increase YELLOW_TRIGGER_AREA:
        #     Robot approaches closer before turning.
        #
        # Decrease YELLOW_TRIGGER_AREA:
        #     Robot turns earlier.
        #
        # ============================================================
        if self._checkYellowSign(frame_image):
            self.pause()
            return

        # ============================================================
        # 2. Black line detection
        # ============================================================
        frame_findline = cv2.cvtColor(
            frame_image,
            cv2.COLOR_BGR2GRAY
        )

        _, frame_findline = cv2.threshold(
            frame_findline,
            0,
            255,
            cv2.THRESH_OTSU
        )

        frame_findline = cv2.erode(
            frame_findline,
            None,
            iterations=6
        )

        colorPos_1 = frame_findline[linePos_1]
        colorPos_2 = frame_findline[linePos_2]

        self.center = None
        centers = []

        try:
            lineIndex_Pos1 = np.where(
                colorPos_1 == lineColorSet
            )[0]

            lineIndex_Pos2 = np.where(
                colorPos_2 == lineColorSet
            )[0]

            # Upper scan line
            if lineIndex_Pos1.size > 0:
                self.left_Pos1 = int(
                    lineIndex_Pos1[0]
                )

                self.right_Pos1 = int(
                    lineIndex_Pos1[-1]
                )

                self.center_Pos1 = int(
                    (
                        self.left_Pos1
                        + self.right_Pos1
                    ) / 2
                )

                centers.append(
                    self.center_Pos1
                )

            else:
                self.left_Pos1 = None
                self.right_Pos1 = None
                self.center_Pos1 = None

            # Lower scan line
            if lineIndex_Pos2.size > 0:
                self.left_Pos2 = int(
                    lineIndex_Pos2[0]
                )

                self.right_Pos2 = int(
                    lineIndex_Pos2[-1]
                )

                self.center_Pos2 = int(
                    (
                        self.left_Pos2
                        + self.right_Pos2
                    ) / 2
                )

                centers.append(
                    self.center_Pos2
                )

            else:
                self.left_Pos2 = None
                self.right_Pos2 = None
                self.center_Pos2 = None

            # Average the detected centers.
            if centers:
                self.center = int(
                    sum(centers) / len(centers)
                )

        except Exception:
            self.center = None

        # ============================================================
        # 3. Spider walking control
        # ============================================================
        self.findLineCtrl(
            self.center,
            320
        )

        self.pause()


    def servoMove(ID, Dir, errorInput):
        if ID == CVThread.P_servo:
            errorGenOut = CVThread.kalman_filter_X.kalman(errorInput)
            CVThread.P_anglePos += 0.05*(errorGenOut*Dir)*CVThread.cameraDiagonalW/CVThread.videoW

            if abs(errorInput) > CVThread.tor:
                CVThread.scGear.moveAngle(ID,CVThread.P_anglePos)
                CVThread.X_lock = 0
            else:
                CVThread.X_lock = 1
        elif ID == CVThread.T_servo:
            errorGenOut = CVThread.kalman_filter_Y.kalman(errorInput)
            CVThread.T_anglePos += 0.05*(errorGenOut*Dir)*CVThread.cameraDiagonalH/CVThread.videoH

            if abs(errorInput) > CVThread.tor:
                CVThread.scGear.moveAngle(ID,CVThread.T_anglePos)
                CVThread.Y_lock = 0
            else:
                CVThread.Y_lock = 1
        else:
            print('No servoPort %d assigned.'%ID)

    def findColor(self, frame_image):
        hsv = cv2.cvtColor(frame_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, colorLower, colorUpper)#1
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None
        if len(cnts) > 0:
            self.findColorDetection = 1
            c = max(cnts, key=cv2.contourArea)
            ((self.box_x, self.box_y), self.radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            X = int(self.box_x)
            Y = int(self.box_y)
            error_Y = 240 - Y
            error_X = 320 - X
            # CVThread.servoMove(CVThread.P_servo, CVThread.P_direction, error_X)
            # CVThread.servoMove(CVThread.T_servo, CVThread.T_direction, error_Y)

            if CVThread.X_lock == 1 and CVThread.Y_lock == 1:
                switch.switch(1,1)
                switch.switch(2,1)
                switch.switch(3,1)
                light.setColor(255,64,0)
            else:
                switch.switch(1,0)
                switch.switch(2,0)
                switch.switch(3,0)
                light.setColor(0,64,255)
        else:
            self.findColorDetection = 0
            # move.motorStop()
        self.pause()


    def pause(self):
        self.__flag.clear()

    def resume(self):
        self.__flag.set()

    def run(self):
        while 1:
            self.__flag.wait()
            if self.CVMode == 'none':
                continue
            elif self.CVMode == 'findColor':
                self.CVThreading = 1
                self.findColor(self.imgCV)
                self.CVThreading = 0
            elif self.CVMode == 'findlineCV':
                self.CVThreading = 1
                self.findlineCV(self.imgCV)
                self.CVThreading = 0
            elif self.CVMode == 'watchDog':
                self.CVThreading = 1
                self.watchDog(self.imgCV)
                self.CVThreading = 0
            pass


class Camera(BaseCamera):
    video_source = 0
    modeSelect = 'none'
    # modeSelect = 'findlineCV'
    # modeSelect = 'findColor'
    # modeSelect = 'watchDog'


    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()


    def colorFindSet(self, invarH, invarS, invarV):
        global colorUpper, colorLower
        HUE_1 = invarH+15
        HUE_2 = invarH-15
        if HUE_1>180:HUE_1=180
        if HUE_2<0:HUE_2=0

        SAT_1 = invarS+150
        SAT_2 = invarS-150
        if SAT_1>255:SAT_1=255
        if SAT_2<0:SAT_2=0

        VAL_1 = invarV+150
        VAL_2 = invarV-150
        if VAL_1>255:VAL_1=255
        if VAL_2<0:VAL_2=0

        colorUpper = np.array([HUE_1, SAT_1, VAL_1])
        colorLower = np.array([HUE_2, SAT_2, VAL_2])
        print('HSV_1:%d %d %d'%(HUE_1, SAT_1, VAL_1))
        print('HSV_2:%d %d %d'%(HUE_2, SAT_2, VAL_2))
        print(colorUpper)
        print(colorLower)

    def modeSet(self, invar):
        Camera.modeSelect = invar

    def CVRunSet(self, invar):
        global CVRun
        CVRun = invar

    def linePosSet_1(self, invar):
        global linePos_1
        linePos_1 = invar

    def linePosSet_2(self, invar):
        global linePos_2
        linePos_2 = invar

    def colorSet(self, invar):
        global lineColorSet
        lineColorSet = invar

    def randerSet(self, invar):
        global frameRender
        frameRender = invar

    def errorSet(self, invar):
        global findLineError
        findLineError = invar

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        os_version = get_rpi_os_version()
        if os_version == "bookworm":
            camera = Picamera2() 
            camera.start()
        else:    
            camera = cv2.VideoCapture(Camera.video_source)

        cvt = CVThread()
        cvt.start()

        while True:

            # read current frame
            if os_version == "bookworm":
                img = camera.capture_array()
                img = cv2.cvtColor(
                    img,
                    cv2.COLOR_RGB2BGR
                )

            else:
                ret, img = camera.read()

                if not ret:
                    continue
            if Camera.modeSelect == 'none':
                switch.switch(1, 0)

                # Stop autonomous driving completely.
                if cvt.CVMode == 'findlineCV':
                    cvt._setWalkCommand('stop')

                cvt.CVMode = 'none'

                cvt.yellow_confirm_count = 0
                cvt.yellow_latched = False
                cvt.yellow_status = 'SEARCHING'
                cvt.drive_command = None

                cvt.pause()
            else:
                if cvt.CVThreading:
                    pass
                else:
                    cvt.mode(Camera.modeSelect, img)
                    cvt.resume()
                try:
                    img = cvt.elementDraw(img)
                except:
                    pass
            


            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', img)[1].tobytes()