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
import SpiderG
import libcamera
from picamera2 import Picamera2

light = robotLight.RobotLight()

pid = PID.PID()
pid.SetKp(0.5)
pid.SetKd(0)
pid.SetKi(0)

CVRun = 1
linePos_1 = 440
linePos_2 = 380
lineColorSet = 255
frameRender = 1
findLineError = 20

colorUpper = np.array([44, 255, 255])
colorLower = np.array([24, 100, 100])

# ============================================================
# AUTONOMOUS DRIVING SETTINGS
# ============================================================


# ============================================================
# YELLOW SIGN - LEFT TURN
# ============================================================

# Yellow color range in HSV
YELLOW_LOWER = np.array([20, 100, 100])
YELLOW_UPPER = np.array([40, 255, 255])


# ============================================================
# PARAM TUNING - Yellow Sign Area
#
# Increase this value to turn closer to the yellow sign.
# Decrease this value to turn earlier.
# ============================================================

YELLOW_TRIGGER_AREA = 30000


# ============================================================
# PARAM TUNING - Left Turn Time
#
# Increase this value for a larger left turn.
# Decrease this value for a smaller left turn.
# ============================================================

LEFT_TURN_TIME = 4.5


# Number of consecutive frames required
# to confirm the yellow sign.
YELLOW_CONFIRM_FRAMES = 3


# The previous yellow sign must become smaller than this value
# before the next yellow sign can be detected.
YELLOW_RELEASE_AREA = 3000


# ============================================================
# RED SIGN - STOP
# ============================================================

# Red is located at both ends of the HSV Hue range,
# so two HSV ranges are used.

RED_LOWER_1 = np.array([0, 100, 100])
RED_UPPER_1 = np.array([10, 255, 255])

RED_LOWER_2 = np.array([170, 100, 100])
RED_UPPER_2 = np.array([179, 255, 255])


# ============================================================
# PARAM TUNING - Red Sign Area
#
# Increase this value to stop closer to the red sign.
# Decrease this value to stop earlier.
# ============================================================

RED_TRIGGER_AREA = 30000


# Number of consecutive frames required
# to confirm the red stop sign.
RED_CONFIRM_FRAMES = 3


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

        # ============================================================
        # Yellow sign detection status
        # ============================================================

        self.yellow_area = 0.0
        self.yellow_x = None
        self.yellow_y = None
        self.yellow_w = None
        self.yellow_h = None

        self.yellow_confirm_count = 0
        self.yellow_latched = False
        self.yellow_status = 'SEARCHING'


        # ============================================================
        # Red stop sign detection status
        # ============================================================

        self.red_area = 0.0
        self.red_x = None
        self.red_y = None
        self.red_w = None
        self.red_h = None

        self.red_confirm_count = 0
        self.red_status = 'SEARCHING'


        # ============================================================
        # Left turn state
        # ============================================================

        self.turning_left = False
        self.turn_end_time = 0.0

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

        # Reset autonomous driving status
        # when AUTO mode starts.
        if invar == 'findlineCV' and self.CVMode != 'findlineCV':

            # Yellow sign status
            self.yellow_area = 0.0
            self.yellow_x = None
            self.yellow_y = None
            self.yellow_w = None
            self.yellow_h = None

            self.yellow_confirm_count = 0
            self.yellow_latched = False
            self.yellow_status = 'SEARCHING'

            # Red stop sign status
            self.red_area = 0.0
            self.red_x = None
            self.red_y = None
            self.red_w = None
            self.red_h = None

            self.red_confirm_count = 0
            self.red_status = 'SEARCHING'

            # Left turn status
            self.turning_left = False
            self.turn_end_time = 0.0

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

            # ========================================================
            # AUTO DRIVE
            # ========================================================

            cv2.putText(
                imgInput,
                'AUTO DRIVE',
                (30, 40),
                CVThread.font,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )


            # ========================================================
            # YELLOW SIGN bounding box
            # ========================================================

            if (
                self.yellow_x is not None
                and self.yellow_y is not None
                and self.yellow_w is not None
                and self.yellow_h is not None
            ):
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


            # ========================================================
            # RED SIGN bounding box
            # ========================================================

            if (
                self.red_x is not None
                and self.red_y is not None
                and self.red_w is not None
                and self.red_h is not None
            ):
                cv2.rectangle(
                    imgInput,
                    (self.red_x, self.red_y),
                    (
                        self.red_x + self.red_w,
                        self.red_y + self.red_h
                    ),
                    (0, 0, 255),
                    2
                )


            # ========================================================
            # YELLOW SIGN DEBUG
            # ========================================================

            cv2.putText(
                imgInput,
                'Yellow Area: %d' % int(self.yellow_area),
                (30, 70),
                CVThread.font,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )

            cv2.putText(
                imgInput,
                'Yellow Trigger: %d' % YELLOW_TRIGGER_AREA,
                (30, 95),
                CVThread.font,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )

            cv2.putText(
                imgInput,
                'Yellow Status: %s' % self.yellow_status,
                (30, 120),
                CVThread.font,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )


            # ========================================================
            # RED SIGN DEBUG
            # ========================================================

            cv2.putText(
                imgInput,
                'Red Area: %d' % int(self.red_area),
                (30, 150),
                CVThread.font,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

            cv2.putText(
                imgInput,
                'Red Trigger: %d' % RED_TRIGGER_AREA,
                (30, 175),
                CVThread.font,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

            cv2.putText(
                imgInput,
                'Red Status: %s' % self.red_status,
                (30, 200),
                CVThread.font,
                0.5,
                (0, 0, 255),
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


    def findLineCtrl(self, posInput, setCenter):#2
        pass
        '''
        # if posInput:
        if posInput > (setCenter + findLineError):
            # move.motorStop()
            #turnRight
            if CVRun:
                move.move(80, 'no', 'right', 0.5)
            else:
                move.move(80, 'no', 'no', 0.5)
            # time.sleep(0.2)
            move.motorStop()
            pass
        elif posInput < (setCenter - findLineError):
            # move.motorStop()
            #turnLeft
            if CVRun:
                move.move(80, 'no', 'left', 0.5)
            else:
                move.move(80, 'no', 'no', 0.5)
            # time.sleep(0.2)
            move.motorStop()
            pass
        else:
            if CVRun:
                move.move(80, 'forward', 'no', 0.5)
            else:
                move.move(80, 'no', 'no', 0.5)
            #forward
            pass
        # else:
        #     pass
        '''

    def findlineCV(self, frame_image):

        # ============================================================
        # AUTO MODE CHECK
        # ============================================================

        if Camera.modeSelect != 'findlineCV':
            self.pause()
            return


        # ============================================================
        # LEFT TURN STATE
        #
        # While the robot is turning left,
        # yellow and red sign detection is NOT performed.
        # ============================================================

        if self.turning_left:

            if time.monotonic() >= self.turn_end_time:

                # Resume forward walking after the left turn.
                if Camera.modeSelect == 'findlineCV':
                    SpiderG.walk('forward')

                self.turning_left = False
                self.yellow_status = 'LOCKED'

                print('[ACTION] LEFT TURN END')
                print('[AUTO] FORWARD RESUMED')

            self.pause()
            return


        # ============================================================
        # Convert camera image to HSV
        # ============================================================

        hsv = cv2.cvtColor(
            frame_image,
            cv2.COLOR_BGR2HSV
        )


        # ============================================================
        # 1. RED STOP SIGN DETECTION
        #
        # RED has the highest priority.
        # ============================================================

        red_mask_1 = cv2.inRange(
            hsv,
            RED_LOWER_1,
            RED_UPPER_1
        )

        red_mask_2 = cv2.inRange(
            hsv,
            RED_LOWER_2,
            RED_UPPER_2
        )

        red_mask = cv2.bitwise_or(
            red_mask_1,
            red_mask_2
        )

        # Remove small noise
        red_mask = cv2.erode(
            red_mask,
            None,
            iterations=2
        )

        red_mask = cv2.dilate(
            red_mask,
            None,
            iterations=2
        )

        red_contours = cv2.findContours(
            red_mask.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )[-2]


        # Reset current red information
        self.red_area = 0.0
        self.red_x = None
        self.red_y = None
        self.red_w = None
        self.red_h = None


        if len(red_contours) > 0:

            largest_red = max(
                red_contours,
                key=cv2.contourArea
            )

            self.red_area = float(
                cv2.contourArea(largest_red)
            )

            (
                self.red_x,
                self.red_y,
                self.red_w,
                self.red_h
            ) = cv2.boundingRect(largest_red)


            # ========================================================
            # PARAM TUNING - Red Sign Area
            # ========================================================

            if self.red_area >= RED_TRIGGER_AREA:

                self.red_confirm_count += 1

                self.red_status = (
                    'CANDIDATE %d/%d'
                    % (
                        self.red_confirm_count,
                        RED_CONFIRM_FRAMES
                    )
                )

                print(
                    '[RED] Candidate %d/%d - Area: %d'
                    % (
                        self.red_confirm_count,
                        RED_CONFIRM_FRAMES,
                        int(self.red_area)
                    )
                )


                # ====================================================
                # RED STOP SIGN CONFIRMED
                # ====================================================

                if (
                    self.red_confirm_count
                    >= RED_CONFIRM_FRAMES
                ):

                    self.red_confirm_count = 0

                    self.red_status = 'STOPPED'
                    self.yellow_status = 'STOPPED'

                    print('[RED] Red stop sign detected!')

                    print(
                        '[RED] Area: %d'
                        % int(self.red_area)
                    )

                    print(
                        '[RED] Trigger Area: %d'
                        % RED_TRIGGER_AREA
                    )

                    print('[ACTION] STOP')


                    # Stop the robot completely.
                    SpiderG.move_init()
                    SpiderG.servoStop()


                    # End autonomous driving mode.
                    Camera.modeSelect = 'none'
                    self.CVMode = 'none'

                    self.pause()
                    return


            else:

                self.red_confirm_count = 0
                self.red_status = 'APPROACHING'


        else:

            self.red_confirm_count = 0
            self.red_status = 'SEARCHING'


        # ============================================================
        # 2. YELLOW SIGN DETECTION
        #
        # Yellow sign = LEFT TURN
        # ============================================================

        yellow_mask = cv2.inRange(
            hsv,
            YELLOW_LOWER,
            YELLOW_UPPER
        )

        # Remove small noise
        yellow_mask = cv2.erode(
            yellow_mask,
            None,
            iterations=2
        )

        yellow_mask = cv2.dilate(
            yellow_mask,
            None,
            iterations=2
        )

        yellow_contours = cv2.findContours(
            yellow_mask.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )[-2]


        # Reset current yellow information
        self.yellow_area = 0.0
        self.yellow_x = None
        self.yellow_y = None
        self.yellow_w = None
        self.yellow_h = None


        # ============================================================
        # NO YELLOW SIGN
        # ============================================================

        if len(yellow_contours) == 0:

            self.yellow_confirm_count = 0

            # Previous yellow sign disappeared.
            # Allow detection of the next yellow sign.
            if self.yellow_latched:

                self.yellow_latched = False

                print(
                    '[YELLOW] Sign cleared - detector armed'
                )

            self.yellow_status = 'SEARCHING'

            self.pause()
            return


        # ============================================================
        # USE THE LARGEST YELLOW AREA
        # ============================================================

        largest_yellow = max(
            yellow_contours,
            key=cv2.contourArea
        )

        self.yellow_area = float(
            cv2.contourArea(largest_yellow)
        )

        (
            self.yellow_x,
            self.yellow_y,
            self.yellow_w,
            self.yellow_h
        ) = cv2.boundingRect(largest_yellow)


        # ============================================================
        # DUPLICATE YELLOW SIGN PREVENTION
        # ============================================================

        if self.yellow_latched:

            self.yellow_confirm_count = 0

            # Rearm only after the previous yellow sign
            # becomes sufficiently small.
            if self.yellow_area < YELLOW_RELEASE_AREA:

                self.yellow_latched = False
                self.yellow_status = 'SEARCHING'

                print(
                    '[YELLOW] Sign cleared - detector armed'
                )

            else:

                self.yellow_status = 'LOCKED'

            self.pause()
            return


        # ============================================================
        # PARAM TUNING - Yellow Sign Area
        #
        # Increase YELLOW_TRIGGER_AREA:
        #     Turn closer to the yellow sign.
        #
        # Decrease YELLOW_TRIGGER_AREA:
        #     Turn earlier.
        # ============================================================

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


            # ========================================================
            # YELLOW SIGN CONFIRMED
            # ========================================================

            if (
                self.yellow_confirm_count
                >= YELLOW_CONFIRM_FRAMES
            ):

                self.yellow_latched = True
                self.yellow_confirm_count = 0
                self.yellow_status = 'TURNING LEFT'

                print('[YELLOW] Yellow sign detected!')

                print(
                    '[YELLOW] Area: %d'
                    % int(self.yellow_area)
                )

                print(
                    '[YELLOW] Trigger Area: %d'
                    % YELLOW_TRIGGER_AREA
                )

                print('[ACTION] LEFT TURN START')

                print(
                    '[ACTION] Turn Time: %.2f sec'
                    % LEFT_TURN_TIME
                )


                # ====================================================
                # PARAM TUNING - Left Turn Time
                #
                # Increase LEFT_TURN_TIME:
                #     Larger left turn.
                #
                # Decrease LEFT_TURN_TIME:
                #     Smaller left turn.
                # ====================================================

                if Camera.modeSelect == 'findlineCV':

                    SpiderG.walk('turnleft')

                    self.turning_left = True

                    self.turn_end_time = (
                        time.monotonic()
                        + LEFT_TURN_TIME
                    )


        else:

            # Yellow sign exists,
            # but it is still too small.
            self.yellow_confirm_count = 0
            self.yellow_status = 'APPROACHING'


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
            else:
                _, img = camera.read()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if Camera.modeSelect == 'none':
                switch.switch(1, 0)

                # Mark CV thread as stopped.
                # This resets autonomous state on the next START.
                cvt.CVMode = 'none'

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