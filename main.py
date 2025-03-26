import argparse

import numpy as np
# import psycopg2
import mysql.connector
import cv2
import torch
from PyQt5.QtCore import QPoint
import torch.backends.cudnn as cudnn
from LoginUI import *
from InterfaceUI import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsDropShadowEffect, QFileDialog
import sys
import webbrowser
import random
from models.experimental import attempt_load
from utils.torch_utils import select_device
from utils.general import check_img_size, non_max_suppression, scale_coords

# 全局变量
user_now = ""
camera_url = "rtsp://192.168.43.1:8554/live"
MYSQL_CONN = mysql.connector.connect(host="localhost", port=3306, user="root", password="hfut",
                                           database="wads")


# 该函数实现对图片尺寸的调整
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # img: 原图 hwc
    # new_shape: 缩放后的最长边大小
    # color: pad的颜色
    # auto: True 保证缩放后的图片保持原图的比例 即 将原图最长边缩放到指定大小，再将原图较短边按原图比例缩放（不会失真）
    #       False 将原图最长边缩放到指定大小，再将原图较短边按原图比例缩放,最后将较短边两边pad操作缩放到最长边大小（不会失真）
    # scale_fill: True 简单粗暴的将原图resize到指定的大小 相当于就是resize 没有pad操作（失真）
    # scale_up: True  对于小于new_shape的原图进行缩放,大于的不变
    #           False 对于大于new_shape的原图进行缩放,小于的不变
    # 调整大小和填充图像，同时满足步幅-多重约束
    shape = im.shape[:2]  # 第一层resize后图片大小[h, w] = [343, 512]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)  # (512, 512)

    # scale ratio (new / old)   1.024   new_shape=(384, 512)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # r=1

    # 只进行下采样 因为上采样会让图片模糊
    # (for better test mAP) scale_up = False 对于大于new_shape（r<1）的原图进行缩放,小于new_shape（r>1）的不变
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios (1, 1)
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # wh(512, 343) 保证缩放后图像比例不变
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding  dw=0 dh=41
    if auto:  # minimum rectangle  保证原图比例不变，将图像最大边缩放到指定大小
        # 这里的取余操作可以保证padding后的图片是32的整数倍(416x416)，如果是(512x512)可以保证是64的整数倍
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding dw=0 dh=0
    elif scaleFill:  # stretch 简单粗暴的将图片缩放到指定尺寸
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    # 在较小边的两侧进行pad, 而不是在一侧pad
    dw /= 2  # divide padding into 2 sides  将padding分到上下，左右两侧  dw=0
    dh /= 2  # dh=20.5

    # shape:[h, w]  new_unpad:[w, h]
    if shape[::-1] != new_unpad:  # resize  将原图resize到new_unpad（长边相同，比例相同的新图）
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))  # 计算上下两侧的padding  # top=20 bottom=21
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))  # 计算左右两侧的padding  # left=0 right=0
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    # img: (384, 512, 3) ratio=(1.0,1.0) 这里没有缩放操作  (dw,dh)=(0.0, 20.5)
    return im, ratio, (dw, dh)
    # return: img: letterbox后的图片 HWC
    #              ratio: wh ratios
    #              (dw, dh): w和h的pad


# 该函数实现对图片识别框的配置
def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # x: 预测得到的bounding box  [x1 y1 x2 y2]
    # im: 原图 要将bounding box画在这个图上  array
    # color: bounding box线的颜色
    # labels: 标签上的框框信息  类别 + score
    # line_thickness: bounding box的线宽
    # tl = 框框的线宽  要么等于line_thickness要么根据原图im长宽信息自适应生成一个
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    # c1 = (x1, y1) = 矩形框的左上角   c2 = (x2, y2) = 矩形框的右下角
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # cv2.rectangle: 在im上画出框框   c1: start_point(x1, y1)  c2: end_point(x2, y2)
    # 注意: 这里的c1+c2可以是左上角+右下角  也可以是左下角+右上角都可以
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    # 如果label不为空还要在框框上面显示标签label + score
    if label:
        tf = max(tl - 1, 1)  # label字体的线宽 font thickness
        # cv2.getTextSize: 根据输入的label信息计算文本字符串的宽度和高度
        # 0: 文字字体类型  fontScale: 字体缩放系数  thickness: 字体笔画线宽
        # 返回retval 字体的宽高 (width, height), baseLine 相对于最底端文本的 y 坐标
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        # 同上面一样是个画框的步骤  但是线宽thickness=-1表示整个矩形都填充color颜色
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        # cv2.putText: 在图片上写文本 这里是在上面这个矩形框里写label + score文本
        # (c1[0], c1[1] - 2)文本左下角坐标  0: 文字样式  fontScale: 字体缩放系数
        # [225, 255, 255]: 文字颜色  thickness: tf字体笔画线宽     lineType: 线样式
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


class LoginWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_LoginWindow()
        self.ui.setupUi(self)
        # 设置窗口标志 使窗口无边框
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        # 设置窗口背景透明
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        # 用于添加阴影效果
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setOffset(0, 0)
        self.shadow.setBlurRadius(15)
        self.shadow.setColor(QtCore.Qt.black)
        self.ui.frame.setGraphicsEffect(self.shadow)
        # 按键处理
        self.ui.pushButton_Login.clicked.connect(lambda: self.ui.stackedWidget_2.setCurrentIndex(1))
        self.ui.pushButton_Register.clicked.connect(lambda: self.ui.stackedWidget_2.setCurrentIndex(0))
        self.ui.pushButton_L_Sure.clicked.connect(self.login_in)
        self.ui.pushButton_R_Sure_2.clicked.connect(self.register_in)
        # 记录鼠标按下的初始位置
        self.offset = QPoint()
        # 显示登录窗口
        self.show()

    def register_in(self):
        # 获取账号密码
        account = self.ui.lineEdit_Register_Account.text()  # 获取账号
        password = self.ui.lineEdit_Register_Password.text()  # 获取密码
        # 首先进行非空判断
        if account == "" or password == "":
            self.ui.stackedWidget.setCurrentIndex(4)
        else:
            # 建立空列表存储数据库信息
            account_list = []
            # conn = psycopg2.connect(database="DataMyUI", user="postgres", password="123456", host="127.0.0.1", port=5432)
            conn = MYSQL_CONN
            cur = conn.cursor()
            # 查表
            cur.execute("select * from users ")
            rows = cur.fetchall()
            for row in rows:
                account_list.append(row[0])  # 将数据库的账户存储到账号列表cf
            # 首先判断当前用户是否已经存在
            exsit_flag = False  # 存在标志
            for i in account_list:
                if i == account:
                    exsit_flag = True
                    self.ui.stackedWidget.setCurrentIndex(1)
                    # print("当前用户已存在...")
            # 注册的用户为新用户
            if not exsit_flag:
                # 在数据库进行插入操作
                cur.execute(f"insert into users values ('{account}', '{password}')")
                self.ui.stackedWidget.setCurrentIndex(3)
                # print("注册成功...")
            # 断开数据库连接
            conn.commit()
            cur.close()

    def login_in(self):
        # 获取输入栏的账号密码
        account = self.ui.lineEdit_Login_Account.text()  # 获取账号
        password = self.ui.lineEdit_Login_Password.text()  # 获取密码
        # 首先进行非空判断
        if account == "" or password == "":
            self.ui.stackedWidget.setCurrentIndex(4)
        else:
            # 建立空列表存储数据库信息
            account_list = []
            password_list = []
            # 连接数据库
            # conn = psycopg2.connect(database="DataMyUI", user="postgres", password="123456", host="127.0.0.1", port=5432)
            conn = MYSQL_CONN
            cur = conn.cursor()
            # 获取数据库信息
            cur.execute("select * from users ")
            rows = cur.fetchall()
            for row in rows:
                account_list.append(row[0])  # 将数据库的账户存储到账号列表
                password_list.append(row[1])  # 将数据库的密码存储到密码列表
            # 断开数据库连接
            conn.commit()
            cur.close()
            # 登录的标志
            load_flag = False
            # 遍历账号 密码列表
            for i in range(len(account_list)):
                if account == account_list[i] and password == password_list[i]:
                    # 将当前登录用户记录到全局变量
                    global user_now
                    user_now = account
                    # print("登陆成功...")
                    load_flag = True  # 表示登陆成功
                    self.win = MainWindow()
                    self.close()
            # 如果登陆标志没有被修改 说明没有匹配的账号密码 登陆失败
            if not load_flag:
                # print("登陆失败...")
                self.ui.stackedWidget.setCurrentIndex(2)

    def mousePressEvent(self, event):
        # 记录鼠标按下的初始位置
        self.offset = event.pos()

    def mouseMoveEvent(self, event):
        # 移动窗口位置
        if event.buttons() == QtCore.Qt.LeftButton:
            self.move(self.pos() + event.pos() - self.offset)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # 设置窗口标志 使窗口无边框
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        # 设置窗口背景透明
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        ####################
        self.openfile_name_model = None
        self.ui.timer_video = QtCore.QTimer()  # 用于定时触发事件
        self.ui.timer_video.timeout.connect(self.show_video_frame)  # 当timer_video定时器超时时 调用show_video_frame方法
        self.cap = cv2.VideoCapture()
        self.out = None
        self.num_stop = 1  # 暂停与播放辅助信号 通过奇偶来控制暂停与播放
        self.camera_flag = None  # 用来解决处理视频和处理摄像头时（都是帧处理）开启摄像头按键的矛盾
        ####################
        # 用于添加阴影效果
        # self.shadow = QGraphicsDropShadowEffect(self)
        # self.shadow.setOffset(0, 0)
        # self.shadow.setBlurRadius(15)
        # self.shadow.setColor(QtCore.Qt.black)
        # self.ui.frame.setGraphicsEffect(self.shadow)
        # 按钮处理
        self.ui.pushButton_Logout.clicked.connect(self.logout)
        self.ui.pushButton_Log_Out.clicked.connect(self.logout)
        self.ui.pushButton_Main.clicked.connect(self.go_Main)
        self.ui.pushButton_Me.clicked.connect(self.go_Me)
        self.ui.pushButton_Change_Password.clicked.connect(self.go_change_password)
        self.ui.pushButton_Sure_Change.clicked.connect(self.change_password)
        self.ui.pushButton_Back.clicked.connect(self.go_back)
        self.ui.pushButton_Author.clicked.connect(lambda: webbrowser.open("https://blog.csdn.net/weixin_73858883?spm"
                                                                          "=1000.2115.3001.5343"))
        """功能区"""
        # 选择权重
        self.ui.pushButton_Model_Select.clicked.connect(self.open_model)
        # 模型初始化
        self.ui.pushButton_Model_Init.clicked.connect(self.model_init)
        # 选择图片
        self.ui.pushButton_Import_Pic.clicked.connect(self.import_pic)
        # 选择视频
        self.ui.pushButton_Import_Video.clicked.connect(self.import_video)
        # 视频的暂停和继续
        self.ui.pushButton_Pause.clicked.connect(self.pause_or_continue)
        # 结束视频检测
        self.ui.pushButton_Over.clicked.connect(self.finish_video)
        # 打开 or 关闭摄像头
        self.ui.pushButton_Open_Camera.clicked.connect(self.open_or_close_camera)
        """功能区"""
        # 显示当前登录用户信息
        self.ui.pushButton_User_Now.setText("当前用户：" + user_now)
        # 记录鼠标按下的初始位置
        self.offset = QPoint()
        #  显示窗口
        self.show()

    def open_or_close_camera(self):
        """打开摄像头"""
        # 允许使用摄像头flag 这样在进行帧检测时 不会把按键ban掉
        self.camera_flag = True
        # 判断逻辑
        if not self.ui.timer_video.isActive():
            global camera_url
            camera_url = self.ui.lineEdit_Camera_URL.text()
            flag = self.cap.open(camera_url)
            # 改进cv库延时问题
            # self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 10.0)
            # self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            if not flag:
                QtWidgets.QMessageBox.warning(self, u"Warning", u"打开摄像头失败", buttons=QtWidgets.QMessageBox.Ok,
                                              defaultButton=QtWidgets.QMessageBox.Ok)
                self.ui.textBrowser.append("Failed to open the camera...")
            else:
                self.out = cv2.VideoWriter('prediction.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20,
                                           (int(self.cap.get(3)), int(self.cap.get(4))))
                self.ui.timer_video.start(30)
                self.ui.textBrowser.append("Camera enabled...")
                # 禁止其他按键使用
                self.ui.pushButton_Model_Select.setDisabled(True)
                self.ui.pushButton_Model_Init.setDisabled(True)
                self.ui.pushButton_Import_Pic.setDisabled(True)
                self.ui.pushButton_Import_Video.setDisabled(True)
                self.ui.pushButton_Pause.setDisabled(True)
                self.ui.pushButton_Over.setDisabled(True)
                self.ui.pushButton_Open_Camera.setText(u" 关闭摄像头")
        else:
            self.ui.timer_video.stop()
            self.cap.release()
            self.out.release()
            self.ui.label_Camera_Video_Pic.clear()
            self.ui.textBrowser.append("Camera off...")
            self.ui.pushButton_Model_Select.setDisabled(False)
            self.ui.pushButton_Model_Init.setDisabled(False)
            self.ui.pushButton_Import_Pic.setDisabled(False)
            self.ui.pushButton_Import_Video.setDisabled(False)
            self.ui.pushButton_Pause.setDisabled(False)
            self.ui.pushButton_Over.setDisabled(False)
            self.ui.pushButton_Open_Camera.setText(u" 开启摄像头")

    def finish_video(self):
        """结束视频检测"""
        # 释放资源
        self.cap.release()  # 释放video_capture资源
        self.out.release()  # 释放video_writer资源
        self.ui.label_Camera_Video_Pic.clear()  # 清空label画布
        self.ui.textBrowser.append("Video has ended...")
        # 启动其他检测按键功能
        self.ui.pushButton_Import_Video.setDisabled(False)
        self.ui.pushButton_Import_Pic.setDisabled(False)
        self.ui.pushButton_Open_Camera.setDisabled(False)
        self.ui.pushButton_Model_Init.setDisabled(False)
        self.ui.pushButton_Model_Select.setDisabled(False)
        # 结束检测时，查看暂停功能是否复位，将暂停功能恢复至初始状态
        # Note:点击暂停之后，num_stop为偶数状态
        if self.num_stop % 2 == 0:
            self.ui.pushButton_Pause.setText(u' 暂停')
            self.num_stop += 1
            self.ui.timer_video.blockSignals(False)

    def pause_or_continue(self):
        """对播放的视频进行暂停或继续"""
        self.ui.timer_video.blockSignals(False)
        # 暂停检测
        # 若QTimer已经触发，且激活
        if self.ui.timer_video.isActive() and self.num_stop % 2 == 1:
            self.ui.pushButton_Pause.setText(u' 继续')  # 当前状态为暂停状态
            self.num_stop += 1  # 调整标记信号为偶数
            self.ui.timer_video.blockSignals(True)
            self.ui.textBrowser.append("Video has been paused...")
        # 继续检测
        else:
            self.num_stop += 1
            self.ui.pushButton_Pause.setText(u' 暂停')
            self.ui.textBrowser.append("Video continue...")

    def show_video_frame(self):
        """对视频帧进行分析处理"""
        flag, img = self.cap.read()
        if not flag:
            self.ui.timer_video.stop()
            self.cap.release()
            if self.out is not None:
                self.out.release()
            self.ui.label_Camera_Video_Pic.clear()
            self.ui.textBrowser.append("Failed to read the video frame...")
            # 恢复按钮状态
            self.ui.pushButton_Import_Video.setDisabled(False)
            self.ui.pushButton_Import_Pic.setDisabled(False)
            self.ui.pushButton_Open_Camera.setDisabled(False)
            self.ui.pushButton_Model_Init.setDisabled(False)
            self.ui.pushButton_Model_Select.setDisabled(False)
            # 也可以对后半部分使用 else
            return

        # 成功读取到视频帧
        showimg = img.copy()
        # 用于存储每个检测类别的数量
        class_counts = {}  # Initialize/Reset the class counts

        with torch.no_grad():
            # Image preprocessing and prediction steps
            # 使用 letterbox 函数调整图像大小并进行预处理
            img = letterbox(img, new_shape=self.opt.img_size)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            # 将图像转换为 PyTorch 张量并移动到指定设备（如 GPU）
            img = torch.from_numpy(img).to(self.device)
            img = img.float()  # uint8 to fp32, half precision if using CUDA
            img /= 255.0  # Normalize image to [0,1]
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # 进行推理，获取预测结果 pred
            pred = self.model(img, augment=self.opt.augment)[0]
            # 应用非极大值抑制（NMS）以过滤重复的检测框
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                       agnostic=self.opt.agnostic_nms)
            # Process detections
            # 处理检测结果
            for i, det in enumerate(pred):  # detections per image
                if det is not None and len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], showimg.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        cls_name = self.names[int(cls)]
                        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                        label = f'{cls_name} {conf:.2f}'
                        plot_one_box(xyxy, showimg, label=label, color=self.colors[int(cls)], line_thickness=3)

        # Update the text browser
        total_counts = sum(class_counts.values())
        text_browser_content = f"The total number of targets detected: {total_counts}"
        for cls_name, count in class_counts.items():
            text_browser_content += f"\n{cls_name}: {count}"
        self.ui.textBrowser.append(str(text_browser_content))
        # 将处理后的图像调整为指定大小并转换为 QImage 格式
        show = cv2.resize(showimg, (640, 480))
        self.result = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                 QtGui.QImage.Format_RGB888)
        self.ui.label_Camera_Video_Pic.setPixmap(QtGui.QPixmap.fromImage(showImage))
        # 视频帧显示期间，禁用其他检测按键功能
        self.ui.pushButton_Import_Video.setDisabled(True)  # 禁止导入视频
        self.ui.pushButton_Import_Pic.setDisabled(True)  # 禁止导入图片
        self.ui.pushButton_Model_Init.setDisabled(True)  # 禁止初始化
        self.ui.pushButton_Model_Select.setDisabled(True)  # 禁止选择权重
        # 如果flag为奇数 该按键为视频处理服务
        if not self.camera_flag:
            self.ui.pushButton_Open_Camera.setDisabled(True)  # 禁止开启摄像头

    def import_video(self):
        """该函数实现视频的导入"""
        video_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "打开视频", "", "*.mp4;;*.avi;;All Files(*)")
        flag = self.cap.open(video_name)
        if not flag:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"导入视频失败", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
            self.ui.textBrowser.append("Video import failure...")
        else:
            self.ui.textBrowser.append("Video import successfully...")
            self.ui.textBrowser.append("video path: " + video_name)
            self.out = cv2.VideoWriter('prediction.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20,
                                       (int(self.cap.get(3)), int(self.cap.get(4))))
            # 开启定时器 30ms溢出一次 一旦溢出调用函数 show_video_frame
            self.ui.timer_video.start(30)
            # 自适应界面大小
            self.ui.label_Camera_Video_Pic.setScaledContents(True)
            # 进行视频识别时，关闭其他按键点击功能
            self.ui.pushButton_Import_Video.setDisabled(True)  # 禁止导入视频
            self.ui.pushButton_Import_Pic.setDisabled(True)  # 禁止导入图片
            self.ui.pushButton_Open_Camera.setDisabled(True)  # 禁止开启摄像头
            self.ui.pushButton_Model_Init.setDisabled(True)  # 禁止初始化
            self.ui.pushButton_Model_Select.setDisabled(True)  # 禁止选择权重
            # 在视频识别时，只允许使用按钮 暂停 or 结束
            self.camera_flag = False    # 静止使用摄像头

    def import_pic(self):
        """该函数实现图片的导入"""
        img_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        # 对图片情况进行判断
        if not img_name:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"导入图片失败", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
            self.ui.textBrowser.append("Image import failure...")
        else:
            self.ui.textBrowser.append("Image import successfully...")
            self.ui.textBrowser.append("image path: " + img_name)
            # 对图片进行处理
            name_list = []
            # 图片的路径
            img = cv2.imread(img_name)
            showimg = img
            with torch.no_grad():
                # 根据导入的图片的尺寸来调节图片尺寸
                img = letterbox(img, new_shape=self.opt.img_size)[0]
                # Convert
                # BGR to RGB, to 3x416x416
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                # Inference
                pred = self.model(img, augment=self.opt.augment)[0]
                # Apply NMS
                pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                           agnostic=self.opt.agnostic_nms)
                # 对目标物体的检测结果
                print(pred)
                self.ui.textBrowser.append("target information:")
                self.ui.textBrowser.append(str(pred))

                # Process detections
                for i, det in enumerate(pred):
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(
                            img.shape[2:], det[:, :4], showimg.shape).round()

                        for *xyxy, conf, cls in reversed(det):
                            label = '%s %.2f' % (self.names[int(cls)], conf)
                            name_list.append(self.names[int(cls)])
                            plot_one_box(xyxy, showimg, label=label,
                                         color=self.colors[int(cls)], line_thickness=2)

            cv2.imwrite('prediction.jpg', showimg)
            self.result = cv2.cvtColor(showimg, cv2.COLOR_BGR2BGRA)
            self.result = cv2.resize(self.result, (489, 351), interpolation=cv2.INTER_AREA)
            self.QtImg = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                      QtGui.QImage.Format_RGB32)
            self.ui.label_Camera_Video_Pic.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
            self.ui.label_Camera_Video_Pic.setScaledContents(True)  # 自适应界面大小

    def open_model(self):
        """选择权重文件"""
        self.openfile_name_model, _ = QFileDialog.getOpenFileName(self.ui.pushButton_Model_Select, '选择weights文件',
                                                                  'quanzhong/exp88/weights', "*.pt;")
        if not self.openfile_name_model:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"权重打开失败", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
            self.ui.textBrowser.append("Weight opening failure...")
        else:
            self.ui.textBrowser.append("Weight open successfully...")
            self.ui.textBrowser.append('weight address: ' + str(self.openfile_name_model))
            # self.ui.textBrowser.moveCursor(QTextCursor.End)

    def model_init(self):
        """模型初始化"""
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov5s.pt',
                            help='model path(s)')
        parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
        parser.add_argument('--data', type=str, default='data/coco128.yaml', help='(optional) dataset.yaml path')
        parser.add_argument('--img-size', nargs='+', type=int, default=640, help='inference size h,w')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
        parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
        parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='show results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--visualize', action='store_true', help='visualize features')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
        parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
        parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
        self.opt = parser.parse_args()
        # 打印模型默认初始化参数
        print(self.opt)
        # 放上去装逼用
        self.ui.textBrowser.append("default initialization parameter:")
        self.ui.textBrowser.append(str(self.opt))
        # 默认使用'--weights'中的权重来进行初始化 这里进行更新
        source, weights, view_img, save_txt, imgsz = self.opt.source, self.opt.weights, self.opt.view_img, self.opt.save_txt, self.opt.img_size
        # 若openfile_name_model不为空，则使用openfile_name_model权重进行初始化
        if self.openfile_name_model:
            weights = self.openfile_name_model

        self.device = select_device(self.opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        cudnn.benchmark = True

        # Load model
        self.model = attempt_load(
            weights, map_location=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16

        # Get names and colors
        self.names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255)
                        for _ in range(3)] for _ in self.names]
        # 清空画面
        self.ui.label_Camera_Video_Pic.clear()
        # print("model initial done")
        QtWidgets.QMessageBox.information(self, u"!", u"模型初始化成功", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
        self.ui.textBrowser.append("Model initialization succeeded...")

    def go_change_password(self):
        self.ui.stackedWidget_2.setCurrentIndex(1)

    def go_back(self):
        self.ui.stackedWidget_2.setCurrentIndex(0)

    def change_password(self):
        # 首先 两次输入的密码均不能为空
        if self.ui.lineEdit_Change_P1.text() == "" or self.ui.lineEdit_Change_P2.text() == "":
            self.ui.stackedWidget_3.setCurrentIndex(3)
        # 两次输入的密码必须一致
        elif self.ui.lineEdit_Change_P1.text() != self.ui.lineEdit_Change_P2.text():
            self.ui.stackedWidget_3.setCurrentIndex(1)
        # 正确的情况
        else:
            # 获取当前用户信息
            global user_now
            # 获取密码
            password = self.ui.lineEdit_Change_P1.text()
            # conn = psycopg2.connect(database="DataMyUI", user="postgres", password="123456", host="127.0.0.1", port=5432)
            conn = MYSQL_CONN
            cur = conn.cursor()
            # 获取数据库信息
            cur.execute(f"update users set passwords='{password}' where accounts='{user_now}'")
            # 断开数据库连接
            conn.commit()
            cur.close()
            # 提示修改成功
            self.ui.stackedWidget_3.setCurrentIndex(2)

    def logout(self):
        # 从主页面退出到登录页面
        global user_now
        user_now = ""
        self.close()
        self.login = LoginWindow()

    def go_Main(self):
        self.ui.stackedWidget.setCurrentIndex(0)

    def go_Me(self):
        self.ui.stackedWidget.setCurrentIndex(1)

    def mousePressEvent(self, event):
        # 记录鼠标按下的初始位置
        self.offset = event.pos()

    def mouseMoveEvent(self, event):
        # 移动窗口位置
        if event.buttons() == QtCore.Qt.LeftButton:
            self.move(self.pos() + event.pos() - self.offset)


if __name__ == '__main__':
    # 解决界面模糊，缩放比例问题
    # QApplication.setHighDpiScaleFactorRoundingPolicy(QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    # 适应高DPI设备
    QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    # 解决图片在不同分辨率显示模糊问题
    QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
    # 创建了一个 QApplication 实例
    app = QApplication(sys.argv)
    # 创建了一个 LoginWindow 实例
    window = LoginWindow()
    # 运行应用程序事件循环，直到应用程序结束退出
    sys.exit(app.exec_())


#pyinstaller main.py --noconsole --workpath F:\Code\Python\WADS  --distpath d:\pybuild\dist
