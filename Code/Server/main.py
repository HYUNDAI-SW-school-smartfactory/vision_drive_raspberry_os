import argparse
import signal
import struct
import sys
import threading
import time

import cv2
import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow

from camera import Camera
from car import Car
from command import Command
from crosswalk_detector import CrosswalkDetector
from drive import AutoRaceLaneDrive, build_parser
from message import Message_Parse
from server import Server
from server_ui import Ui_server_ui


class mywindow(QMainWindow, Ui_server_ui):
    def __init__(self):
        self.app = QApplication(sys.argv)
        super(mywindow, self).__init__()
        self.setupUi(self)
        self.ui_button_state = True
        self.config_task()
        self.Button_Server.clicked.connect(self.on_pushButton_handle)
        if self.ui_button_state:
            self.on_pushButton_handle()
        self.app.lastWindowClosed.connect(self.close_application)
        signal.signal(signal.SIGINT, self.signal_handler)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_signals)
        self.timer.start(100)

    def config_task(self):
        self.tcp_server = Server()
        self.command = Command()
        self.car = Car()
        self.camera = Camera(stream_size=(640, 480))
        self.crosswalk_detector = CrosswalkDetector(stop_duration=5.0)

        lane_args = build_parser().parse_args([])
        lane_args.headless = True
        lane_args.lane_side = "right"
        self.lane_driver = AutoRaceLaneDrive(lane_args, motor=self.car.motor, init_camera=False)

        self.queue_cmd = []
        self.cmd_parse = Message_Parse()

        self.cmd_thread = None
        self.video_thread = None
        self.car_thread = None
        self.cmd_thread_is_running = False
        self.video_thread_is_running = False
        self.car_thread_is_running = False

    def stop_car(self):
        self.lane_driver.stop()
        self.camera.stop_stream()
        self.camera.close()
        self.car.close()

    def on_pushButton_handle(self):
        if self.label.text() == "Server Off":
            self.label.setText("Server On")
            self.Button_Server.setText("Off")
            self.tcp_server.start_tcp_servers()
            self.set_threading_cmd_receive(True)
            self.set_threading_video_send(True)
            self.set_threading_car_task(True)
        elif self.label.text() == "Server On":
            self.label.setText("Server Off")
            self.Button_Server.setText("On")
            self.tcp_server.stop_tcp_servers()
            self.set_threading_cmd_receive(False)
            self.set_threading_video_send(False)
            self.set_threading_car_task(False)
            self.tcp_server = Server()

    def send_power_data(self):
        if not self.tcp_server.get_command_server_busy():
            power = self.car.adc.read_adc(2) * (3 if self.car.adc.pcb_version == 1 else 2)
            cmd = self.command.CMD_POWER + "#" + str(power) + "\n"
            self.tcp_server.send_data_to_command_client(cmd)

    def set_threading_cmd_receive(self, state, close_time=0.3):
        if self.cmd_thread is None:
            buf_state = False
        else:
            buf_state = self.cmd_thread.is_alive()
        if state != buf_state:
            if state:
                self.cmd_thread_is_running = True
                self.cmd_thread = threading.Thread(target=self.threading_cmd_receive)
                self.cmd_thread.start()
            else:
                self.cmd_thread_is_running = False
                if self.cmd_thread is not None:
                    self.cmd_thread.join(close_time)
                    self.cmd_thread = None

    def threading_cmd_receive(self):
        ignored_commands = {
            self.command.CMD_MOTOR,
            self.command.CMD_M_MOTOR,
            self.command.CMD_CAR_ROTATE,
            self.command.CMD_MODE,
            self.command.CMD_SERVO,
            self.command.CMD_LED,
            self.command.CMD_LED_MOD,
            self.command.CMD_BUZZER,
            self.command.CMD_SONIC,
            self.command.CMD_LIGHT,
            self.command.CMD_LINE,
        }

        while self.cmd_thread_is_running:
            cmd_queue = self.tcp_server.read_data_from_command_server()
            if cmd_queue.qsize() > 0:
                _client_address, all_message = cmd_queue.get()
                main_message = all_message.strip()
                if "\n" in main_message:
                    self.queue_cmd.extend(msg for msg in main_message.split("\n") if msg)
                elif main_message:
                    self.queue_cmd.append(main_message)

            while self.queue_cmd:
                msg = self.queue_cmd.pop(0)
                self.cmd_parse.clear_parameters()
                self.cmd_parse.parse(msg)
                print(self.cmd_parse.input_string)

                if self.cmd_parse.command_string == self.command.CMD_POWER:
                    self.send_power_data()
                elif self.cmd_parse.command_string in ignored_commands:
                    print(f"Ignored command in locked lane-drive mode: {self.cmd_parse.command_string}")

            if not self.queue_cmd:
                time.sleep(0.001)

    def set_threading_car_task(self, state, close_time=0.3):
        if self.car_thread is None:
            buf_state = False
        else:
            buf_state = self.car_thread.is_alive()
        if state != buf_state:
            if state:
                self.car_thread_is_running = True
                self.car_thread = threading.Thread(target=self.threading_car_task)
                self.car_thread.start()
            else:
                self.car_thread_is_running = False
                if self.car_thread is not None:
                    self.car_thread.join(close_time)
                    self.car_thread = None

    def threading_car_task(self):
        while self.car_thread_is_running:
            self.mode_lane_drive()
            time.sleep(0.01)

    def mode_lane_drive(self):
        if not self.camera.streaming:
            try:
                self.camera.start_stream()
            except Exception as e:
                print(f"Lane drive stream error: {e}")
                time.sleep(0.1)
                return

        frame = self.camera.get_frame()
        if frame is None:
            return

        try:
            if self.crosswalk_detector.process_jpeg_frame(frame):
                print(
                    "Crosswalk detected: green_pixels={}, stopping for {:.1f}s".format(
                        self.crosswalk_detector.get_green_pixels(),
                        self.crosswalk_detector.stop_duration,
                    )
                )

            if self.crosswalk_detector.should_stop():
                self.lane_driver.pid.reset()
                self.car.motor.set_motor_model(0, 0, 0, 0)
                return

            frame_array = np.frombuffer(frame, dtype=np.uint8)
            frame_bgr = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                return

            result = self.lane_driver.process_frame(frame_bgr)
            if result["stop_required"]:
                self.car.motor.set_motor_model(0, 0, 0, 0)
            elif result["lane_detected"]:
                self.lane_driver.apply_drive_command(result["cmd_left"], result["cmd_right"])
            print(f"[LANE] {result['msg']}")
        except cv2.error as e:
            print(f"Lane drive cv2 error: {e}")
        except Exception as e:
            print(f"Lane drive error: {e}")

    def set_threading_video_send(self, state, close_time=0.3):
        if self.video_thread is None:
            buf_state = False
        else:
            buf_state = self.video_thread.is_alive()
        if state != buf_state:
            if state:
                self.video_thread_is_running = True
                self.video_thread = threading.Thread(target=self.threading_video_send)
                self.video_thread.start()
            else:
                self.video_thread_is_running = False
                if self.video_thread is not None:
                    self.video_thread.join(close_time)
                    self.video_thread = None

    def threading_video_send(self):
        while self.video_thread_is_running:
            if self.tcp_server.is_video_server_connected():
                self.camera.start_stream()
                while self.tcp_server.is_video_server_connected():
                    frame = self.camera.get_frame()
                    len_frame = len(frame)
                    length_bin = struct.pack("<I", len_frame)
                    try:
                        self.tcp_server.send_data_to_video_client(length_bin)
                        self.tcp_server.send_data_to_video_client(frame)
                    except Exception:
                        break
            else:
                time.sleep(0.1)

    def close_application(self):
        self.ui_button_state = False
        self.set_threading_cmd_receive(False)
        self.set_threading_video_send(False)
        self.set_threading_car_task(False)
        if self.tcp_server:
            self.tcp_server.stop_tcp_servers()
            self.tcp_server = None
        self.stop_car()
        if self.cmd_thread and self.cmd_thread.is_alive():
            self.cmd_thread.join(0.1)
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(0.1)
        if self.car_thread and self.car_thread.is_alive():
            self.car_thread.join(0.1)
        self.app.quit()
        sys.exit(1)

    def signal_handler(self, _signal, _frame):
        print("Caught Ctrl+C, stopping application...")
        self.close_application()

    def check_signals(self):
        if self.app.hasPendingEvents():
            self.app.processEvents()
        if not self.ui_button_state and not self.cmd_thread_is_running and not self.video_thread_is_running:
            self.app.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Freenove 4WD Smart Car Server")
    parser.add_argument("-t", "--terminal", action="store_true", help="Run in terminal mode (no GUI)")
    parser.add_argument("-n", "--no-gui", action="store_true", help="Run in terminal mode (no GUI)")
    args = parser.parse_args()

    headless_mode = args.terminal or args.no_gui
    if headless_mode:
        app = QApplication(sys.argv)
        server_window = mywindow()
        if server_window.label.text() == "Server Off":
            server_window.on_pushButton_handle()
        signal.signal(signal.SIGINT, server_window.signal_handler)
        try:
            sys.exit(app.exec_())
        except KeyboardInterrupt:
            server_window.close_application()
            sys.exit(0)
    else:
        myshow = mywindow()
        myshow.show()
        sys.exit(myshow.app.exec_())
