import os
import time
from spot_controller import SpotController
import cv2

ROBOT_IP = "192.168.50.3"#os.environ['ROBOT_IP']
SPOT_USERNAME = "admin"#os.environ['SPOT_USERNAME']
SPOT_PASSWORD = "2zqa8dgw7lor"#os.environ['SPOT_PASSWORD']


def main():
    # Use wrapper in context manager to lease control, turn on E-Stop, power on robot and stand up at start
    # and to return lease + sit down at the end
    with SpotController(username=SPOT_USERNAME, password=SPOT_PASSWORD, robot_ip=ROBOT_IP) as spot:
        time.sleep(2)
    
        camera_capture = cv2.VideoCapture(0)

        _, center_image = camera_capture.read()
        camera_capture.release()

        # Move head to specified positions with intermediate time.sleep
        spot.move_head_in_points(yaws=[0.2, 0],
                                 pitches=[0, 0],
                                 rolls=[0, 0],
                                 sleep_after_point_reached=1)
        time.sleep(3)

        _, left_image = camera_capture.read()
        camera_capture.release()

        spot.move_head_in_points(yaws=[0.8, 0],
                                 pitches=[0, 0],
                                 rolls=[0, 0],
                                 sleep_after_point_reached=1)
        
        time.sleep(3)

        _, right_image = camera_capture.read()
        camera_capture.release()

        spot.move_head_in_points(yaws=[0, 0],
                                 pitches=[0.2, 0],
                                 rolls=[0, 0],
                                 sleep_after_point_reached=1)
        
        time.sleep(3)

        _, top_image = camera_capture.read()
        camera_capture.release()

        spot.move_head_in_points(yaws=[0, 0],
                                 pitches=[0.8, 0],
                                 rolls=[0, 0],
                                 sleep_after_point_reached=1)

        time.sleep(3)

        _, bottom_image = camera_capture.read()
        camera_capture.release()

        
        # Make Spot to move by goal_x meters forward and goal_y meters left
        spot.move_to_goal(goal_x=0.5, goal_y=0)
        time.sleep(3)

        # Control Spot by velocity in m/s (or in rad/s for rotation)
        spot.move_by_velocity_control(v_x=-0.3, v_y=0, v_rot=0, cmd_duration=2)
        time.sleep(3)


if __name__ == '__main__':
    main()
