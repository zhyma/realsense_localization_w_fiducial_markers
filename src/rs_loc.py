# Covert raw RealSense Depth data to RViz PointCloud2 data
# Use Pyrealsense2 to obtain RS data, no launch file is needed

import sys, copy, time, cv2

import pyrealsense2 as rs
from cv_bridge import CvBridge, CvBridgeError

from get_marker   import *
from workspace_tf import workspace_tf

import numpy as np

import rospy

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField

## convert RealSense depth data to ROS PointCloud2
import struct
from sensor_msgs import point_cloud2
from std_msgs.msg import Header


class rs2pc2():
    ## 

    def __init__(self, serial, tag_id=1, alias="", ht_tag2world=None, width = 640, height = 480):
        '''
        serial: The serial NO. of the RealSense
        tag_id: The ID of the ArUco marker ("id" is a Python built-in function)
        alias: name your camera topic
        ht_tag2world: homogeneous transformation matrix, use world as the reference frame
        The default width and height are set to  640x480
        '''
        self.id = tag_id
        if ht_tag2world is None:
            ## ht_ar2world is a homogeneous transformation matrix
            ## that represent offset of the ArUco tag to the world
            ## example
            ## t_ar2world = np.array([[ 0.,  0.,  1.,  0.005],\
            ##                        [ 1.,  0.,  0.,  0.0  ],\
            ##                        [ 0.,  1.,  0.,  0.070],\
            ##                        [ 0.,  0.,  0.,  1.   ]])
            self.ht_tag2world = np.identity(4)
        else:
            self.ht_tag2world = copy.deepcopy(ht_tag2world)

        if alias=="":
            self.alias = "/rs_" + str(serial)
        else:
            self.alias = alias

        self.height = height
        self.width = width

        self.image_pub = rospy.Publisher(self.alias+"/color/raw", Image, queue_size = 10)
        self.debug_pub = rospy.Publisher(self.alias+"/debug", Image, queue_size = 10)
        self.bridge = CvBridge()

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(str(serial))
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(self.config)
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        print("Depth Scale is: " , self.depth_scale)

        self.k = [0.0]*9 # camera's intrinsic parameters
        self.distort = [0.0]*5
        self.get_cam_param()
        print(self.k)
        self.is_data_updated = False

        ## homogeneous transformation matrix from the camera (obj) to the world (ref)
        self.ht = np.zeros([4,4])

        self.pc2_pub = rospy.Publisher(self.alias+"/pc2", PointCloud2, queue_size=10)

        ## wait for 1s to maker sure color images arrive
        rospy.sleep(1)


    def get_cam_param(self):
        st_profile = self.profile.get_stream(rs.stream.depth)
        self.intr = st_profile.as_video_stream_profile().get_intrinsics()
        self.k[0] = self.intr.fx
        self.k[2] = self.intr.ppx
        self.k[4] = self.intr.fy
        self.k[5] = self.intr.ppy
        self.k[8] = 1.0

        for i in range(5):
            self.distort[i] = self.intr.coeffs[i]

    def get_rgbd(self):
        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        align = rs.align(align_to)

        try:
            # Get frameset of color and depth
            frames = self.pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                return -1

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            #depth image is 1 channel, color is 3 channels
            depth_image_3d = np.dstack((depth_image,depth_image,depth_image))
            # Render images:
            #   depth align to color on left
            #   depth on right
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # self.image = np.hstack((bg_removed, depth_colormap))
            self.color_img = color_image
            self.depth_1d = depth_image
            self.depth_img = depth_colormap
            return 1

        except:
            return -1

    def get_marker_3d_pos(self):
        _ = self.get_rgbd()
        ## get the homogeneous transformation matrix from the marker to the camera
        ## get the point location on the 2D image
        ## render a image that shows the coordinate frame on the 2D image
        img, ht, pt_2d = find_aruco(self.color_img, self.k, self.distort, id=self.id)
        if ht is not None:
            z = self.depth_1d[pt_2d[1], pt_2d[0]]
            pt_3d = rs.rs2_deproject_pixel_to_point(self.intr, pt_2d, z)
            ## mm to meters
            ht[:3,3] = [i*self.depth_scale for i in pt_3d]
            rot = np.array([[ 0, 0,  1, 0],
                            [-1, 0,  0, 0],
                            [ 0,-1,  0, 0],
                            [ 0, 0,  0, 1]]).astype("float64")
            ht = np.dot(rot, ht)

        return ht, img

    def get_cam_pose(self):
        ht, _ = self.get_marker_3d_pos()
        self.ht_cam2world = np.dot(self.ht_tag2world, np.linalg.inv(ht))

    def depth_1d_to_3d(self):
        self.depth_3d = np.zeros([self.height, self.width, 3])
        for iy in range(self.height):
            for ix in range(self.width):
                z = self.depth_1d[iy, ix]
                pt_3d = rs.rs2_deproject_pixel_to_point(self.intr, [ix, iy], z)
                self.depth_3d[iy,ix] = [i*self.depth_scale for i in pt_3d]

    def depth_to_pc2(self):
        self.depth_1d_to_3d()
        #self.get_cam_pose()

        points = []
        cam_ht = np.array([[0., 0., 1., 0.],
                           [-1., 0., 0., 0],
                           [0., -1., 0., 0.],
                           [0., 0., 0., 1.]])

        t = np.dot(self.ht_cam2world, cam_ht)
        for iy in range(self.height):
            for ix in range(self.width):
                pos = np.ones((4,1))
                pos[:3,0] = self.depth_3d[iy, ix]
                pos = np.dot(t, pos)
                x = pos[0]
                y = pos[1]
                z = pos[2]
                ## opencv use bgr
                r = self.color_img[iy, ix, 2]
                g = self.color_img[iy, ix, 1]
                b = self.color_img[iy, ix, 0]
                a = 255
                # print r, g, b, a
                rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
                # print hex(rgb)
                pt = [x, y, z, rgb]
                points.append(pt)

        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  # PointField('rgb', 12, PointField.UINT32, 1),
                  PointField('rgba', 12, PointField.UINT32, 1),
                  ]

        header = Header()
        header.frame_id = "world"
        pc2 = point_cloud2.create_cloud(header, fields, points)
        return pc2


    def pub_img(self):
        img_msg = self.bridge.cv2_to_imgmsg(self.color_img)
        self.image_pub.publish(img_msg)

    def pub_debug(self, img=None):
        if img is None:
            image = np.hstack((self.color_img, self.depth_img))
            img_msg = self.bridge.cv2_to_imgmsg(image)
        else:
            img_msg = self.bridge.cv2_to_imgmsg(img)
        self.debug_pub.publish(img_msg)

    def pub_pc2(self):
        pc2 = self.depth_to_pc2()
        pc2.header.stamp = rospy.Time.now()
        self.pc2_pub.publish(pc2)



if __name__ == '__main__':
    print(cv2.__version__)
    rospy.init_node("d405", anonymous = True)
    np.set_printoptions(suppress=True)
    

    tag_offset = np.array([[ 0.,  0.,  1.,  0.005],\
                           [ 1.,  0.,  0.,  0.0  ],\
                           [ 0.,  1.,  0.,  0.070],\
                           [ 0.,  0.,  0.,  1.   ]])

    # front_cam = rs2pc2("218622278069")
    # front_cam = rs2pc2("849412061719")
    front_cam = rs2pc2("851112063978", tag_id=1, alias="front_cam", ht_tag2world=tag_offset)


    rospy.sleep(1)
    ws_tf = workspace_tf()

    front_cam.get_cam_pose()
    ws_tf.add_static_tf("world", "marker_1", tag_offset)
    ws_tf.add_static_tf("world", "front_cam_link", front_cam.ht_cam2world)


    while not rospy.is_shutdown():

        front_cam.pub_pc2()
        rospy.sleep(1.0)