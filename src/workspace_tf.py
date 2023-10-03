# Covert raw RealSense `/camera/depth/image_rect_raw` data to Open3D point cloud data
# Run this first: `roslaunch realsense2_camera rs_camera.launch`

import sys
import rospy
import tf
import tf2_ros
# import geometry_msgs.msg
from tf.transformations import quaternion_from_matrix, quaternion_matrix
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion, TransformStamped

class workspace_tf():
    def __init__(self):
        self.static_tfs = []
        self.broadcaster = tf2_ros.StaticTransformBroadcaster()
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

    def add_static_tf(self, ref, obj, ht):
        static_transformStamped = TransformStamped()

        static_transformStamped.header.stamp = rospy.Time.now()
        static_transformStamped.header.frame_id = ref
        static_transformStamped.child_frame_id = obj

        quat = quaternion_from_matrix(ht)

        static_transformStamped.transform.translation.x = float(ht[0,3])
        static_transformStamped.transform.translation.y = float(ht[1,3])
        static_transformStamped.transform.translation.z = float(ht[2,3])

        static_transformStamped.transform.rotation.x = quat[0]
        static_transformStamped.transform.rotation.y = quat[1]
        static_transformStamped.transform.rotation.z = quat[2]
        static_transformStamped.transform.rotation.w = quat[3]

        self.static_tfs.append(static_transformStamped)

        self.update()

    def update(self):
        self.broadcaster.sendTransform(self.static_tfs)

    def get_tf(self, ref_frame, obj):
        updated = False
        while updated==False:
          try:
              msg = self.tfBuffer.lookup_transform(ref_frame, obj, rospy.Time())
              trans = [msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z]
              quat = [msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w]
              h = quaternion_matrix(quat)
              h[:3,3] = trans
              # return geo
              return h
          except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
              rospy.sleep(0.1)


if __name__ == '__main__':
  rospy.init_node('tf_converter', anonymous=True)
  ws_tf = workspace_tf()
  # rate = rospy.Rate(10)
  while not rospy.is_shutdown():
    ws_tf.get_tf()
    if ws_tf.tf_updated:
      print(ws_tf.trans)
      print(ws_tf.rot)
      print("====")
      # rate.sleep()
    else:
      print("marker not found")
      print("====")