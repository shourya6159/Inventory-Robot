import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    home_dir = os.path.expanduser('~')
    
    #NOTE : CHANGE 'task4_ws' to your workspace name!!!
    world_path = os.path.join(home_dir, 'task4_ws/src/inventory_bot/worlds/warehouse.sdf')
    urdf_path = os.path.join(home_dir, 'task4_ws/src/inventory_bot/urdf/differential_robot.urdf')
    map_path = os.path.join(home_dir, 'task4_ws/src/inventory_bot/maps/warehouse.yaml')
    rviz_config_path = '/opt/ros/humble/share/nav2_bringup/rviz/nav2_default_view.rviz'

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': f'-r {world_path}'}.items()
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        arguments=[urdf_path],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )


    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-topic', 'robot_description', '-name', 'my_robot', '-x', '0.0', '-y', '0.0', '-z', '0.5'],
        output='screen'
    )

    ros_gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image'
        ],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    nav2_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('nav2_bringup'), 'launch', 'bringup_launch.py')
        ),
        launch_arguments={
            'map': map_path,
            'use_sim_time': 'True'
        }.items()
    )

    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_path],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )
   
    return LaunchDescription([
        #Start Gazebo immediately
        gz_sim,
        
        #Start Robot State Publisher 0.5 seconds later
        TimerAction(
            period=0.5,
            actions=[robot_state_publisher]
        ),
        
        #Spawn the robot 1.0 seconds later (0.5s after RSP)
        TimerAction(
            period=1.0,
            actions=[spawn_robot]
        ),
        
        #Start the Bridge 1.5 seconds later (0.5s after spawning)
        TimerAction(
            period=1.5,
            actions=[ros_gz_bridge]
        ),
        
        #Start Nav2 2.0 seconds later (0.5s after Bridge)
        TimerAction(
            period=2.0,
            actions=[nav2_bringup]
        ),
        
        #Start RViz 3.5 seconds later (1.5s after Nav2 bringup starts)
        TimerAction(
            period=3.5,
            actions=[rviz2]
        )
     
        
    ])
