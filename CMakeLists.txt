cmake_minimum_required(VERSION 2.8.3)

project(jelly_locomotion)

find_package(catkin REQUIRED COMPONENTS
    rospy
)

catkin_package(CATKIN_DEPENDS xacro)
catkin_python_setup()

find_package(roslaunch)

#foreach(dir launch)
#    install(DIRECTORY ${dir}/
#        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
#endforeach(dir)

include_directories(
    ${catin_INCLUDE_DIRS}
)
