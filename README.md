# dog_locomotion
prototyping locomotion for a quadruped robot

## Overview
This repository provides the basic framework for quadruped controls. This is formatted as a ROS package which can be used with conjuction with https://github.com/AMABerkeley/jelly_core to control a physical jelly (the name of the AMAC quadruped) robot. Testing in simulation alone is supported without the use of ROS through the pybullet.

## Dependencies
* pyKDL
  * pip install PyKDL
* quadprog
  * pip install quadprog
* pybullet
  * pip install pybullet
* urdf parser
  * pip install urdf-parser-py
