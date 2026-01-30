SCRIPT_DIR="$1"
# MY_DIR="$(realpath -s "$SCRIPT_DIR")"
# LIMPIAR
unset PYTHONPATH
unset LD_LIBRARY_PATH

# CONFIG ROS
export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# PATHS ESPECÍFICOS DE ISAAC SIM 5.0.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/isaacsim/_build/linux-x86_64/release/exts/isaacsim.ros2.bridge/humble/lib
export PYTHONPATH=~/isaacsim/_build/linux-x86_64/release/exts/isaacsim.ros2.bridge/humble/lib/site-packages:$PYTHONPATH

# VERIFICAR (opcional)
#echo "Python version: $(./python.sh -c 'import sys; print(sys.version)')"
#echo "rclpy location: $(python3 -c 'import rclpy; print(rclpy.__file__)')"

# EJECUTAR
./python.sh $SCRIPT_DIR
