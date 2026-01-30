# TFG_Adrian_Fernandez_Casas
Recopilación de los códigos utilizados para el TFG Control en impedancia para la actuación rápida de robots con visión basada en eventos

- saveVideo: programa que llama a la funcion cartesianSave para mover al robot a una serie de trayectorias cartesianas, guardadas en txt, para que haga la trayectoria mientras guarda tanto el robot_state (a 60Hz) como el video.

- saveVideo_joint: programa que guarda mueve al robot a una serie de trayectorias articulares y guarda el video (no se puede grabar robot_state).

- go_home: cierra la pinza y se dirige a home mediante movimiento articular.

- calibracion_joint: realiza la autocalibracion mediante movimiento articular, genera 2 archivos de salida, el vector de traslacion del efector a la camara, y el de la base al centro del chessboard (no centro geometrico). Utiliza las funciones acquire-data, eye-in-hand y chessboard.

- controlFranka: movimiento cartesiano, mueve el ee a una pose seleccionada, con una determinada velocidad lineal y angular.

- joint_impedance_control: ejemplo de control en impedancia, mueve el ee en una trayectoria circular en el plano YZ

- control_impedancia: utiliza la base de controlFranka para controlarlo en impedancia, calculando los torques necesarios. No funciona bien, ya que hay que configurar los valores de damping y stiffness mediante reinforcement learning.

- moveTrajectory: programa muy simple que llama a la funcion cartesianSave para realizar una trayectoria cartesiana

- moveJointTrajectory: programa que lee un txt con configuraciones articulares y realiza la trayectoria articular
