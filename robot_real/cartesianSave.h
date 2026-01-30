#pragma once
#include <array>
#include <cmath>

// Declaración de la función principal para mover el robot
bool control_cart(std::array<double, 3> goal_pos,
                std::array<double, 3> goal_orient,
                float v_max = 0.3,
                float w_max = M_PI_4,
                bool deg = false,
                bool save_data = false);

bool go_home();
