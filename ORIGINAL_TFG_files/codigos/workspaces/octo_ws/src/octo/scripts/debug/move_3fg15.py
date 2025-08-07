import urx

rob = urx.Robot("169.254.128.101")

# Load boilerplate from file
with open("./boilerplates/3fg15_boilerplate.txt", "r") as f:
    boilerplate = f.read()

WIDTH = input("Internal width (range[19, 135]): ")

close_grip_command = f"tfg_release(diameter={WIDTH}, tool_index=0 ) \n"

robot_program = boilerplate + close_grip_command + "end \n test_3FG15()"

rob.send_program(robot_program)