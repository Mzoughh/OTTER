#####################################
# Second Container 
# module_2.py
#  TCP Socket Client x2 
#####################################
# Libraries
import time
# Libraries
import logging
from otter_net_utils import OtterUtils
tcp_tools = OtterUtils()
print("OTTER Utils class imported successfully!")
#####################################

#####################################
# Placeholder for future AI class
#####################################

if __name__ == "__main__":

    time.sleep(0.5)  # Wait during initialization of server containers
    # Log Initialization
    log_file_path = "/app/logs/container.log"
    tcp_tools.build_log_file(log_file_path) 

    ##################################### NETWORK PART #####################################
    # Network Parameters
    # C1 (Module 1 Container)
    HOST = 'module_1_container'  
    PORT = 5000             
    # C3 (Module 3 Container)
    HOST_3 = 'module_3_container' 
    PORT_3 = 5000

    ########### Interaction with C1 ###########
    s = tcp_tools.init_client_TCP_connection(HOST, PORT)
    # Receive the prediction from module_1
    buffer_size=1024
    data_decode = tcp_tools.wait_for_container_variable_TCP(s, buffer_size)

    ##########################################
    # Placeholder for future AI prediction 
    prediction = 'prediction 2'
    ##########################################

    ########### Interaction with C3 ###########
    s2 = tcp_tools.init_client_TCP_connection(HOST_3, PORT_3)
    tcp_tools.send_variable_to_container_TCP(s2, prediction) # Send Prediction 2 to C3


