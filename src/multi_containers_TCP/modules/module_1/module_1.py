#####################################
# First Container 
# module_1.py
# TCP Socket SERVER Deployment 
#####################################
# Libraries
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

    # Log Initialization
    log_file_path = "/app/logs/container.log"
    tcp_tools.build_log_file(log_file_path)

    ##################################### NETWORK PART #####################################
    # Network Parameters
    HOST = '0.0.0.0'  # Listen on all network interfaces (Server configuration)
    PORT = 5000       # Port for listening
    conn = tcp_tools.init_server_TCP_connection(HOST, PORT)

    ##########################################
    # Placeholder for future AI prediction 
    prediction = 'prediction'
    ##########################################

    ##################################### NETWORK PART #####################################
    tcp_tools.send_variable_to_container_TCP(conn, prediction)

