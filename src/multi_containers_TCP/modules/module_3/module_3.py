#####################################
# Third Container
# module_3.py
# TCP Socket SERVER Deployment 
#####################################
#####################################
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
    # Nerwork parameters
    HOST = '0.0.0.0'
    PORT = 5000  

    # Buffer size for send/receive variable
    buffer_size =  1024

    ########### Interaction with C2 ###########
    # Receive client request which ask for addr
    conn = tcp_tools.init_server_TCP_connection(HOST, PORT)
    data_decode = tcp_tools.wait_for_container_variable_TCP(conn, buffer_size)

    ##########################################
    # Placeholder for future AI prediction 
    logging.info(f"End of deployment")
    ##########################################

