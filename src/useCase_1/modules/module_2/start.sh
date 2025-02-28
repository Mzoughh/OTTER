#!/bin/bash
# Display the IPv4 address, subnet mask, broadcast address, and MAC address
ip addr show eth0 | awk '
  /inet / {print "IPv4 Address:", $2; print "Broadcast Address:", $4}
  /link\/ether/ {print "MAC Address:", $2}
'

# Run the Python script
python module_2.py
