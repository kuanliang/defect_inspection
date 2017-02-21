import socket
import sys

host = ''
port = 1224
    
mySocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
try:
    mySocket.bind((host, port))
except socket.error as e:
    print(str(e))
    
# it is a quee, how many connections
mySocket.listen(5)
    
conn, addr = mySocket.accept()
print("Connection from: " + str(addr))
