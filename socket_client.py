import socket
import sys


host = '127.0.0.1'
port = 1124
    
mySocket = socket.socket()
mySocket.connect((host, port))
    
message = input(" -> ")
    
while message != 'q':
    mySocket.send(message.encode())
    data = mySocket.recv(1024).decode()
        
    print('Received from server: ' + data)
        
    message = input(" -> ")
        
mySocket.close()
        
