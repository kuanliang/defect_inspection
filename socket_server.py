import socket
from pathlib import Path
import sys

from dqlib.Predict import load_and_predict


host = '127.0.0.1'
port = 1124
    
mySocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
try:
    mySocket.bind((host, port))
except socket.error as e:
    print(str(e))
    
# it is a quee, how many connections
mySocket.listen(1)

conn, addr = mySocket.accept()
print("Connection from: " + str(addr))
    
while True:
    data = conn.recv(1024).decode()
    if not data:
        break
    print("from connected user: " + str(data))
        
    p = Path(data)
    path_to_image = data
        
    print(path_to_image)
    # print(path_to_image[0])
        
    defect_name = p.parts[1]
    path_to_model = './Models/' + defect_name + '/model.pkl'
        
    print(path_to_model)
    # print(path_to_model[0])
                
    predict_result = load_and_predict(path_to_image, path_to_model)
        
    print("sending: " + str(data))
        
    conn.send(data.encode())
        
    # conn.close()
    
