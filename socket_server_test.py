import socket
import sys

from dqlib.Predict import load_and_predict

# Create a TCP/IP socket
# connection type
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

print(sock)
# Bind the socket to the port
server_address = ('127.0.0.1', 1124)
#print >>sys.stderr, 'starting up on %s port %s' % server_address
print('starting up on {} port {}'.format(server_address[0], server_address[1]))
sock.bind(server_address)

# Listen for incoming connections
sock.listen(1)

while True:
    # Wait for a connection
    # print >>sys.stderr, 'waiting for a connection'
    # print('waiting for')
    connection, client_address = sock.accept()
    
    try:
        print('connection from'.format(client_address))

        # Receive the data in small chunks and retransmit it
        while True:
            data = connection.recv(30)
            print('received {}'.format(data))
            
            # predict_result = load_and_predict(path_to_image, path_to_model)
            
            if data:
                print('sending data back to the client')
                connection.sendall(data)
            else:
                print('no more data from'.format(client_address))
                break
            
    finally:
        # Clean up the connection
        connection.close()