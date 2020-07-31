#!/usr/bin/env python3

import socket
import select
import time
from collections import deque


class SensorServer(object):
    """
    Communication between SmartLighting and detection algorithm is defined by this class.
    Each of the applications uses 2 sockets: one for sending and one for receiving.
    Thus, the sending port in one application is the receiving port in another, and vice versa.
    """

    def __init__(self, server_addr=('', 65433), max_clients=2):
        self.sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_recv.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)  # Set size of buf to minimal value to
        # limit the length of a queue that may be required when applications' iteration times are different

        self.sock_recv.bind(server_addr)
        self.clients = deque(maxlen=max_clients)

    def receive(self):
        """
        Non blocking receive to avoid try-except by timeout.
        """
        ready = select.select([self.sock_recv], [], [], 0)
        if ready[0]:
            data, addr = self.sock_recv.recvfrom(16)
            data = data.decode()
            if addr not in self.clients:
                self.clients.append(addr)
                print('Registered new client: {}'.format(addr))
            else:
                return data

    def send(self, data):
        for client_addr in self.clients:
            self.sock_recv.sendto(str(data).encode(), client_addr)

    def quit(self):
        self.sock_recv.close()


class SensorClient(object):
    """
    Communication between SmartLighting and detection algorithm is defined by this class.
    Each of the applications uses 2 sockets: one for sending and one for receiving.
    Thus, the sending port in one application is the receiving port in another, and vice versa.
    """

    class Decorators(object):
        @classmethod
        def atomic_rating_change(cls, decorated):
            #print(cls)
            print('asd')
            return decorated

    def tester(self):
        self.counter +=1
        self._lol = 123


    #@Decorators.atomic_rating_change
   # @tester
    def test(self):
        print(self._lol)
        print('olol')

    def __init__(self, server_addr=('', 65433)):
        self.sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_addr = server_addr
        self.counter = int()
        self.notify_server()

    def notify_server(self):
        self.send('INIT')



    def receive(self):
        """
        Non blocking receive to avoid try-except by timeout.
        """
        ready = select.select([self.sock_send], [], [], 0)
        if ready[0]:
            data, addr = self.sock_send.recvfrom(16)
            data = data.decode()
            return data

    def send(self, data):
        self.sock_send.sendto(str(data).encode(), self.server_addr)

    def quit(self):
        self.sock_send.close()


if __name__ == '__main__':
    sensors_amount = 4

    sl_app = SensorServer(('', 65433), max_clients=sensors_amount)
    sensor = SensorClient(('', 65433))
    sensor.tester()

    sensor.test()
    # sensors = [SensorClient(('', 65433)) for i in range(sensors_amount)]
    #
    # try:
    #     print('Server registers clients')
    #     for i in range(sensors_amount):
    #         sl_app.receive()
    #
    #     print('\nClients notify server')
    #     for i, sensor in enumerate(sensors):
    #         sensor.send('Yo, sensor {}'.format(i))
    #         print(sl_app.receive())
    #
    #     print('\nServer notifies all clients')
    #     sl_app.send('Yo, server')
    #     for i, sensor in enumerate(sensors):
    #         print(sensor.receive())
    #
    # except KeyboardInterrupt:
    #     pass
    #
    # finally:
    #     sl_app.quit()
