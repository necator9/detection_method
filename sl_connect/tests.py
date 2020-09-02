#!/usr/bin/env python3

# Created by Ivan Matveev at 04.08.20
# E-mail: ivan.matveev@hs-anhalt.de

# Scenarios for testing the connection interface between Smartlighting and detection algorithms

import logging
import sl_connect
import time
from threading import Thread

# Set up logging for standalone tests
logger = logging.getLogger('detect')
logger.setLevel(logging.DEBUG)
logger = logging.getLogger('sl_connect')
ch = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s %(asctime)s %(threadName)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def test_transport():
    sensors_amount = 3

    sl_app = sl_connect.SensorServer(('', 35433), max_clients=sensors_amount)
    sensors = [sl_connect.SensorClient(('', 35433)) for i in range(sensors_amount)]

    print('Server registers clients')
    for i in range(sensors_amount):
        sl_app.receive()
    print(sl_app.clients)

    print('\nClients notify server')
    for i, sensor in enumerate(sensors):
        sensor.send('Yo, sensor {}'.format(i))
        print(sl_app.receive())

    print('\nServer notifies all clients')
    sl_app.send('Yo, server')
    for i, sensor in enumerate(sensors):
        print(sensor.receive())

    sl_app.quit()


def test_sl_cycle(duration):
    i = 0
    det = sl_connect.DetectAlgorithm(server_addr=('', 35433), max_clients=2)
    while i < duration:
        det.check_detect_status()
        if i % 30 == 0:
            det.lamp_on_inform()
        i += 1
        time.sleep(0.01)


def test_det_cycle(duration):
    i = 0
    sl = sl_connect.SlApp(server_addr=('', 35433))
    while i < duration:
        logger.info('Lamp status: {}'.format(sl.check_lamp_status()))
        if i % 10 == 0:
            sl.switch_on_lamp()
        i += 1
        time.sleep(0.01)


if __name__ == '__main__':
    test_transport()

    print('Testing in parallel')
    sl_app = Thread(target=test_sl_cycle, args=(300, ), name='sl_app')
    det = Thread(target=test_det_cycle, args=(400, ), name='detection')
    det2 = Thread(target=test_det_cycle, args=(400,), name='detection2')

    sl_app.start()
    det.start()
    det2.start()

    sl_app.join()
    det.join()
    det2.join()
