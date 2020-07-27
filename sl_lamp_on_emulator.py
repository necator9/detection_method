import time

from sl_sensor_connect import SlSensor


sl_app = SlSensor(send_port=65434, recv_port=65433)


def sender_behaviour():
    sl_app.send('Lamp_ON')
    time.sleep(0.5)
    sl_app.send('Lamp_OFF')
    time.sleep(0.5)


def receiver_behaviour():
    stat = sl_app.recieve()
    if stat:
        print('SL_app  received: {}'.format(stat))
    time.sleep(0.01)


while True:
    receiver_behaviour()
