#!/usr/bin/env python

# _______________________________________________________________________________________________
# Author: Ivan Matveev
# E-mail: ivan.matveev@student.emw.hs-anhalt.de
# Project: "Development of the detection module for a SmartLighting system"
# Name: "Experimental setup"
# Source code available on github: https://github.com/Necator94/sensors.git.
# _______________________________________________________________________________________________

# Program is targeted for empirical researches of sensors characteristics.
# Program includes collection of sensors response data during defined time.
# Program is supposed to be used in couple with a mobile device client.
# Incoming string via TCP should contain information about time duration and experiment number.

import threading
import Queue
import sys
import socket
import time
import Adafruit_BBIO.GPIO as GPIO                               # The library for GPIO handling
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("experimental_setup")


def polling(gpio_pins, out_raw_data, exp_parameter, name):      # Function for RW sensor polling
    starttime = time.time()                                     # Get start experiment time
    logger.info(name + ' started')
    t_time = 0
    raw_data = []                                               # Array to append by registered data
    for i in range(2): raw_data.append([])

    while t_time < exp_parameter['duration']:                   # Perform during defined time
        check = GPIO.input(gpio_pins['signal_pin'])             # Check connected to sensor GPIO
        t_time = time.time() - starttime                        # Define time from experiment start
        raw_data[0].append(t_time)                              # Write current time to output array [0]
        raw_data[1].append(check)                               # Write detector status to output array [1]
        if name == 'PIR-1' or name == 'PIR-2': time.sleep(0.1)  # Set sleeping time depending on the sensor type
        if name == 'RW': time.sleep(0.001)
    logger.info(name + ' finished')
    out_raw_data.put(raw_data)                                  # Put collected data to main program



# Pins configuration
# 0 - out pin     1 - LED pin
xBandPins = {'signal_pin': 'P8_12', 'LED_pin': 'P8_11'}         # GPIO assigned for RW sensor
pir1Pins = {'signal_pin': 'P8_15', 'LED_pin': 'P8_13'}          # GPIO assigned for PIR-1 sensor
pir2Pins = {'signal_pin': 'P8_17', 'source_pin': 'P8_18'}       # GPIO assigned for PIR-2 sensor

logger.info('Program start')

# Required configurations for GPIO
GPIO.setup(xBandPins['signal_pin'], GPIO.IN)
GPIO.setup(pir1Pins['signal_pin'], GPIO.IN)
GPIO.setup(pir2Pins['signal_pin'], GPIO.IN)
GPIO.setup(pir2Pins['source_pin'], GPIO.OUT)
GPIO.output(pir2Pins['source_pin'], GPIO.HIGH)

# Create objects for resource sharing
xBand_raw_data_queue = Queue.Queue()
pir1_detect_signal_queue = Queue.Queue()
pir2_detect_signal_queue = Queue.Queue()

# socket part**********************************************************************
HOST = ''
PORT = 5566
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
logger.info('Socket created')
try:
    s.bind((HOST, PORT))
except socket.error as msg:
    logger.error('Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1])
    sys.exit()
while True:
    s.listen(10)
    logger.info('Socket now listening')
    conn, addr = s.accept()
    data = conn.recv(64)
# **********************************************************************************
    # Arguments to recieve by socket:
    args = data.split()
    # args[0] - experiment time duration;
    # args[1] - number of an experiment (e.g. distance to the moving object);

    # if arguments were not received, set some by default
    if len(args) < 2:
        exp_parameter = {'number': 'test', 'duration': 10}
        print '______________________________________________'
        print "parameters are set by default:%snumber - %s%stime duration = %i s" % ('\n', exp_parameter['number'],
                                                                                  '\n', exp_parameter['duration'])
    else:
        exp_parameter = {'duration': int(args[0]),
                         'number': str(args[1])}
        print '______________________________________________'
        print "parameters are set manually:%snumber - %s%stime duration = %i s" % ('\n', exp_parameter['number'],
                                                                                   '\n', exp_parameter['duration'])
    logger.info('Experiment start')

    # Define threads targets and input arguments
    xBandThread = threading.Thread(target=polling, args=(xBandPins, xBand_raw_data_queue, exp_parameter, 'RW'))
    pir1Thread = threading.Thread(target=polling, args=(pir1Pins, pir1_detect_signal_queue, exp_parameter, 'PIR-1'))
    pir2Thread = threading.Thread(target=polling, args=(pir2Pins, pir2_detect_signal_queue, exp_parameter, 'PIR-2'))

    # Start threads for RW, PIR-2 and PIR-2 sensors
    xBandThread.start()
    pir1Thread.start()
    pir2Thread.start()

    # Wait for threads finishing
    xBandThread.join()
    pir1Thread.join()
    pir2Thread.join()

    conn.close()
    logger.info('Connection closed')

    # Get collected data from threads
    xBand_raw_data = xBand_raw_data_queue.get()
    pir1_detect_signal = pir1_detect_signal_queue.get()
    pir2_detect_signal = pir2_detect_signal_queue.get()


    # results writing into 'args[3].data' file
    file = open("/root/ex_data/plot_data_%s.data" % (exp_parameter['number']), "w")
    file.write("exp_parameter" + '\n')
    sym = ' '
    file.write(str(exp_parameter['number']) + sym + str(exp_parameter['duration']) + '\n')
    file.write("/end_of_exp_parameter" + '\n')

    file.write("row_data" + '\n')
    for index in range(len(xBand_raw_data[0])): file.write(str(xBand_raw_data[0][index]) + sym +
                                                           str(xBand_raw_data[1][index]) + "\n")
    file.write("/end_of_row_data" + '\n')

    file.write("pir1_detect_signal" + '\n')
    for index in range(len(pir1_detect_signal[0])): file.write(str(pir1_detect_signal[0][index]) + sym +
                                                               str(pir1_detect_signal[1][index]) + "\n")
    file.write("/end_of_pir1_detect_signal" + '\n')

    file.write("pir2_detect_signal" + '\n')
    for index in range(len(pir2_detect_signal[0])): file.write(str(pir2_detect_signal[0][index]) + sym +
                                                               str(pir2_detect_signal[1][index]) + "\n")
    file.write("/end_of_pir2_detect_signal" + '\n')
    file.close()

s.close()
