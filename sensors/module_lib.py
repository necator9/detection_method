#   statistic_lib.py
#
#   The class for the detection module.
#
#   Following args for constructor have to be defined when class is declared:
#   st_event - stop event for the module
#   pir=False - when True, enables only PIR sensor
#   rw=False - when True, enables only RW sensor
#   control=False - when True, control thread starts (one of sensors has to be activated).Thread is used to control
#   lighting or to form main output signal by means of comparison. If only one sensor is activated, make a decision
#   based on its status

#   Available methods:
#       get_status() - returns resulted motion status of the detection module,
#   can be executed only if control is activated

#       set_fr() - change default polling frequency of a sensor. Frequency has to be declared in Hz
#   eg: def set_fr(pir_fr=10, rw_fr=10)

#       set_statistic_lvl(full=False, pir_pol=False, rw_pol=False, rw_proc=False, control=False) -
#   enables data collection for chosen sensor:
#           full=True - collect all available data from sensors
#           pir_pol=True - collect polling data from PIR sensor
#           rw_pol=True - collect raw polling data from RW sensor
#           rw_proc=True - collect processed data for RW sensor
#   Data collection for chosen sensor is possible, if sensor is defined as class argument
#   (declared while class instance creation)

#   Author: Ivan Matveev
#   E-mail: i.matveev@emw.hs-anhalt.de
#   Date: 30.11.2016

import Queue
import numpy as np
import threading
import time
import logging.config

import Adafruit_BBIO.GPIO as GPIO           # The library for GPIO handling
from statistic_lib import Statistic

logger = logging.getLogger(__name__)


class Module(threading.Thread):
    def __init__(self, st_event, pir=False, rw=False, control=False, full=False):
        threading.Thread.__init__(self, name="Main thread")
        self.start_time = time.time()
        # Flags which define threads to be started
        self.pir_flag = pir
        self.rw_flag = rw
        self.control_flag = control
        if full:
            self.pir_flag = True
            self.rw_flag = True
            self.control_flag = True

        # Attributes for statistic module
        self.st_flag = False                 # Attribute can be modified through set_stat_param() method
        self.st_args = {}

        self.control_qs = {}                 # Dictionary to be passed into control thread

        if self.control_flag:
            self.control_stat = {}
            self.control_sample = False

        if self.pir_flag:
            self.pir_gpio = {'signal_pin': 'P8_15', 'LED_pin': 'P8_13'}     # GPIOs are connected to the sensor
            self.pir_polling_qs = {"polling": Queue.Queue()}                # Q for polling thread, will be updated
            self.control_qs.update({"PIR": self.pir_polling_qs["polling"]}) # The same Q for control thread
            self.pir_tm = 0.1                                               # Sleep time for sensor by default

        if self.rw_flag:
            self.rw_gpio = {'signal_pin': 'P8_12', 'LED_pin': 'P8_18'}
            self.rw_polling_qs = {"polling": Queue.Queue()}                 # Raw RW Q for processing thread
            self.rw_processing_qs = {"processing": Queue.Queue()}           # Processed RW signal
            self.control_qs.update({"RW": self.rw_processing_qs["processing"]}) # Processed RW signal for control thread
            self.rw_tm = 0.001

        self.stop_ev = st_event                                             # External stop event

    # Returns resulted motion status
    def get_status(self):
        if self.control_flag: return self.control_sample
        else:
            logger.error("Control thread is not activated, resulted data is no available")
            return False
    # Method to change default polling frequency
    def set_fr(self, pir_fr=10, rw_fr=100):
        self.pir_tm = 1 / pir_fr
        self.rw_tm = 1 / rw_fr

    def set_stat_param(self, name, q, args, clms):                          # For set_statistic_lvl()
        self.st_flag = True
        q.update({"statistic": Queue.Queue()})
        args.update({name: {"col_name": clms, "queue": q["statistic"]}})

    # Method defines which sensors data will be collected and stored
    def set_statistic_lvl(self, full=False, pir_pol=False, rw_pol=False, rw_proc=False, control=False):
        if full:
            pir_pol = True
            rw_pol = True
            rw_proc = True
            control = True

        if pir_pol and self.pir_flag:
            self.set_stat_param("PIR_polling", self.pir_polling_qs, self.st_args, ["Time", "Value"])
        if pir_pol and not self.pir_flag:
            logger.warning("PIR sensor is not defined in the module, the data collection is not possible")

        if rw_pol and self.rw_flag:
            self.set_stat_param("RW_polling", self.rw_polling_qs, self.st_args, ["Time", "Value"])
        if rw_pol and not self.rw_flag:
            logger.warning("RW sensor is not defined in the module, the data collection is not possible")

        if rw_proc and self.rw_flag:
            self.set_stat_param("RW_processing", self.rw_processing_qs, self.st_args, ["Time", "Value"])
        if rw_proc and not self.rw_flag:
            logger.warning("RW processing is not defined in the module, the data collection is not possible")

        if control and self.control_flag:
            self.set_stat_param("Control", self.control_stat, self.st_args, ["Time", "Value"])
        if control and (not self.rw_flag or not self.pir_flag):
            logger.warning("Control is not defined in the module, the data collection is not possible")

    # Used as a threads for polling
    def polling(self, gpio, qs, tm):
        logger.info("Started")
        while self.stop_ev.isSet():
            sample = [time.time() - self.start_time, GPIO.input(gpio['signal_pin'])]
            for key in qs:
                qs[key].put(sample)                                       # Put value to corresponding q
            time.sleep(tm)
        logger.info("Finished")

    # Processing of data from rw sensor
    def rw_processing(self):
        logger.info("Started")
        f_buffer_time = []
        f_buffer_data = []
        s_buffer = []
        result_buffer_fr = []
        result_buffer_time = []
        while self.stop_ev.isSet():
            try:
                check = self.rw_polling_qs["polling"].get(timeout=3)      # Get value from appropriate polling q

                f_buffer_time.append(check[0])
                f_buffer_data.append(check[1])

                if len(f_buffer_data) == 300:
                    for i in range(len(f_buffer_data) - 1):
                        if f_buffer_data[i + 1] > f_buffer_data[i]:
                            s_buffer.append(f_buffer_time[i + 1])
                    if len(s_buffer) > 1:
                        for k in range(len(s_buffer) - 1):
                            freq = 1 / (s_buffer[k + 1] - s_buffer[k])
                            result_buffer_fr.append(freq)
                            result_buffer_time.append(s_buffer[k + 1])
                        mean_vol = np.mean(result_buffer_fr)
                        result_buffer_fr = []
                    else:
                        mean_vol = 0
                    # Put calculated value to out queues
                    for x in self.rw_processing_qs:
                        self.rw_processing_qs[x].put([time.time() - self.start_time, mean_vol])

                    s_buffer = []
                    f_buffer_time = []
                    f_buffer_data = []

            except Queue.Empty:
                logger.info("RW queue timeout")
        logger.info("Finished")

    #   Thread to control lighting or to form main output signal by means of comparison.
    #   If only one sensor is activated, make a decision based on its status
    def control(self, in_qs):
        logger.info("Started")
        qs = dict.fromkeys(in_qs)
        light = False
        while self.stop_ev.isSet():
            # Receive all available queues
            for name in qs:
                try:
                    qs[name] = (in_qs[name].get(timeout=3))
                except Queue.Empty:
                    logger.info("%s queue timeout" % name)
            # If both sensors are activated
            if len(qs) == 2:
                # qs["PIR"][1] or qs["RW"][1] - current motion statuses of a sensors
                if qs["PIR"][1] > 0 and qs["RW"][1] > 0:
                    light = True
                else:
                    light = False

            # If only sensor is activated
            if len(qs) == 1:
                for name in qs:
                    if name == "PIR":
                        if qs["PIR"][1] > 0:
                            light = True
                        else:
                            light = False
                    if name == "RW":
                        if qs["RW"][1] > 0:
                            light = True
                        else:
                            light = False

            if light:
                GPIO.output('P8_18', GPIO.HIGH)      # To do, optimization is possible (avoid double light activation)
                self.control_sample = [time.time() - self.start_time, 1]
            else:
                GPIO.output('P8_18', GPIO.LOW)
                self.control_sample = [time.time() - self.start_time, 0]

            # If True, put data to Statistic module
            #if self.st_flag:
                #self.control_stat["statistic"].put([time.time() - self.start_time, self.control_sample])

        logger.info("Finished")

    # Method starts threads depending on set flags
    def run(self):
        if not self.pir_flag and not self.rw_flag:
            self.stop_ev.clear()
            logger.error("No sensors are specified. Exit")
            return 1

        # PIR polling thread
        if self.pir_flag:
            GPIO.setup(self.pir_gpio['signal_pin'], GPIO.IN)
            pir_polling = threading.Thread(name='Polling PIR', target=self.polling,
                                           args=(self.pir_gpio, self.pir_polling_qs, self.pir_tm))
            pir_polling.start()

        # RW polling and processing threads
        if self.rw_flag:
            GPIO.setup(self.rw_gpio['signal_pin'], GPIO.IN)
            rw_polling = threading.Thread(name='Polling RW', target=self.polling,
                                          args=(self.rw_gpio, self.rw_polling_qs, self.rw_tm))
            rw_processing = threading.Thread(name='Rw processing ', target=self.rw_processing)
            rw_polling.start()
            rw_processing.start()

        # Control thread
        if self.control_flag:
            GPIO.setup('P8_18', GPIO.OUT)
            control_thread = threading.Thread(name='Control thread', target=self.control, args=(self.control_qs,))
            control_thread.start()

        # Start of the statistic module
        if self.st_flag:
            st_module = Statistic(self.stop_ev, self.st_args)
            st_module.start()

        # Waiting for a stop event
        while self.stop_ev.is_set():
            time.sleep(1)
        logger.info("Stop event received")

        if self.pir_flag:           # To do, perform other implementation
            pir_polling.join()
        if self.rw_flag:
            rw_polling.join()
            rw_processing.join()
        if self.control_flag:
            control_thread.join()
        if self.st_flag:
            st_module.join()
        logger.info("Finished")
