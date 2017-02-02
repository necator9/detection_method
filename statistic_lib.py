#   statistic_lib.py
#
#   The class provides data collection functionality for multi threading usage.
#   All data stored in sqlite <filename>.db file
#   (by default - sen_info_0.db)
#
#   Input arguments:
#   stop - stop event for the class
#   args - array of names and queues for writing
#   in_parameters - incoming dictionary with queues and names (has to be created in parent process)
#   base_name = 'sen_info_0' - name of a database by default
#   buf_size=10000 - amount of samples which are collected to a memory before writing to a file
#   commit_interval=60 - time interval between commits to database in seconds
#
#   Important:
#   in_parameters - should be formatted as multidimensional dictionary as shown bellow
#   in_parameters = {"PIR": {"col_name": ["Time", "Value"], "queue": Queue.Queue()}
#                    .............}
#                   where, "PIR" - name of the data table
#                         "col_name" - key for columns names, has not be modified
#                         "["Time", "Value"]" - names of columns
#                         "queue" - key for queues, has not be modified
#                         "Queue.Queue()" - queue with sensor data
#   Amount of sensors in dictionary is not limited (theoretically, only one limitation - CPU performance)
#   Sample of the queue item has to contain such amount of elements as table's columns amount


#   Author: Ivan Matveev
#   E-mail: i.matveev@emw.hs-anhalt.de
#   Date: 17.11.2016


import threading
import Queue
import sqlite3 as lite
import os
import time

import logging

logger = logging.getLogger(__name__)


class Statistic(threading.Thread):
    def __init__(self, stop, in_parameters, base_name='sen_info_0', buf_size=10000, commit_interval=60):
        threading.Thread.__init__(self, name="Main thread")

        self.in_parameters = in_parameters
        self.buffered_qs = dict.fromkeys(self.in_parameters)
        for key in self.buffered_qs: self.buffered_qs[key] = Queue.Queue()

        self.commit_interval = commit_interval
        self.buf_size = buf_size

        self.stop_event = stop
        self.base_name = base_name

        self.internal_stop = threading.Event()
        self.internal_stop.set()

    def writer(self):
        logger.info("Started")
        conn = lite.connect(self.base_name)
        cur = conn.cursor()

        records_num = dict.fromkeys(self.buffered_qs)
        for key in records_num: records_num[key] = 0

        for name in self.in_parameters:
            cur.execute("CREATE TABLE %s (%s REAL)" % (name, self.in_parameters[name]["col_name"][0]))
            logger.debug("%s table created" % name)

            for x in range(len(self.in_parameters[name]["col_name"])):
                if x != 0:
                    cur.execute("ALTER TABLE %s ADD COLUMN %s REAL" % (name, self.in_parameters[name]["col_name"][x]))

        st_time = time.time()
        while True:

            for name in self.in_parameters:
                try:
                    packet = self.buffered_qs[name].get(timeout=3)
                    records_num[name] += len(packet)
                    cur.executemany("INSERT INTO %s VALUES(%s)"
                                    % (name, ("?, " * len(self.in_parameters[name]["col_name"]))[:-2]), packet)
                except Queue.Empty: logger.debug("%s queue timeout" % name)

            qs_counter = 0
            for name in self.buffered_qs:
                qs_counter += self.buffered_qs[name].qsize()

            if (time.time() - st_time) > self.commit_interval:
                st_time = time.time()
                conn.commit()
                logger.info("Commit performed")

            if not self.internal_stop.isSet() and qs_counter == 0: break

        for name in records_num:
            logger.info("Number of records to database: %s = %s" % (name, records_num[name]))
        conn.commit()
        conn.close()
        logger.info("Finished")

    def wrapper(self, in_q):
        temp = []
        while len(temp) < self.buf_size:
            try: temp.append(in_q.get(timeout=3))
            except Queue.Empty: return temp, False
        return temp, True

    def buffering(self, in_q, out_q):
        logger.info("Started")
        while True:
            packet, flag = self.wrapper(in_q)

            if not flag:
                logger.warning("Packet wrapper timeout")
            out_q.put(packet)

            if not self.stop_event.isSet() and in_q.qsize() == 0: break

        logger.debug("Items in queue rest  " + str(in_q.qsize()))
        logger.info("Finished")

    def check_on_file(self):
        logger.debug("Check file on existence")
        nm_b = 0
        while nm_b < 1000:
            self.base_name = 'sen_info_%s.db' % nm_b
            if os.path.exists(self.base_name):
                nm_b += 1
                logger.debug("File exists, number is incremented: <filename>_%s" % nm_b)
            else:
                logger.info("Database filename: %s" % self.base_name)
                break

    def run(self):
        logger.info("START")
        self.check_on_file()

        buf_threads = []
        for key in self.in_parameters:
            thr = threading.Thread(name='%s buffering thread' % key, target=self.buffering,
                                   args=(self.in_parameters[key]["queue"], self.buffered_qs[key]))
            thr.start()
            buf_threads.append(thr)

        wr = threading.Thread(name='Writer thread', target=self.writer)
        wr.start()

        while self.stop_event.is_set(): time.sleep(1)
        logger.info("Stop event received")

        for i in buf_threads: i.join()
        self.internal_stop.clear()
        logger.info("Internal event is cleared")
        wr.join()
        logger.warning("END")
