from statistic_lib import Statistic
import time
import threading
import Queue
import copy
import collections

a = collections.deque()






#in_ar = {"PIR": {"col_name": ["Time", "Value"], "queue": a}}

event = threading.Event()
event.set()

#stat = Statistic( event, in_ar, commit_interval=40)
b = copy.copy(a)
#stat.start()
#b = a
i = 0
try:
    while True:
        a.append((time.time(), i))
        time.sleep(0.1)

        print "a", len(a)

        print  b.pop(),len(b)

        i += 1
except KeyboardInterrupt:
    event.clear()
event.clear()
