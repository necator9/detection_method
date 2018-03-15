import sqlite3
import os
import config
import pickle
import threading

# import csv


class Saver(threading.Thread):
    def __init__(self, data_frame_q):
        super(Saver, self).__init__(name="Saver")
        self.running = bool()
        self.data_frame_q = data_frame_q
        self.db_obj = Database(self.__gen_name("sql_database"))
        # self.csv_obj = Csv(self.__gen_name("csv_file"))
        self.pickle_obj = PickleWrap(self.__gen_name("pickle_data.pkl"))

    def run(self):
        self.running = True

        while self.running:

            data_frame = self.data_frame_q.get()
            # print data_frame
            self.db_obj.write(data_frame)
            self.pickle_obj.add(data_frame)

    def quit(self):
        self.running = False
        self.db_obj.quit()
        self.pickle_obj.quit()

    @staticmethod
    def __gen_name(name):
        i = 0
        while True:
            name_plus_counter = ("{0}{1}" + name).format(str(i).zfill(3), "_")
            path_plus_name = os.path.join(config.OUT_DIR, name_plus_counter)
            if not os.path.exists(path_plus_name):
                return path_plus_name
            else:
                i += 1


class Database(object):
    def __init__(self, db_name):
        if config.WRITE_TO_DB:
            self.db_name = db_name
            self.table_name = "Data_for_" + config.OUT_DIR.split("/")[-1]
            # logger.info("Database name: %s" % db_name)
            self.db = sqlite3.connect(self.db_name, check_same_thread=False)
            self.cur = self.db.cursor()
            self.write_table()
        else:
            self.write = (lambda (arg): None)
            self.quit = (lambda: None)

    def write_table(self):

        self.cur.execute('''CREATE TABLE {} (Img_name TEXT, Obj_id INT, Status TEXT, Base_status TEXT, Br_status TEXT,  
                                        Rect_coeff REAL, Extent_coeff REAL, Br_ratio REAL, hw_ratio REAL, 
                                        Contour_area REAL, Rect_area INT, Rect_perimeter INT, Br_cross_area INT, 
                                        x INT, y INT, w INT, h INT )'''.format(self.table_name))

        self.db.commit()

    def write(self, d_frame):

        img_name = str(config.COUNTER)
        db_arr = self.get_base_params(d_frame.base_objects, img_name)

        self.cur.executemany('''INSERT INTO {}(Img_name, Obj_id, Status, Base_status, Br_status,  Rect_coeff, 
                                          Extent_coeff, Br_ratio, hw_ratio, Contour_area, Rect_area, Rect_perimeter, 
                                          Br_cross_area, x, y, w, h) 
                                          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'''.format(self.table_name), db_arr)

        if len(d_frame.ex_objects) > 0:
            img_name += "_split"
            db_split_arr = self.get_base_params(d_frame.ex_objects, img_name)
            self.cur.executemany('''INSERT INTO {}(Img_name, Obj_id, Status, Base_status, Br_status, Rect_coeff, 
                                              Extent_coeff, Br_ratio, hw_ratio, Contour_area, Rect_area, Rect_perimeter, 
                                              Br_cross_area, x, y, w, h) 
                                              VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'''.format(
                self.table_name),
                            db_split_arr)
        print "write"

        self.db.commit()

    @staticmethod
    def get_base_params(objects, img_name):

        # img_name = str(config.COUNTER).zfill(4)
        db_arr = list()
        if len(objects) > 0:
            for obj in objects:
                db_arr.append(
                    [img_name, obj.obj_id, str(obj.gen_status), str(obj.base_status), str(obj.br_status), obj.rect_coef,
                     obj.extent, obj.br_ratio,
                     obj.h_w_ratio, obj.contour_area, obj.rect_area, obj.rect_perimeter, obj.br_cr_area,
                     obj.base_rect[0], obj.base_rect[1], obj.base_rect[2], obj.base_rect[3]])

        return db_arr

    def quit(self):
        print "quit"
        self.db.close()


class PickleWrap(object):
    def __init__(self, pickle_name):
        if config.WRITE_TO_PICKLE:
            self.pickle_name = pickle_name
            self.pickle_data = list()
        else:
            self.add = (lambda (arg): None)
            self.quit = (lambda: None)

    def add(self, data_frame):
        self.pickle_data.append(data_frame.base_objects)

    def quit(self):
        with open(self.pickle_name, 'wb') as output:
            pickle.dump(self.pickle_data, output, pickle.HIGHEST_PROTOCOL)


# class Csv(object):
#     def __init__(self, name):
#         self.name = name + ".csv"
#         fieldnames = ["Img_name", "Object_no", "Status", "Rect_coeff", "hw_ratio", "Contour_area", "Rect_area",
#                       "Rect_perimeter", "Extent", "x", "y", "w", "h"]
#         self.f = open(name, 'w')
#         self.writer = csv.DictWriter(self.f, fieldnames=fieldnames)
#         self.writer.writeheader()
#
#     def write(self, base_objects, img_name):
#         for i, obj in enumerate(base_objects):
#             self.writer.writerow({"Img_name": img_name, "Object_no": i + 1, "Status": obj.obj_status,
#                                   "Rect_coeff": obj.rect_coef, "hw_ratio": obj.h_w_ratio,
#                                   "Contour_area": obj.contour_area, "Rect_area": obj.rect_area,
#                                   "Rect_perimeter": obj.rect_perimeter, "Extent": obj.extent,
#                                   "x": obj.x, "y": obj.y, "w": obj.w, "h": obj.h})
#
#     def quit(self):
#         self.f.close()


