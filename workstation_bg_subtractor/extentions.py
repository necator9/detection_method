import sqlite3
import config


class Saver(object):
    def __init__(self):
        # db_name = self.gen_name("Database")
        db_name = "Database"

        self.db = DbSave(db_name)
        if not config.WRITE_TO_DB:
            self.db.db_write.func_code = (lambda: None).func_code
            # logger.info("Database name: %s" % db_name)

    def save(self, data_frame):
        self.db_write(data_frame)

    def db_write(self, data_frame):
        self.db.db_write(data_frame)

    def no_write(self):
        pass


class DbSave(object):
    def __init__(self, db_name):
        self.db_name = db_name
        self.table_name = str()

    def write_table(self):
        db = sqlite3.connect(self.db_name)
        self.table_name = "Data_for_" + config.OUT_DIR.split("/")[-1]

        cur = db.cursor()

        cur.execute('''CREATE TABLE %s (Img_name TEXT, Obj_id INT, Status TEXT, Base_status TEXT, Br_status TEXT,  Rect_coeff REAL, Extent_coeff REAL, 
                                                    Br_ratio REAL, hw_ratio REAL, Contour_area REAL, Rect_area INT, 
                                                    Rect_perimeter INT, Br_cross_area INT, x INT, y INT, w INT, h INT )'''
                         % self.table_name)

        db.commit()

        self.write_table.func_code = (lambda: None).func_code

    def db_write(self, d_frame):
        self.write_table()

        db = sqlite3.connect(self.db_name)

        img_name = str(config.COUNTER)
        db_arr = self.get_base_params(d_frame.base_objects, img_name)

        cur = db.cursor()

        cur.executemany('''INSERT INTO %s(Img_name, Obj_id, Status, Base_status, Br_status,  Rect_coeff, Extent_coeff, Br_ratio, hw_ratio, 
                                              Contour_area, Rect_area, Rect_perimeter, Br_cross_area, x, y, w, h) 
                                              VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''' % self.table_name, db_arr)

        if len(d_frame.ex_objects) > 0:
            img_name += "_split"
            db_split_arr = self.get_base_params(d_frame.ex_objects, img_name)
            cur.executemany('''INSERT INTO %s(Img_name, Obj_id, Status, Base_status, Br_status, Rect_coeff, Extent_coeff, Br_ratio, hw_ratio, 
                                                         Contour_area, Rect_area, Rect_perimeter, Br_cross_area, x, y, w, h) 
                                                         VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''' % self.table_name,
                                 db_split_arr)

        db.commit()

    @staticmethod
    def get_base_params(objects, img_name):

        # img_name = str(config.COUNTER).zfill(4)
        db_arr = list()
        if len(objects) > 0:
            for obj in objects:
                db_arr.append([img_name, obj.obj_id, str(obj.gen_status), str(obj.base_status), str(obj.br_status), obj.rect_coef, obj.extent, obj.br_ratio,
                               obj.h_w_ratio, obj.contour_area, obj.rect_area, obj.rect_perimeter, obj.br_cr_area,
                               obj.base_rect[0], obj.base_rect[1], obj.base_rect[2], obj.base_rect[3]])

        return db_arr

    def pass_function(self):
        pass

 #   def quit(self):

   #     self.db.close()
