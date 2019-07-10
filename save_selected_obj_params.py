#!/usr/bin/env python
import sqlite3
import readline
import numpy as np

OLD_DB_PATH = "/home/ivan/experiments/bicyclist_out_d/018_sql_database"
NEW_DB_PATH = "/home/ivan/experiments/bicyclist_out.db"


def get_sql_column(db_name):
    conn = sqlite3.connect(db_name)
    conn.row_factory = lambda cursor, row: row[0]
    c = conn.cursor()
    image_name = c.execute('SELECT Img_name FROM Data_for_').fetchall()

    return np.array(image_name)

def create_attach_db(db_path):
    conn = sqlite3.connect(db_path)
    curs = conn.cursor()
    try:
        curs.execute('''CREATE TABLE {} (Img_name TEXT, Obj_id INT, Status TEXT, Base_status TEXT, Br_status TEXT,  
                                     Rect_coeff REAL, Extent_coeff REAL, Br_ratio REAL, hw_ratio REAL, 
                                     Contour_area REAL, Rect_area INT, Rect_perimeter INT, Br_cross_area INT, 
                                     x INT, y INT, w INT, h INT )'''.format("Data_for_"))
    except Exception:
        print 'Such table already exists'

    return conn, curs

if __name__ == "__main__":
    readline.parse_and_bind("tab: complete")
    try:
        while True:
            args = raw_input("Enter the operation mode:")
            # Split by space sign
            args = args.split(" ")

            if args[0] == "a":
                print('In the add mode')
                img_names = get_sql_column(OLD_DB_PATH)

                conn = sqlite3.connect(OLD_DB_PATH)
                c = conn.cursor()
                new_conn = sqlite3.connect(NEW_DB_PATH)
                newc = new_conn.cursor()
                try:
                    newc.execute('''CREATE TABLE {} (Img_name TEXT, Obj_id INT, Status TEXT, Base_status TEXT, Br_status TEXT,  
                                                 Rect_coeff REAL, Extent_coeff REAL, Br_ratio REAL, hw_ratio REAL, 
                                                 Contour_area REAL, Rect_area INT, Rect_perimeter INT, Br_cross_area INT, 
                                                 x INT, y INT, w INT, h INT )'''.format("Data_for_"))
                except Exception:
                    print 'Such table already exists'

                for i, image in enumerate(img_names):
                    try:
                        if image == img_names[i + 1]:
                            continue
                    except IndexError:
                        break

                    args1 = raw_input('Processing the {} image:'.format(image))
                    args1 = args1.split(" ")

                    if args1[0] == '':
                        continue

                    if args1[0] == 'back':
                        break

                    for obj in args1:
                        c.execute('SELECT * FROM Data_for_ WHERE Img_name=? AND Obj_id=?', (image, obj))
                        params =  c.fetchone()

                        newc.execute('''INSERT INTO {}(Img_name, Obj_id, Status, Base_status, Br_status,  Rect_coeff, 
                                          Extent_coeff, Br_ratio, hw_ratio, Contour_area, Rect_area, Rect_perimeter, 
                                          Br_cross_area, x, y, w, h) 
                                          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'''.format("Data_for_"), list(params))
                        new_conn.commit()

            if args[0] == "s":
                print('In the selective add mode')
                conn = sqlite3.connect(OLD_DB_PATH)
                c = conn.cursor()

                new_conn, new_curs = create_attach_db(NEW_DB_PATH)

                while True:
                    args1 = raw_input('Image name and objects: ')
                    args1 = args1.split(" ")

                    if args1[0] == '':
                        continue

                    if args1[0] == 'back':
                        break

                    for i, obj in enumerate(args1):
                        if i > 0:
                            try:
                                c.execute('SELECT * FROM Data_for_ WHERE Img_name=? AND Obj_id=?', (args1[0], obj))
                                params =  c.fetchone()

                                new_curs.execute('''INSERT INTO {}(Img_name, Obj_id, Status, Base_status, Br_status,  Rect_coeff, 
                                                  Extent_coeff, Br_ratio, hw_ratio, Contour_area, Rect_area, Rect_perimeter, 
                                                  Br_cross_area, x, y, w, h) 
                                                  VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'''.format("Data_for_"), list(params))
                                new_conn.commit()

                            except Exception as err:
                                print err
                                continue

    except KeyboardInterrupt:
        print "\nExiting..."
        try:
            conn.close()
            new_conn.close()
        except Exception:
            pass
