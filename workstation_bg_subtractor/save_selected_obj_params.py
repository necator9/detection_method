#!/usr/bin/env python
import sqlite3
import readline

OLD_DB_PATH = "/home/ivan/experiments/chaos_out_clashe_8.0/chaos.db"
NEW_DB_PATH = "/home/ivan/experiments/chaos_out_clashe_8.0/obj_magnification_3.db"

conn = sqlite3.connect(OLD_DB_PATH)
c = conn.cursor()

new_conn = sqlite3.connect(NEW_DB_PATH)
newc = new_conn.cursor()
newc.execute('''CREATE TABLE {} (Img_name TEXT, Obj_id INT, Status TEXT, Base_status TEXT, Br_status TEXT,  
                             Rect_coeff REAL, Extent_coeff REAL, Br_ratio REAL, hw_ratio REAL, 
                             Contour_area REAL, Rect_area INT, Rect_perimeter INT, Br_cross_area INT, 
                             x INT, y INT, w INT, h INT )'''.format("Data_for_"))

if __name__ == "__main__":
    readline.parse_and_bind("tab: complete")
    try:
        while True:
            args = raw_input("Image name:")
            # Split by space sign
            args = args.split(" ")

            if args[0] == "":
                continue

            else:
                for i, obj in enumerate(args):
                    if i > 0:
                        c.execute('SELECT * FROM Data_for_ WHERE Img_name=? AND Obj_id=?', (args[0], obj))
                        params =  c.fetchone()

                        newc.execute('''INSERT INTO {}(Img_name, Obj_id, Status, Base_status, Br_status,  Rect_coeff, 
                                          Extent_coeff, Br_ratio, hw_ratio, Contour_area, Rect_area, Rect_perimeter, 
                                          Br_cross_area, x, y, w, h) 
                                          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'''.format("Data_for_"), list(params))
                        new_conn.commit()

    except KeyboardInterrupt:
        print "\nExiting..."
	conn.close()
        new_conn.close()
