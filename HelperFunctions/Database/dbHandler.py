import sqlite3
import cv2

from Database import init_db

 

class DbHandler:
    def __init__(self, db_loc, connection=None, train = False, split = False, full = False, validate = False):
        self.db_loc = db_loc
        self.connection = connection
        self.cursor = None
        self.train = train
        self.split = split
        self.validate = validate
        self.full = full
        if self.connection is None:
            self.connection = sqlite3.connect(db_loc)
            
        if self.train is True:
            init_db.connect_train(db_loc)
        if self.split is True:
            init_db.connect_split(db_loc)
        if self.validate is True:
            init_db.connect_validate(db_loc)
        if self.full is True:
            init_db.annote_3digit(db_loc)
            
            

# =============================================================================
#     def store_field(self, name, img):
#         # Name, raw byte data, height width
#         self.connection.cursor().execute('insert or IGNORE into fields VALUES (?, ?, ?, ?)', (name, img, img.shape[0], img.shape[1]))
# =============================================================================
		
    def store_field_updated(self, name, img):
        img_bytes = cv2.imencode('.jpg', img)[1].tostring()
        self.connection.cursor().execute('insert or IGNORE into fields VALUES (?, ?, ?, ?)', (name, sqlite3.Binary(img_bytes), img.shape[0], img.shape[1]))
        
        
    def test_exists(self, name):
        c = self.connection.cursor()
        c.execute("SELECT EXISTS(SELECT 1 FROM fields WHERE name LIKE :name LIMIT 1)", {'name': name})
        return c.fetchone()
        
    def store_original(self, name, img):
        img_bytes = cv2.imencode('.jpg', img)[1].tostring()
        self.connection.cursor().execute('INSERT OR IGNORE into original (name, image) VALUES (?, ?)', (name, sqlite3.Binary(img_bytes)))
        self.connection.commit()
        
    def store_cells(self, name, imgs, row, code, source):
        orig_bytes = cv2.imencode('.jpg', imgs[0])[1].tostring()
        bw_bytes = cv2.imencode('.jpg', imgs[1])[1].tostring()
        
        if len(imgs) > 2:
            grey_bytes = cv2.imencode('.jpg', imgs[2])[1].tostring()
    
            self.connection.cursor().execute('INSERT OR IGNORE into cell_grey (name, image, row, actual_digits, source) VALUES (?, ?, ?, ?, ?)', (name, sqlite3.Binary(grey_bytes), row, code, source))
        
        self.connection.cursor().execute('INSERT OR IGNORE into cell_orig (name, image, row, actual_digits, source) VALUES (?, ?, ?, ?, ?)', (name, sqlite3.Binary(orig_bytes), row, code, source))
        self.connection.cursor().execute('INSERT OR IGNORE into cell_bw (name, image, row, actual_digits, source) VALUES (?, ?, ?, ?, ?)', (name, sqlite3.Binary(bw_bytes), row, code, source))    
        
        self.connection.commit()
        
    def store_cells_validation(self, name, imgs, row, source):
        orig_bytes = cv2.imencode('.jpg', imgs[0])[1].tostring()
        bw_bytes = cv2.imencode('.jpg', imgs[1])[1].tostring()
        
        if len(imgs) > 2:
            grey_bytes = cv2.imencode('.jpg', imgs[2])[1].tostring()
            self.connection.cursor().execute('INSERT OR IGNORE into cell_grey (name, image, row, source) VALUES (?, ?, ?, ?)', (name, sqlite3.Binary(grey_bytes), row, source))
        
        self.connection.cursor().execute('INSERT OR IGNORE into cell_orig (name, image, row, source) VALUES (?, ?, ?, ?)', (name, sqlite3.Binary(orig_bytes), row, source))
        self.connection.cursor().execute('INSERT OR IGNORE into cell_bw (name, image, row, source) VALUES (?, ?, ?, ?)', (name, sqlite3.Binary(bw_bytes), row, source))   
        
        
    def store_cells_one_table(self, name, imgs, row, code, source):
        orig_bytes = cv2.imencode('.jpg', imgs[0])[1].tostring()
        bw_bytes = cv2.imencode('.jpg', imgs[1])[1].tostring()
        
        if len(imgs) > 2:
            grey_bytes = cv2.imencode('.jpg', imgs[2])[1].tostring()
        
            self.connection.cursor().execute('INSERT OR IGNORE into cells (name, original, black_white, greyscale, row, code, source) VALUES (?, ?, ?, ?, ?, ?, ?)', 
                                   (name, sqlite3.Binary(orig_bytes), sqlite3.Binary(bw_bytes), sqlite3.Binary(grey_bytes), row, code, source))
            self.connection.commit()
            
        else:
            self.connection.cursor().execute('INSERT OR IGNORE into cells (name, original, black_white, row, code, source) VALUES (?, ?, ?, ?, ?, ?)', 
                                   (name, sqlite3.Binary(orig_bytes), sqlite3.Binary(bw_bytes), row, code, source))
            self.connection.commit()
        
    def store_single_splits_training(self, name, imgs, row, position, code, nr_digits, source):
        orig_bytes = cv2.imencode('.jpg', imgs[0])[1].tostring()
        bw_bytes = cv2.imencode('.jpg', imgs[1])[1].tostring()
        grey_bytes = cv2.imencode('.jpg', imgs[2])[1].tostring()
        
        self.connection.cursor().execute('INSERT OR IGNORE into split_orig (name, image, row, position, actual_digits, number_of_digits, source) VALUES (?, ?, ?, ?, ?, ?, ?)',
                               (name, sqlite3.Binary(orig_bytes), row, position, code, nr_digits, source))
        self.connection.cursor().execute('INSERT OR IGNORE into split_bw (name, image, row, position, actual_digits, number_of_digits, source) VALUES (?, ?, ?, ?, ?, ?, ?)',
                               (name, sqlite3.Binary(bw_bytes), row, position, code, nr_digits, source))
        self.connection.cursor().execute('INSERT OR IGNORE into split_grey (name, image, row, position, actual_digits, number_of_digits, source) VALUES (?, ?, ?, ?, ?, ?, ?)',
                               (name, sqlite3.Binary(grey_bytes), row, position, code, nr_digits, source))
        
        self.connection.commit()
        
    def store_single_splits_validation(self, name, imgs, row, position, nr_digits, source):
        orig_bytes = cv2.imencode('.jpg', imgs[0])[1].tostring()
        bw_bytes = cv2.imencode('.jpg', imgs[1])[1].tostring()
        grey_bytes = cv2.imencode('.jpg', imgs[2])[1].tostring()
        
        self.connection.cursor().execute('INSERT OR IGNORE into split_orig (name, image, row, position, number_of_digits, source) VALUES (?, ?, ?, ?, ?, ?)',
                               (name, sqlite3.Binary(orig_bytes), row, position, nr_digits, source))
        self.connection.cursor().execute('INSERT OR IGNORE into split_bw (name, image, row, position, number_of_digits, source) VALUES (?, ?, ?, ?, ?, ?)',
                               (name, sqlite3.Binary(bw_bytes), row, position, nr_digits, source))
        self.connection.cursor().execute('INSERT OR IGNORE into split_grey (name, image, row, position, number_of_digits, source) VALUES (?, ?, ?, ?, ?, ?)',
                               (name, sqlite3.Binary(grey_bytes), row, position, nr_digits, source))
        
        self.connection.commit()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


    def store_digit(self, name, img):
        img_bytes = cv2.imencode('.jpg', img)[1].tostring()
        self.connection.cursor().execute('insert or IGNORE into digit VALUES (?, ?, ?, ?)', (name, sqlite3.Binary(img_bytes), img.shape[0], img.shape[1]))
    
    
    def store_dropped(self, name, reason):
        c = self.connection.cursor()
        c.execute('SELECT * FROM fields WHERE name=:name', {'name': name})
        
        result = c.fetchone()
        
        self.connection.cursor().execute('insert or IGNORE into dropped_images VALUES (?, ?, ?, ?, ?)', (name, result[1], result[2], result[3], reason))



    def test_exists_digit(self, name):
        c = self.connection.cursor()
        c.execute("SELECT EXISTS(SELECT 1 FROM digit WHERE name LIKE :name LIMIT 1)", {'name': "%"+name+"%"})
        return c.fetchone()

    def test_exists_dropped(self, name):
        c = self.connection.cursor()
        c.execute("SELECT EXISTS(SELECT 1 FROM dropped_images WHERE name LIKE :name LIMIT 1)", {'name': "%" + name + "%"})
        return c.fetchone()
    
    def test_exists_any(self, name, table):
        c = self.connection.cursor()
        query = 'SELECT * FROM {} WHERE name LIKE "{}" LIMIT 1'.format(table, name)
        c.execute(query)
        return c.fetchone()
    
    def test_exists_any_source(self, source, table):
        c = self.connection.cursor()
        query = 'SELECT * FROM {} WHERE source LIKE "{}"'.format(table, source)
        c.execute(query)
        
        return c.fetchone()

    def select_image(self, name):
        c = self.connection.cursor()
        c.execute("SELECT * FROM fields WHERE name=:name", {'name': name})
        return c.fetchone()
    
    def select_image_any(self, name, table):
        c = self.connection.cursor()
        query = 'SELECT * FROM "{}" WHERE name = "{}"'.format(table, name)
        c.execute(query)
        return c.fetchone()

    def select_all_images(self):
        return self.connection.cursor().execute("SELECT * FROM fields")
    
    def select_all_training_images(self, table):
        query = 'SELECT * FROM {} ORDER BY actual_digits'.format(table)
        
        c = self.connection.cursor()
        c.execute(query)
        return c.fetchall()
        
    def select_all_image_names_any(self, table):
        query = 'SELECT Name FROM "{}"'.format(table)
        
        c = self.connection.cursor()
        c.execute(query)
        return c.fetchall()
    
    def select_all_images_any(self, table):
        c = self.connection.cursor()
        
        query = 'SELECT * FROM "{}"'.format(table)
        c.execute(query)
        return c.fetchall()
    
    def select_by_actual_digit(self, table, digit):
        query = 'SELECT * FROM "{}" WHERE actual_digits = "{}"'.format(table, digit)
        return self.connection.cursor().execute(query)
    
    def select_by_multiple_digits(self, table, digit_list):
        digit_list = ["1", "2", "3", "4"]
        digits = ', '.join(digit_list)
        
        query = 'SELECT * FROM "{}" WHERE actual_digits = "{}"'.format(table, digits)
        return self.connection.cursor().execute(query)
    
    def remove_by_name(self, table, name):
        query = 'DELETE FROM {} WHERE name = "{}"'.format(table, name)
        
        c = self.connection.cursor()
        c.execute(query)

    def count_rows_in_fields(self):
        return self.connection.cursor().execute("SELECT Count(*) FROM fields")

    def count_rows_in_digit(self):
        return self.connection.cursor().execute("SELECT Count(*) FROM digit")

    def count_rows_in_dropped(self):
        return self.connection.cursor().execute("SELECT Count(*) FROM dropped")
    
    def count_rows_any(self, table):
        query = 'SELECT Count(*) FROM "{}"'.format(table)
        return self.connection.cursor().execute(query)

    def __enter__(self):
        try:
            if self.connection is None:
                self.connection = sqlite3.connect(self.db_loc)
            return self
        except Exception as e:
            print(e)
            exit(1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connection.commit()
        self.connection.close()
        
    def close(self):
        self.connection.close()
