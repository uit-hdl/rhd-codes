import sqlite3
import cv2

from Database import init_db

class DbHandler:
    def __init__(self, db_loc, connection=None):
        self.db_loc = db_loc
        self.connection = connection
        self.cursor = None
        if self.connection is None:
            self.connection = sqlite3.connect(db_loc)
            init_db.connect_split(db_loc)
            
            

# =============================================================================
#     def store_field(self, name, img):
#         # Name, raw byte data, height width
#         self.connection.cursor().execute('insert or IGNORE into fields VALUES (?, ?, ?, ?)', (name, img, img.shape[0], img.shape[1]))
# =============================================================================
		
    def store_field_updated(self, name, img):
        img_bytes = cv2.imencode('.jpg', img)[1].tostring()
        self.connection.cursor().execute('insert or IGNORE into fields VALUES (?, ?, ?, ?, ?)', (name, sqlite3.Binary(img_bytes), img.shape[0], img.shape[1], None))

    def store_digit(self, name, img):
        img_bytes = cv2.imencode('.jpg', img)[1].tostring()
        self.connection.cursor().execute('insert or IGNORE into digit VALUES (?, ?, ?, ?)', (name, sqlite3.Binary(img_bytes), img.shape[0], img.shape[1]))
    
    

    def store_dropped(self, name, reason):
        c = self.connection.cursor()
        c.execute('SELECT * FROM fields WHERE name=:name', {'name': name})
        
        result = c.fetchone()
        
        self.connection.cursor().execute('insert or IGNORE into dropped_images VALUES (?, ?, ?, ?, ?)', (name, result[1], result[2], result[3], reason))

    def test_exists(self, name):
        c = self.connection.cursor()
        c.execute("SELECT EXISTS(SELECT 1 FROM fields WHERE name LIKE :name LIMIT 1)", {'name': name})
        return c.fetchone()

    def test_exists_digit(self, name):
        c = self.connection.cursor()
        c.execute("SELECT EXISTS(SELECT 1 FROM digit WHERE name LIKE :name LIMIT 1)", {'name': "%"+name+"%"})
        return c.fetchone()

    def test_exists_dropped(self, name):
        c = self.connection.cursor()
        c.execute("SELECT EXISTS(SELECT 1 FROM dropped_images WHERE name LIKE :name LIMIT 1)", {'name': "%" + name + "%"})
        return c.fetchone()

    def select_image(self, name):
        c = self.connection.cursor()
        c.execute("SELECT * FROM fields WHERE name=:name", {'name': name})
        return c.fetchone()

    def select_all_images(self):
        return self.connection.cursor().execute("SELECT * FROM fields")

    def count_rows_in_fields(self):
        return self.connection.cursor().execute("SELECT Count(*) FROM fields")

    def count_rows_in_digit(self):
        return self.connection.cursor().execute("SELECT Count(*) FROM digit")

    def count_rows_in_dropped(self):
        return self.connection.cursor().execute("SELECT Count(*) FROM dropped")

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
