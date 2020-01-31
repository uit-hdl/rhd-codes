import sqlite3

def connect_split(db):
    conn = sqlite3.connect(db)
    c = conn.cursor()

        
    # Create table for dropped images
    c.execute(''' Create table IF NOT EXISTS dropped_images (
                name TEXT,
                image BLOB,
                height INT,
                width INT,
                reason TEXT
                )''')
    
    c.execute('''
              CREATE INDEX IF NOT EXISTS idx_dropped ON dropped_images (name)
              ''')
    
    
    # Create table for split digits
    c.execute(''' Create table IF NOT EXISTS digit (
                Name TEXT,
                Image BLOB,
                Height INT,
                Width INT
                )''')
    
    c.execute('''
              CREATE INDEX IF NOT EXISTS idx_digit ON digit (name)
              ''')
        
    # Create table for images
    c.execute(''' Create table IF NOT EXISTS fields (
                Name TEXT,
                Image BLOB,
                Height INT,
                Width INT,
                Counter INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT
                )''')
    
    # Creating index for faster sorting and search later
    c.execute(''' 
              CREATE INDEX IF NOT EXISTS idx ON fields (Name);
              ''')
    
    c.connection.commit()

def connect(db):
    
    conn = sqlite3.connect(db)
    c = conn.cursor()
        
    # Create table for images
    c.execute(''' Create table IF NOT EXISTS fields (
                Name TEXT,
                Image BLOB,
                Height INT,
                Width INT
                )''')
    
    # Creating index for faster sorting and search later
    c.execute(''' 
              CREATE INDEX IF NOT EXISTS idx ON fields (Name);
              ''')
    
    c.connection.commit()
