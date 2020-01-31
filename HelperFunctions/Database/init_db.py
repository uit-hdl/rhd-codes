import sqlite3

def connect_train(db):

    
    """ Her skal tabellen(e) som skal holde på de forskjellige treningsbildene og originalene som dem kommer fra lages """
    conn = sqlite3.connect(db)
    c = conn.cursor()
        
    
    # Create table for original census images
    c.execute(''' Create table IF NOT EXISTS original 
              (
              RowID INTEGER PRIMARY KEY not null,
              name TEXT,
              image BLOB,
              date default (datetime('now', 'localtime'))
                )
              ''')
    c.execute(''' CREATE INDEX IF NOT EXISTS idx_original ON original (RowID);
              ''')
    
    # Create table for cell images taken directly from the original census image
    c.execute(''' Create table IF NOT EXISTS cell_orig 
              (
              RowID INTEGER PRIMARY KEY not null,
              name TEXT,
              image BLOB,
              date default current_timestamp,
              row TEXT,
              actual_digits TEXT,
              source TEXT,
              predicted_label TEXT
              )
              ''')
    c.execute(''' CREATE INDEX IF NOT EXISTS idx_cell_orig ON cell_orig (RowID)''')
    
    # Create table for cell images that have been converted to black and white
    c.execute(''' Create table IF NOT EXISTS cell_bw
              (
              RowID INTEGER PRIMARY KEY not null,
              name TEXT,
              image BLOB,
              date default current_timestamp,
              row TEXT,
              actual_digits TEXT,
              source TEXT,
              predicted_label TEXT
              )
              ''')
    c.execute(''' CREATE INDEX IF NOT EXISTS idx_cell_bw ON cell_bw (RowID)''')
    
    # Create table for cell images that have been converted to greyscale
    c.execute(''' Create table IF NOT EXISTS cell_grey
              (
              RowID INTEGER PRIMARY KEY not null,
              name TEXT,
              image BLOB,
              date default current_timestamp,
              row TEXT,
              actual_digits TEXT,
              source TEXT,
              predicted_label TEXT
              )
              ''')
    c.execute(''' CREATE INDEX IF NOT EXISTS idx_cell_grey ON cell_grey (RowID)''')
    
    # Create table for cell images that have been cut directly from the original census image, and split up into separate digits
    c.execute(''' Create table IF NOT EXISTS split_orig 
              (
              RowID INTEGER PRIMARY KEY not null,
              name TEXT,
              image BLOB,
              date default current_timestamp,
              row TEXT,
              position TEXT,
              actual_digits TEXT,
              number_of_digits TEXT,
              source TEXT,
              predicted_label TEXT
              )
              ''')
    c.execute(''' CREATE INDEX IF NOT EXISTS idx_split_orig ON split_orig (RowID) ''')
        
    # Create table for cell images that have been cut directly from the original census image, and split up into separate digits and converted to black and white
    c.execute(''' Create table IF NOT EXISTS split_bw 
              (
              RowID INTEGER PRIMARY KEY not null,
              name TEXT,
              image BLOB,
              date default current_timestamp,
              row TEXT,
              position TEXT,
              actual_digits TEXT,
              number_of_digits TEXT,
              source TEXT,
              predicted_label TEXT
              )
              ''')
    c.execute(''' CREATE INDEX IF NOT EXISTS idx_split_bw ON split_bw (RowID) ''')
    
    # Create table for cell images that have been cut directly from the original census image, and split up into separate digits and converted to greyscale
    c.execute(''' Create table IF NOT EXISTS split_grey 
              (
              RowID INTEGER PRIMARY KEY not null,
              name TEXT,
              image BLOB,
              date default current_timestamp,
              row TEXT,
              position TEXT,
              actual_digits TEXT,
              number_of_digits TEXT,
              source TEXT,
              predicted_label TEXT
              )
              ''')
    c.execute(''' CREATE INDEX IF NOT EXISTS idx_split_grey ON split_grey (RowID) ''')
    
    c.connection.commit()    
    c.close()
    
def connect_validate(db):

    
    """ Her skal tabellen(e) som skal holde på de forskjellige treningsbildene og originalene som dem kommer fra lages """
    conn = sqlite3.connect(db)
    c = conn.cursor()
        
    # Create table for cell images taken directly from the original census image
    c.execute(''' Create table IF NOT EXISTS cell_orig 
              (
              RowID INTEGER PRIMARY KEY not null,
              name TEXT,
              image BLOB,
              date default current_timestamp,
              row TEXT,
              source TEXT,
              predicted_label TEXT
              )
              ''')
    c.execute(''' CREATE INDEX IF NOT EXISTS idx_cell_orig ON cell_orig (RowID)''')
    
    # Create table for cell images that have been converted to black and white
    c.execute(''' Create table IF NOT EXISTS cell_bw
              (
              RowID INTEGER PRIMARY KEY not null,
              name TEXT,
              image BLOB,
              date default current_timestamp,
              row TEXT,
              source TEXT,
              predicted_label TEXT
              )
              ''')
    c.execute(''' CREATE INDEX IF NOT EXISTS idx_cell_bw ON cell_bw (RowID)''')
    
    # Create table for cell images that have been converted to greyscale
    c.execute(''' Create table IF NOT EXISTS cell_grey
              (
              RowID INTEGER PRIMARY KEY not null,
              name TEXT,
              image BLOB,
              date default current_timestamp,
              row TEXT,
              source TEXT,
              predicted_label TEXT
              )
              ''')
    c.execute(''' CREATE INDEX IF NOT EXISTS idx_cell_grey ON cell_grey (RowID)''')
    
    # Create table for cell images that have been cut directly from the original census image, and split up into separate digits
    c.execute(''' Create table IF NOT EXISTS split_orig 
              (
              RowID INTEGER PRIMARY KEY not null,
              name TEXT,
              image BLOB,
              date default current_timestamp,
              row TEXT,
              position TEXT,
              number_of_digits TEXT,
              source TEXT,
              predicted_label TEXT
              )
              ''')
    c.execute(''' CREATE INDEX IF NOT EXISTS idx_split_orig ON split_orig (RowID) ''')
        
    # Create table for cell images that have been cut directly from the original census image, and split up into separate digits and converted to black and white
    c.execute(''' Create table IF NOT EXISTS split_bw 
              (
              RowID INTEGER PRIMARY KEY not null,
              name TEXT,
              image BLOB,
              date default current_timestamp,
              row TEXT,
              position TEXT,
              number_of_digits TEXT,
              source TEXT,
              predicted_label TEXT
              )
              ''')
    c.execute(''' CREATE INDEX IF NOT EXISTS idx_split_bw ON split_bw (RowID) ''')
    
    # Create table for cell images that have been cut directly from the original census image, and split up into separate digits and converted to greyscale
    c.execute(''' Create table IF NOT EXISTS split_grey 
              (
              RowID INTEGER PRIMARY KEY not null,
              name TEXT,
              image BLOB,
              date default current_timestamp,
              row TEXT,
              position TEXT,
              number_of_digits TEXT,
              source TEXT,
              predicted_label TEXT
              )
              ''')
    c.execute(''' CREATE INDEX IF NOT EXISTS idx_split_grey ON split_grey (RowID) ''')
    
    c.connection.commit()    
    c.close()
    
def annote_3digit(db):
    conn = sqlite3.connect(db)
    c = conn.cursor()
    
    # Create table for the manually transcribed cell images that we have as part of our coordinate file
    c.execute(''' Create table IF NOT EXISTS cells 
              (
              RowID INTEGER PRIMARY KEY not null,
              name TEXT,
              original BLOB,
              black_white BLOB,
              greyscale BLOB,
              row TEXT,
              date default (datetime('now', 'localtime')),
              code TEXT,
              source TEXT
                )
              ''')
    c.execute(''' CREATE INDEX IF NOT EXISTS idx_original ON cells (RowID);
              ''')    
  

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
                Width INT
                )''')
    
    # Creating index for faster sorting and search later
    c.execute(''' 
              CREATE INDEX IF NOT EXISTS idx ON fields (Name);
              ''')
    
    c.connection.commit()

    
