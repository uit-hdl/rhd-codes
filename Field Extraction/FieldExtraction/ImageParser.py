import cv2
import numpy as np
import os

class ImageParser:
    def __init__(self, args):
        self.col_names = {"husholdnings_nr": 0,
                          "person_nr": 1,
                          "navn": 2,
                          "stilling_i_hus": 3,
                          "kjonn": 4,
                          "fast_bosatt_og_tilstedet": 5,
                          "midlertidig_fraværende": 6,
                          "midlertidig_tilstedet": 7,
                          "fødselsdato": 8,
                          "fødested": 9,
                          "ekteskapelig_stilling": 10,
                          "ekteskaps_aar": 11,
                          "barnetall": 12,
                          "arbeid": 13,
                          "egen_virksomhet": 14,
                          "bedrift_arbeidsgiver": 15,
                          "arbeidssted": 16,
                          "biyrke": 17,
                          "hjelper_hovedperson": 18,
                          "utdanning_artium": 19,
                          "høyere_utdanning": 20,
                          "trossamfunn": 21,
                          "borgerrett": 22,
                          "innflytning": 23,
                          "sist_kommune": 24,
                          "bosatt_i_1946": 25
                          }
        self.cols_number = args.cols_number
        self.cols_name = args.cols_name
        self.type = args.type
        self.process_images = args.process_images
        self.color = args.color
        self.target_fields = []
        if len(self.cols_number) != 0:
            self.target_fields = self.cols_number
        elif len(self.cols_name) != 0:
            self.target_fields = self.cols_name
                        
            
    def process_rows(self, filename, rows):
        extracted_rows = self._extract_rows(rows)
        
        image_fields = []
        img = cv2.imread(filename)
        
        if img is None or len(img) == 0:
            print("Image not found: " + filename + " , check path prefix or remote connections")
            self.image_error(filename)
            return []
        for i in range(0, len(extracted_rows) - 1, 2):
            row_1 = extracted_rows[i][0]
            row_2 = extracted_rows[i + 1][0]
            fields = self._split_row(img, row_1, row_2)
            i += 1
            image_fields.append((fields, extracted_rows[i][1]))
        return image_fields, filename

    # If the image could not be read
    @staticmethod
    def image_error(filename):
        with open('Errorlist.txt', 'a+') as file:
            file.write(filename + '\n')

    # Save image to local folder
    @staticmethod
    def write_field_image(fn, rows, output_dir):
        fn_path = os.path.join(output_dir, fn.split(".")[0])
        if not os.path.exists(fn_path):
            os.mkdir(fn_path)
        for row in rows:
            for field in row[0]:
                field_name = os.path.join(fn_path, str(row[1]) + "_" + str(field[1]) + fn)
                cv2.imwrite(field_name, field[0])

    # Save image to database
    @staticmethod
    def upload_field_image(fn, rows, db):
        for row in rows:
            # This if statement was added, if this no longer works, remove the statement
            if row[0] is not None:

                for field in row[0]:
                    field_name = str(row[1]) + "_" + str(field[1]) + fn
                    #db.store_field(field_name, field[0])
                    db.store_field_updated(field_name, field[0])

    @staticmethod
    def _convert_img_bw(img):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # lower mask
        lower_red = np.array([0, 45, 50])
        upper_red = np.array([20, 255, 255])
        mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

        # upper mask
        lower_red = np.array([160, 50, 50])
        upper_red = np.array([190, 255, 255])
        mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

        mask = mask0 + mask1

        output_hsv = img_hsv.copy()
        output_hsv[np.where(mask == 0)] = 0
        b_w = cv2.split(output_hsv)[2]
        retval, b_w = cv2.threshold(b_w, 100, 255, cv2.THRESH_BINARY)
        b_w = cv2.GaussianBlur(b_w, (3, 3), 0)
        b_w = cv2.bitwise_not(b_w)

        return b_w
    
    @staticmethod
    def _convert_img_gray(img):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # lower mask
        lower_red = np.array([0, 45, 50])
        upper_red = np.array([20, 255, 255])
        mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
        
        # upper mask
        lower_red = np.array([160, 50, 50])
        upper_red = np.array([190, 255, 255])
        mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
        
        mask = mask0 + mask1
        
        # Check for any red pixels (Means there exists writing in the image)        
        red_pix = np.count_nonzero(mask)
        
        # If the image contains red pixels, return a tuple of True and the converted image
        if red_pix > 300:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return (True, gray)
        # If the image contains no red pixels, return a tuple of False and the oiriginal image
        else:
            return (False, img)
    
    # Function to remove images that have no writing in them
    @staticmethod
    def _remove_empty_img(img):
        n_white_pix = np.sum(img == 255)
        height = img.shape[0]
        width = img.shape[1]
        allpixels = height * width
        
        pwhitepixels = ((n_white_pix*100)/allpixels)
        
        if pwhitepixels < 99:
            return False
        else: 
            return True
        
    @staticmethod
    def _adjust_edges(self, img, x1, x2, y1, y2):
      
        check_dist = 8                                                                              # The number of pixel rows that are checked before the opposite border is moved. If the move would cut off the digit, the border will not move
        move_dist = 5                                                                               # The distance that a border should be moved each time
        done = False                                                                                # Variable for when no more borders needs to be expanded
            
        original = img                                                                              # The original image, this will be cut and returned as conversion to both grayscale and black/white for upload happens later
        backup = img[y1:y2, x1:x2]                                                                  # In case we run into a crossed out section, or another problematic field
        orig_values = [x1, x2, y1, y2]
        
        
        border_counter = {'left': 0, 'right': 0, 'top': 0, 'bottom': 0}                                # A dictionary to keep track of how many times each border has been moved
        ignored_borders = []                                                                        # If they have been moved too many times, the image is reset and that border will be ognored
        
        # Keep moving borders until the whole digit(s) are in the image, or until they have tried to expand in all directions too many times
        while done == False:
        
            field_img = img[y1:y2, x1:x2]                                                           # Use the (adjusted) coordinates to cut out a field from the image
            field_img = self._convert_img_bw(field_img)                                             # Converting the image to black/white makes it easier to spot when a number has been cut off
                                                                                   
            # Get the "border pixels" for each side of the image
            left = ('left', field_img[:, 0])
            right = ('right', field_img[:, field_img.shape[1]-1])
            
            top = ('top', field_img[0, :])
            bottom = ('bottom', field_img[field_img.shape[0]-1, :])
            
            borders = [left, right, top, bottom]
            
            done = True                                                                             # Continue as long as borders are getting expanded
            for border in borders:                                                                  #Check the borders for any cells with a sum of 0 -> indicates that the number got cut off
                if str(border[0]) in ignored_borders: 
                    continue
            
                for elem in border[1]:
                    if elem == 0:
                        done = False
                        
                        if border[0] == 'left':                                                     # If there is at least one cell with a sum of 0. The left border needs to be shifted.
                            check = field_img[:, field_img.shape[1]-check_dist:field_img.shape[1]]  # Checking the opposite (Right) border to see if we can shift it as well, to keep the image size ratio
                            if 0 in check:                                                          # If any element in the next 5 pixels of the right border is a 0, we cannot change that border anymore without cutting off parts
                                x1 -= move_dist                                                     #Instead we alter the Left border and break
                                border_counter['left'] += 1                                         # We note that we have moved this border 
                                break
                            else:
                                x1 -= move_dist                                                      # If none of the elements in Right's next row is a 0, then we can alter Both borders
                                x2 -= move_dist
                                border_counter['left'] += 1
                                break                                                               # After it's been determined that the border needs to be moved, Continue to the next border
                        
                        elif border[0] == 'right':                                  
                            check = field_img[:, 0:check_dist]
                            if 0 in check:
                                x2 += move_dist
                                border_counter['right'] += 1
                                break
                            else:
                                x2 += move_dist
                                x1 += move_dist
                                border_counter['right'] += 1
                                break
                        
                        elif border[0] == 'top':
                            check = field_img[field_img.shape[0]-check_dist:field_img.shape[0]]
                            if 0 in check:
                                y1 -= move_dist
                                border_counter['top'] += 1
                                break
                            else:
                                y2 -= move_dist
                                y1 -= move_dist
                                border_counter['top'] += 1
                                break
                        
                        elif border[0] == 'bottom': 
                            check = field_img[0:check_dist]           
                            if 0 in check:                                  
                                y2 += move_dist  
                                border_counter['bottom'] += 1
                                break
                            else:
                                y2 += move_dist                                     
                                y1 += move_dist
                                border_counter['bottom'] += 1
                                break
            
            # Check how many times each border has been moved
            for key, value in border_counter.items():
                if value >= 10:                                                                     # No border should need to move 10+ times for a non-faulty or compromised cell
                    ignored_borders.append(key)                                                     # If it has, we will start again but this time the border that was faulty will be ignored
                    
                    border_counter = {key: 0 for key in border_counter}                             # We reset the dictionary of border movements, as we now will 
                    
                    x1 = orig_values[0]                                                             # And reset the image back to the original
                    x2 = orig_values[1]
                    y1 = orig_values[2]
                    y2 = orig_values[3]
                    
                
        # Return the coordinates for the new borders. If no change was done, or every border tried to expand too much, the coordinates will remain unchanged
        if len(ignored_borders) == 4:
            return backup
        else:
            return_image = original[y1:y2, x1:x2]
            return return_image
                        

    @staticmethod
    def _extract_field(self, img, row_1, row_2, i):
        # x position different index on same row
        """
        x1-----------x2
        x1-----------x2
        """        
        x1 = row_1[i][0]
        x2 = row_1[i + 2][0]
        # y position same index on different row
        """
        y1----------y1
        y2----------y2
        """
        y1 = row_1[i][1]
        y2 = row_2[i][1]

        field_img = self._adjust_edges(self, img, x1, x2, y1, y2)
        
# =============================================================================
#         cv2.imshow('image', field_img)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
# =============================================================================
        
        return field_img

    def _check_extraction(self, img, row_1, row_2, i):
        field_img = []
        if len(self.target_fields) > 0:
            # check if the current field is a field that is wanted
            if row_1[i - 1] in self.target_fields:
                field_img = self._extract_field(self, img, row_1, row_2, i)
            # Same as above, only the field is defined as a string
            elif isinstance(self.target_fields[0], str):
                for name in self.target_fields:
                    if self.col_names[name] == row_1[i - 1]:
                        field_img = self._extract_field(img, row_1, row_2, i)

        else:
            field_img = self._extract_field(img, row_1, row_2, i)

        return field_img

    def _split_row(self, img, row_1, row_2):
       
        try:
            fields = []

            for i in range(1, len(row_1) - 2, 2):
                field_img = self._check_extraction(img, row_1, row_2, i)        # Image is still in original form here
                if len(field_img) != 0:
                    
                    # Check if image will be processed 
                    if self.process_images:
                        
                        # If the image will be converted to black and white
                        if self.color == 'bw':
                            field_img = self._convert_img_bw(field_img)     # Do the conversion to black and white
                            blank = self._remove_empty_img(field_img)       # Check if the image is blank (All white)
                            
                            # If the image contains at least Some black pixels, add it to the list
                            if not blank:
                                fields.append((field_img, i))
                            
                            # If the image only contains white pixels, move on to the next image
                            else:
                                continue
                            
                        # If the image will be converted to grayscale
                        if self.color == 'gray':
                            return_tuple = self._convert_img_gray(field_img)
                            
                            # If the boolean value in the return_tuple is True, then the image contained red pixels and were converted
                            if return_tuple[0] == True:
                                fields.append((return_tuple[1], i))
                            
                            # If the boolean value in the return_tuple is False, then the image was a "blank" and we skip to the next image
                            else:
                                continue
                    
                    # If no processing is to be done, add the raw image
                    else:
                        temp_img = self._convert_img_bw(field_img) # Convert a temp copy of the image to B&W to check if it is blank or not
                        blank = self._remove_empty_img(temp_img)

                        if not blank:
                            fields.append((field_img, i))
                        else:
                            continue
                    
            return fields
        except Exception as e:
            print(e)

    def _extract_rows(self, rows):
        index = 0
        step = 1
        if self.type == "digits":
            step = 2
            index = 1
        elif self.type == "writing":
            step = 2
        extracted_row = []
        for k in range(index, len(rows) - 1, step):
            row_1, row_1_index = self._split_row_str(rows[k])
            row_2, row_2_index = self._split_row_str(rows[k + 1])
            extracted_row.append((row_1, row_1_index))
            extracted_row.append((row_2, row_2_index))

        return extracted_row

    @staticmethod
    def _split_row_str(line):
        _row = line.split('<')

        row = []
        # Start the coordinate extraction after the column index
        for token in _row[2:]:
            coordinate = token.split(',')
            if len(coordinate) == 2:
                row.append((int(coordinate[0]), int(coordinate[1])))
            else:
                row.append(int(token))

        # Returns the row and the row index
        return row, int(_row[1])
