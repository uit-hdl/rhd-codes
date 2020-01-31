import os


class CoordinateFileReader:
    def __init__(self, filename, mod, img_range):
        self.filename = filename
        self.mod = mod
        self.img_range = img_range
        if self.img_range:
            assert len(self.img_range) % 2 == 0
        self.file = None
        self.first_line = None

    def read_full_image_lines(self):
        first_line = self.first_line

        rows = []
        while True:

            second_line = self.file.readline()
            filename = self._get_img_filename(first_line)
            filename2 = self._get_img_filename(second_line)
            rows.append(first_line)
            if filename != filename2:
                self.first_line = second_line
                break

            # rows.append(second_line)
            first_line = second_line

        return rows, filename

    def create_img_list(self):

        if not self.img_range:
            return None

        proper_img_list = []
        count = 0
        # img_list_start = ["fs10061402170436"]
        # img_list_end =   ["fs10061402177225"]
        for i in range(0, len(self.img_range), 2):
            start_num = int(self.img_range[i].split('fs')[-1])
            end_num = int(self.img_range[i + 1].split('fs')[-1])
            for index in range(start_num, end_num):
                proper_img_list.append("fs" + str(index))
                count += 1

        return proper_img_list

    def continue_reading(self):
        return self.file.tell() != os.fstat(self.file.fileno()).st_size

    def _get_img_filename(self, line):
        img_filename = ""
        if line == "":
            return img_filename
        try:
            # Filename is the first segment after the first comma
            img_filename = line.split('<')[0].split(',')[1]
            if self.mod != "":
                img_filename = img_filename.split('=')[1]
                img_filename = self.mod + img_filename + ".jpg"
                img_filename = img_filename.replace("/", os.path.sep)
        except Exception as e:
            print(e)

        return img_filename

    def __enter__(self):
        self.file = open(self.filename, 'r')
        # Skip the header line
        self.file.readline()
        # Read first line to initialize the reader
        self.first_line = self.file.readline()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
