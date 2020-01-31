class ProgressFileHandler:
    def __init__(self, filename):
        self.filename = filename
        self.file = None
        self.completed_images = {}

    def get_completed_images(self):
        return self.completed_images

    def update_file(self, line):
        self.file.write(line)

    def __enter__(self):
        self.file = open(self.filename, 'a+')
        self.file.seek(0)
        for line in self.file:
            line = line.rstrip()
            self.completed_images[line] = True

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
