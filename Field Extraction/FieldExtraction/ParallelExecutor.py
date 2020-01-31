import os
import time
import concurrent.futures as cf


class ParallelExecutor:
    def __init__(self, cf_reader, img_parser, pf_handler, workers, num_to_do, db = None, output_dir = None):
        self.db = db
        self.cf_reader = cf_reader
        self.img_parser = img_parser
        self.pf_handler = pf_handler
        self.workers = workers
        self.num_to_do = num_to_do
        self.output_dir = output_dir
        self.num_reads = 0
        self.num_skipped = 0
        self.num_completed = 0
        self.completed_images = self.pf_handler.get_completed_images()
        self.futures = []
        self.img_list = self.cf_reader.create_img_list()

        if output_dir is not None:
            if not os.path.exists(self.output_dir):
                print("Creating new directory for output: " + self.output_dir)
                os.mkdir(self.output_dir)

    def run(self):
        start_time = time.time()

        with cf.ProcessPoolExecutor(max_workers=self.workers) as executor:
            while self.cf_reader.continue_reading():
                rows, img_filename = self.cf_reader.read_full_image_lines()
                if self.skip_image(img_filename):
                    continue
                if self.img_list is not None:
                    if not self.submit_listed_img(executor, img_filename, rows):
                        break
                else:
                    if not self.submit_and_continue(executor, img_filename, rows):
                        break

            self.handle_all_submitted(start_time)

    def submit_listed_img(self, executor, img_filename, rows):
        if img_filename.split(os.path.sep)[-1].split(".")[0] in self.img_list:
            if not self.submit_and_continue(executor, img_filename, rows):
                return False
        else:
            self.num_skipped += 1

        return True

    def handle_done(self, done, futures, num_reads):      
        if done.result() != [] or done.result()[0] is not None:

            rows, fn = done.result()
        
            fn = fn.split(os.path.sep)[-1]
            self.pf_handler.update_file(fn + '\n')
    
            if self.output_dir is None:
                self.img_parser.upload_field_image(fn, rows, self.db)
            else:
                self.img_parser.write_field_image(fn, rows, self.output_dir)
    
            futures.remove(done)
            self.num_completed += 1
            if self.num_completed % 10 == 0:
                
                if self.db is not None:
                    self.db.connection.commit()
                print(str(self.num_completed / num_reads * 100) + "%")
                
        else:
            futures.remove(done)
            self.handle_done_error(done)
                    
    def handle_done_error(self, done):
        with open('Done_errors.txt', 'a+') as file:
            if done.result() == []:
                file.write('Empty list\n')
            elif done.result()[0] is None:
                file.write('Rows were None for ' + str(done.result()[1]) + '\n')


    def skip_image(self, img_filename):
        if img_filename == "":
            print("Something went wrong with reading the image filename from the coordinate file")
            return True
        if self.completed_images.get(img_filename.split(os.path.sep)[-1]):
            print("Skipping " + img_filename)
            self.num_skipped += 1
            return True

        return False

    def submit_and_continue(self, executor, img_filename, rows):
        
# =============================================================================
#         if len(rows) > 21 or self.num_reads == 26 or self.num_reads == 63:
#             stop = True
# =============================================================================
        
        print('Reading image: {} - {}'.format(self.num_reads, img_filename))
        #test = self.img_parser.process_rows(img_filename, rows)
        self.futures.append(executor.submit(self.img_parser.process_rows, img_filename, rows))
        self.num_reads += 1

        # If self.num_reads is high enough, call handle_done with the images that have been completed up to now
        if self.num_reads % 1000 == 0:
            for done in cf.as_completed(self.futures):
              self.handle_done(done, self.futures, self.num_reads)

        # This if check is how the code is meant to handle being finished reading all the images. 
        # Revert back to this if the above did not work
        if self.num_reads == self.num_to_do:
            return False
        return True

            
    
    def handle_all_submitted(self, start_time):
        print("Skipped a total of " + str(self.num_skipped) + " images")
        for done in cf.as_completed(self.futures):
            self.handle_done(done, self.futures, self.num_reads)
        print("Completed a total of " + str(self.num_completed) + " images")
        print("--- " + str(time.time() - start_time) + " ---")
        
