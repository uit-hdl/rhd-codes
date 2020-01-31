from FieldExtraction.CoordinateFileReader import CoordinateFileReader
from FieldExtraction.ProgressFileHandler import ProgressFileHandler
from FieldExtraction.ImageParser import ImageParser
from FieldExtraction.ParallelExecutor import ParallelExecutor
from Database.dbHandler import DbHandler


def run(args):
    print(args)
    print(args.image_range)
    with CoordinateFileReader(args.coordinate_file, args.img_path_mod, args.image_range) as cf_reader:
        with ProgressFileHandler(args.progress_file) as pf_handler:
            img_parser = ImageParser(args)
            
            # If the images should be uploaded to a database
            if args.output is None:
                with DbHandler(args.db) as db:
                    executor = ParallelExecutor(cf_reader, img_parser, pf_handler, args.workers, args.number, db, None)
                    executor.run()
                    
            # If the images should be stored in a local folder
            else:
                executor = ParallelExecutor(cf_reader, img_parser, pf_handler, args.workers, args.number, None, args.output)
                executor.run()
            
            
            
# =============================================================================
#             with DbHandler(args.db) as db:
#                 init_db.connect()
#                 img_parser = ImageParser(args)
#                 
#                 if args.output is None: 
#                     executor = ParallelExecutor(db, cf_reader, img_parser, pf_handler, args.workers, args.number)
#                 else:
#                     executor = ParallelExecutor(db, cf_reader, img_parser, pf_handler, args.workers, args.number, args.output)
#                 executor.run()
# =============================================================================
