import argparse
from FieldExtraction.FieldSplitter import run


def main():
    arg_parser = argparse.ArgumentParser(
        description="Extract fields from the given image using pre-calculated coordinates")
    arg_parser.add_argument( "coordinate_file", type=str, help="path to the coordinate file")
    arg_parser.add_argument("--img_path_mod", "-m", metavar="M", type=str, default="",
                            help="replaces the share= in the coordinate file and replaces it with the given argument")
    arg_parser.add_argument("--type", "-t", metavar="T", type=str,
                            help="specifies the type of fields to be extracted", choices=["all", "digits", "writing"],
                            default="all")
    arg_parser.add_argument("--color", "-l", metavar="L", type=str,
                            help="specifies the kind of coloring preprocessing to utilize", choices=['bw', 'gray'],
                            default="bw")
    arg_parser.add_argument("--process_images", "-P", action="store_true", help="convert and filter images")
    arg_parser.add_argument("--cols_number", "-c", nargs='+',
                            help="specifies the column indexes to search for. "
                                 "Defined in a space separated list 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26",
                            default=[])
    arg_parser.add_argument("--cols_name", "-C", nargs='+',
                            help="specifies the column by name, defined as a list of names separated by spaces"
                                 "husholdnings_nr  "
                                 "person_nr  "
                                 "navn "
                                 "stilling_i_hus " 
                                 "kjønn " 
                                 "fast_bosatt_og_tilstedet "
                                 "midlertidig_fraværende "
                                 "midlertidig_tilstedet "
                                 "fødselsdato "
                                 "fødested "
                                 "ekteskapelig_stilling "
                                 "ekteskaps_aar "
                                 "barnetall "
                                 "arbeid "
                                 "egen_virksomhet "
                                 "bedrift_arbeidsgiver "
                                 "arbeidssted "
                                 "biyrke "
                                 "hjelper_hovedperson "
                                 "utdanning_artium "
                                 "høyere_utdanning "
                                 "trossamfunn "
                                 "borgerrett "
                                 "innflytning "
                                 "sist_kommune "
                                 "bosatt_i_1946",
                            default=[])
    # Currently does not appear to function
    
    arg_parser.add_argument("--number", "-n", type=int,
                            help="specify the number of images to process starting from the top of the coordinate "
                                 "files or from continuation point specified in the checkpoint file",
                            default=0)
    
    arg_parser.add_argument("--output", "-o", type=str, help="output path")
    arg_parser.add_argument("--workers", "-w", type=int, help="max number of parallel worker processes", default=4)
    arg_parser.add_argument("--progress_file", type=str,
                            help="location of progress file, will be created if it does not exist",
                            default="progress.txt")
    arg_parser.add_argument("--image_range", "-r", nargs='+', type=str,
                            help="Specify a range fsxxxx fsyyyy to process all images in the given range")
    arg_parser.add_argument("--db", type=str, help="full path to database location")
    args = arg_parser.parse_args()
    _tmp = []
    for num in args.cols_number:
        _tmp.append(int(num))
    args.cols_number = _tmp
    run(args)


if __name__ == '__main__':
    main()
