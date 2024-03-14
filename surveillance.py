from detect import main
from generate_roi import preprocess
import json

roi_coord = preprocess()
#roi_coord = json.load(open("roi_coordinates.json", "r"))
main(roi_coord)
