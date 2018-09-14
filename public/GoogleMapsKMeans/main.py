import sys
from k_means_for_lines_experiments_final import ExperimentsKMeansForLines
print(sys.argv[1])
file_name = './googlemapskmeans/GoogleMapsKMeans/lines.csv'
k = 2
try:
    means = ExperimentsKMeansForLines.get_means_for_lines_from_file(file_name, k)
    for point in means.points: 
        print(point)
except OSError as err:
    print("OS error: {0}".format(err))
except:
    print("Unexpected error:", sys.exc_info()[0])
