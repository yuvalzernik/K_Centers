import sys, traceback
import csv
from set_of_lines import SetOfLines
import numpy as np
from k_means_for_lines_experiments_final import ExperimentsKMeansForLines
# print(sys.argv[1])
# print(sys.argv[2])
# print(sys.argv[3])
# print(sys.argv[4])
bounds = [sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]]
Algorithm = sys.argv[5]
file_name = './googlemapskmeans/GoogleMapsKMeans/lines.csv'
k = 5

def line_is_in_bound(line, bounds):
    south_west = [bounds[0],bounds[1]]
    north_east = [bounds[2],bounds[3]]
    return (bound_conditions(line[0], line[1], south_west, north_east) and (bound_conditions(line[2], line[3], south_west, north_east)))

def bound_conditions(X, Y, south_west, north_east):
    return (X >= float(south_west[0]) and X <= float(north_east[0]) and Y >= float(south_west[1]) and Y <= float(north_east[1]))


def create_file_with_lines_in_bounds(file_name, bounds):
    lines = []
    lines_nuber = 100000
    lines_in_bound = []
    with open('lines_for_yair.file','w') as f:
        f.write("Start\n")
    with open(file_name, 'rt') as csvfile:
        spamreader = csv.reader(csvfile)
        i = 0
        for row in spamreader:
            row_splited = [element.split(' ') for element in row]
            row_final = [float(entry) for entry in row_splited[0]]
            lines.append(row_final)
            if line_is_in_bound(row_final, bounds):
                lines_in_bound.append(row_final)
                with open('lines_for_yair.file','a') as f:
                    f.write(str(row_final)+ "\n")
            i += 1
            if i == lines_nuber:
                break
    
    # for lin in lines_in_bound:
    #     print(lin)
    L = SetOfLines(lines=lines_in_bound, is_points=True)
    # print("lines number total: ", L.get_size())
    # sys.stdout.flush()
    return L


try:
    lines = create_file_with_lines_in_bounds(file_name, bounds)
    ExperimentsKMeansForLines.bounds = bounds
    #get only RANSACcenters
    if Algorithm == "Ransac":
        ransac_means = ExperimentsKMeansForLines.get_ransac_means(lines)
        print(ransac_means.points)

    #get only coreset centers
    if Algorithm == "Coreset":
        coreset_means = ExperimentsKMeansForLines.get_coreset_means(lines)
        print(coreset_means.points)
        
    #get both coreset centers and RANSAC centers
    if Algorithm == "Both": 
        coreset_means, ransac_means = ExperimentsKMeansForLines.get_coreset_and_ransac_means(lines)
        print(coreset_means.points,'$', ransac_means.points)

        
    # #compare coreset to RANSAC and plot results
    # ExperimentsKMeansForLines.error_vs_coreset_size_streaming()
        # means = ExperimentsKMeansForLines.get_means_for_lines_from_file(lines, k)
    # print(means.points)

except OSError as err:
    print("OS error: {0}".format(err))
except:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback.print_tb(exc_traceback, limit=4, file=sys.stdout)
    print("Unexpected error:", sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2], traceback.print_exc())

