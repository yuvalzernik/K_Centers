import sys
import csv
from k_means_for_lines_experiments_final import ExperimentsKMeansForLines
# print(sys.argv[1])
# print(sys.argv[2])
# print(sys.argv[3])
# print(sys.argv[4])
bounds = [sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]]
file_name = './googlemapskmeans/GoogleMapsKMeans/lines.csv'
k = 2

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
    with open(file_name, 'rt') as csvfile:
        spamreader = csv.reader(csvfile)
        i = 0
        for row in spamreader:
            row_splited = [element.split(' ') for element in row]
            row_final = [float(entry) for entry in row_splited[0]]
            lines.append(row_final)
            if line_is_in_bound(row_final, bounds):
                lines_in_bound.append(row_final)
            i += 1
            if i == lines_nuber:
                break
    # for lin in lines_in_bound:
    #     # print(lin)
    return lines_in_bound


try:
    lines = create_file_with_lines_in_bounds(file_name, bounds)
    means = ExperimentsKMeansForLines.get_means_for_lines_from_file(lines, k)
    print(means.points)

except OSError as err:
    print("OS error: {0}".format(err))
except:
    print("Unexpected error:", sys.exc_info()[0], sys.exc_info()[1])





