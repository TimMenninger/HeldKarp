import matplotlib.pyplot as plt
import sys


f = open(sys.argv[2])
path = []

for line in f.readlines():
	if line == "":
		continue
	line = line.strip("\n")
	linelist = line.split(" ")
	linetuple = (int(linelist[0]), int(linelist[1]))
	path.append(linetuple)
f.close()

f = open(sys.argv[1])
data = []

for line in f.readlines():
	if line == "":
		continue
	if len(data) == len(path):
		continue
	line = line.strip("\n")
	linelist = line.split(" ")
	linetuple = (float(linelist[1]), float(linelist[2]))
	data.append(linetuple)
f.close()

plt.scatter(*zip(*data))

for current in path:
	x_vals = [data[current[0]][0], data[current[1]][0]]
	y_vals = [data[current[0]][1], data[current[1]][1]]
	plt.plot(x_vals, y_vals, 'k-')


plt.show()
