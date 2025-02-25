import sys

atl = {}
taco = {}

with open(sys.argv[1], "r") as file:
    # Read and print each line
	format_line = file.readline()

	while format_line:
		framework = format_line.strip().split()[0]
		format = format_line.strip().split()[1]

		mean_line = file.readline()
		stdev_line = file.readline()
		median_line = file.readline()
		format_line = file.readline()

		mean = mean_line.strip().split()[1]

		print format
		if (framework == "ATL"):
			atl[format] = mean
		else:
			taco[format] = mean

parts = sys.argv[1].split(".")

with open(parts[0]+".csv","w") as file:
	file.write("Format,ATL (ms),Taco (ms)\n")
	file.write("Dense,,"+taco["Dense"]+"\n")
	for format in atl:
		file.write(format+","+ atl[format]+","+taco[format]+"\n")
