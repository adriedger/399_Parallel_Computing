set terminal png
set xlabel "Number of Cores"
set ylabel "Time (secs)"
set xrange[0:9]
set output 'times.png'
set key right top
plot "times.dat" using 1:2 title 'Algorithm A' with lines
