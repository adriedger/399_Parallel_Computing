set terminal png
set xlabel "Number of Cores"
set ylabel "Time (normalized)"
set xrange[0:9]
set yrange[0:1.1]
set output 'times-norm.png'
set key left top
plot "times-norm.dat" using 1:2 title 'Algorithm A' with lines,\
     "times-norm.dat" using 1:3 title 'Algorithm B' with lines
