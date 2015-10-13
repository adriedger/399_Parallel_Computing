set terminal png
set xlabel "Number of Cores"
set ylabel "Time (secs)"
set xrange[0:8]
set output 'times.png'
set key right top
plot "times.dat" using 1:2 title 'Super' with lines,\
     "times.dat" using 1:3 title 'Simple' with lines
