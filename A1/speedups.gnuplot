set terminal png
set xlabel "Number of Cores"
set ylabel "Speedup"
set output 'speedups.png'
set xrange [0:9]
set yrange [0:9]
set key left top
plot "speedups.dat" using 1:2 title 'Super' with lines,\
     "speedups.dat" using 1:3 title 'Simple' with lines
