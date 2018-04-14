set terminal png
set output 'linegraph.png'
set xrange[0:10]
set xlabel "Number of Cores"
set ylabel "Time (secs)"
set yrange[0:]
set key left top
plot "gnuplot.dat" using 1:2 title 'Simple' with lines,\
     "gnuplot.dat" using 1:3 title 'Locking' with lines,\
     "gnuplot.dat" using 1:4 title 'Super' with lines
