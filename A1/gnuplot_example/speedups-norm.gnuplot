set terminal png
set output 'speedups-norm.png'
set xlabel "Number of Cores"
set ylabel "Speedup"
set xrange [0:9]
set yrange [0:9]
set key left top
plot "speedups-norm.dat" using 1:2 title 'Simple' with lines,\
     "speedups-norm.dat" using 1:3 title 'Super' with lines,\
     "speedups-norm.dat" using 1:4 title 'Super v. Simple' with lines
