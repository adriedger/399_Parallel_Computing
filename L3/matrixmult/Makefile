CC=gcc

all: mm seq colmajor colmajor-seq

seq: mm.c
	$(CC) -std=c99 -D_BSD_SOURCE -DSEQ -O3 $^ -o $@ -lpthread -lm

mm: mm.c
	$(CC) -std=c99 -D_BSD_SOURCE -O3 $^ -o $@ -lpthread -lm

blocked: blocked.c
	$(CC) -std=c99 -D_BSD_SOURCE -O3 $^ -o $@ -lpthread -lm

colmajor-seq: colmajor.c
	$(CC) -std=c99 -D_BSD_SOURCE -DSEQ -O3 $^ -o $@ -lpthread -lm -msse3

colmajor: colmajor.c
	$(CC) -std=c99 -D_BSD_SOURCE -O3 $^ -o $@ -lpthread -lm -msse3

clean:
	rm -f seq mm colmajor colmajor-seq
