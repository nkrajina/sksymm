SHELL = /bin/bash
CC = icc
CFLAGS = -xHost -Wall -std=c11 -D_XOPEN_SOURCE=600 -O3 -mkl -prec-div -prec-sqrt -no-ftz -fma -fopenmp -restrict -qopt-report=5
.DEFAULT_GOAL = ""

pdk_sng_seq: main.o pdk_sng_seq.o pdkort.o
	$(CC) $(CFLAGS) $^ -o $@

pdk_sng_omp: main.o pdk_sng_omp.o pdkort.o
	$(CC) $(CFLAGS) $^ -o $@

pdk_vec_seq: main.o pdk_vec_seq.o pdkort.o
	$(CC) $(CFLAGS) $^ -o $@

pdk_vec_omp: main.o pdk_vec_omp.o pdkort.o
	$(CC) $(CFLAGS) $^ -o $@

main.o: pdk.h defs.h
pdkort.o: pdkort.h defs.h
pdk_sng_seq.o: pdk.h defs.h pdkort.o
pdk_sng_omp.o: pdk.h defs.h pdkort.o

.PHONY: clean
clean:
	-rm -f *.o

