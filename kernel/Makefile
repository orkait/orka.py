# ORKA_VQ reference kernel - build + test
CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -std=c99

.PHONY: all test lib clean

all: test lib

test: orka_vq.c test_orka_vq.c
	$(CC) $(CFLAGS) orka_vq.c test_orka_vq.c -lm -o test_orka_vq
	./test_orka_vq

lib: orka_vq.c
	$(CC) $(CFLAGS) -fPIC -shared -o liborka_vq.so orka_vq.c -lm

clean:
	rm -f test_orka_vq liborka_vq.so
