# ORKA_VQ reference kernel - build + test
CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -std=c99

# Point at a llama.cpp checkout with a built libggml (build/bin) for ggml-op.
LLAMA ?= ../llama.cpp
GGML_INC := $(LLAMA)/ggml/include
GGML_LIB := $(LLAMA)/build/bin

.PHONY: all test lib ggml-op clean

all: test lib

test: orka_vq.c test_orka_vq.c
	$(CC) $(CFLAGS) orka_vq.c test_orka_vq.c -lm -o test_orka_vq
	./test_orka_vq

lib: orka_vq.c
	$(CC) $(CFLAGS) -fPIC -shared -o liborka_vq.so orka_vq.c -lm

# In-ggml proof: decode an Orka tensor inside a ggml custom op + matmul.
# Requires a built libggml under $(LLAMA)/build/bin.
ggml-op: ggml_orka_op.c orka_vq.c
	$(CC) $(CFLAGS) ggml_orka_op.c orka_vq.c -I$(GGML_INC) -I. \
		-L$(GGML_LIB) -lggml -lggml-base -lggml-cpu -lm \
		-Wl,-rpath,$(GGML_LIB) -o ggml_orka_op

clean:
	rm -f test_orka_vq test_orka_vq_ci liborka_vq.so ggml_orka_op
