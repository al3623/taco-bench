CC := clang
CFLAGS := -g -Wall -std=c99
LDFLAGS := -g

GENFILE := gen.out
STEMS 	:= $(wildcard *.c)
OBJECTS := $(patsubst %.c,%.o, $(STEMS))

libsparse.a: $(OBJECTS)
	ar rc $@ $^
	ranlib $@

%.o: %.c
	$(CC) $(CFLAGS) -ffast-math -O3 -c $^ -o $@

.PHONY: clean scrub

scrub:
	./cleanoutput.sh $(GENFILE)

clean:
	rm -f *.o a.out core main *.a *.BAK
