TARGET := pi.out
CFILE := toss.c 
OCCP := ./SIMDxorshift/src/xorshift128plus.c ./SIMDxorshift/src/simdxorshift128plus.c

CC := g++

# minimalist makefile
.SUFFIXES:
#
.SUFFIXES: .cpp .o .c .h
ifeq ($(DEBUG),1)
CFLAGS = -fPIC  -std=c99 -ggdb -mavx2 -march=native -Wall -Wextra -pedantic -Wshadow -fsanitize=undefined  -fno-omit-frame-pointer -fsanitize=address -pthread
else
CFLAGS = -fPIC -std=c99 -O3 -mavx2  -march=native -Wall -Wextra -pedantic -Wshadow -pthread
endif # debug

HEADERS= SIMDxorshift/include/simdxorshift128plus.h  SIMDxorshift/include/xorshift128plus.h

OBJECTS= simdxorshift128plus.o xorshift128plus.o

$(TARGET) : $(CFILE) $(HEADERS)  $(OBJECTS)
	$(CC) $(OCCP) $(CFILE) -o $(TARGET) $(CFLAGS)  $(OBJECTS) -Iinclude -flto

xorshift128plus.o: ./SIMDxorshift/src/xorshift128plus.c $(HEADERS)
	$(CC) $(CFLAGS) -c ./SIMDxorshift/src/xorshift128plus.c -Iinclude

simdxorshift128plus.o: ./SIMDxorshift/src/simdxorshift128plus.c $(HEADERS)
	$(CC) $(CFLAGS) -c ./SIMDxorshift/src/simdxorshift128plus.c -Iinclude



clean:
	rm -f $(OBJECTS) *.out