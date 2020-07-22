CC = cc
CFLAGS =
UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
	LDFLAGS = -framework OpenCL
else
	LDFLAGS = -lOpenCL
endif
SOURCES = dijkstra.c
OBJECTS = $(notdir $(SOURCES:.c=.o))
EXECUTE = dijkstra

all: $(OBJECTS) $(EXECUTE)

$(EXECUTE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@
$(OBJECTS): $(SOURCES)
	$(CC) $(CFLAGS) $(SOURCES) -c

run:
	../gen 10000 | ./$(EXECUTE)
clean:
	rm -rf *~ *.o $(EXECUTE)
