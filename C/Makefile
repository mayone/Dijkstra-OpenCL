CC = gcc
CFLAGS =
LDFLAGS =
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
