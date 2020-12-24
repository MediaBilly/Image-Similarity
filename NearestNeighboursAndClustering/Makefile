CC = g++
FLAGS = -Wall -O3 -std=c++11
TARGETS = search cluster emd
SRC_DIR = ./src
OBJS = *.o

all:$(TARGETS)

search:search_main.o lsh.o dataset.o image.o hash_table.o hash_function.o utilities.o bruteforce_search.o
	$(CC) $(FLAGS) -o search lsh.o search_main.o dataset.o image.o hash_table.o hash_function.o utilities.o bruteforce_search.o

cluster:cluster_main.o dataset.o image.o bruteforce_search.o utilities.o cluster.o
	$(CC) $(FLAGS) -o cluster cluster_main.o dataset.o image.o bruteforce_search.o utilities.o cluster.o

emd:emd.o dataset.o image.o utilities.o bruteforce_search.o
	$(CC) $(FLAGS) -o emd emd.o dataset.o image.o utilities.o bruteforce_search.o

%.o:$(SRC_DIR)/%.cpp
	$(CC) $(FLAGS) -c -o $@ $<

.PHONY : clean

clean:
	rm -f $(TARGETS) $(OBJS)
