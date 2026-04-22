CC=g++
FLAGS = -I/usr/include/quanser -Wall -pedantic
LIBS   += -lhil -lquanser_runtime -lquanser_common -lrt -lpthread -ldl -lm -lc
MAIN = swingup_then_lqr.cpp
FILE = swingup_then_lqr

all : $(FILE)

debug : FLAGS += DDEBUG -g
debug: $(FILE)

clean: 
	rm -f $(FILE)
	rm -f *.o
	
 
 $(FILE): $(MAIN)
	$(CC) $(MAIN) $(FLAGS) $ -o $(FILE) $(LIBS)


