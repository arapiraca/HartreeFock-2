# http://mrbook.org/blog/tutorials/make/
# http://www.cs.colby.edu/maxwell/courses/tutorials/maketutor/

CC=g++
CFLAGS= -Wall -g #-Wextra -pedantic -std=c++11
CXXFLAGS= -fopenmp
LIBS= -llapack -lblas -lgfortran -lm
EXECUTABLES = HartreeFock5

O_FILES = $(patsubst %.cpp, %.o, $(wildcard *.cpp))

$(EXECUTABLES): $(O_FILES)
	$(CC) $(CXXFLAGS) $(CFLAGS) -o $@ $^ $(LIBS)

%.o: %.cpp  
	$(CC) $(CXXFLAGS) $(CFLAGS) -c -o $@ $< 

clean:
	rm *.o *~ *# 
