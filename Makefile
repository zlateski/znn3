ODIR		=	./bin
CPP		=	g++-4.8
INC_FLAGS	=	-I. -I./src
OPT_FLAGS	=	-DARMA_USE_CXX11 -DARMA_USE_CXX11_RNG
OTH_FLAGS	=	-Wall -Wextra -std=c++11

LIBS		=	 -llapack -lblas -larmadillo -lfftw3 -lpthread -lrt

znn: src/main.cpp
	$(CPP) -o $(ODIR)/znn src/main.cpp $(INC_FLAGS) $(OPT_FLAGS) $(OTH_FLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*