TARGET = matmul
CXX = mpicxx
CCFILE := standard.cc

$(TARGET) : $(CCFILE)
		$(CXX) $(CCFILE) -O3 -o ${TARGET}

clean:
	rm -f $(TARGET) 