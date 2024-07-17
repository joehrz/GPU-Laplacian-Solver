# Makefile for SOR PDE Solver

CXX = g++
CXXFLAGS = -std=c++11 -Wall
LDFLAGS = -lm

# Name of the executable
TARGET = SOR_Solver

# Source files
SOURCES = sor_serial.cpp
OBJECTS = $(SOURCES:.cpp=.o)

# Build the program
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Compile the source files into object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

# Clean up
clean:
	rm -f $(TARGET) $(OBJECTS)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Declare non-file targets
.PHONY: clean run