# Compiler
CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++17 -O2

# Directories
INCDIR = include
SRCDIR = src
BUILDDIR = build
TARGETDIR = bin

# Files
TARGET = $(TARGETDIR)/PDE-GPU-Solver
SOURCES = $(SRCDIR)/main.cpp $(SRCDIR)/sor_methods.cpp
OBJECTS = $(BUILDDIR)/main.o $(BUILDDIR)/sor_methods.o

# Rules
all: $(TARGET)

$(TARGET): $(OBJECTS)
	@mkdir -p $(TARGETDIR)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $@ $(LIBS)
	@echo "Build complete. Executable: $(TARGET)"

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -I$(INCDIR) -c $< -o $@

clean:
	rm -rf $(BUILDDIR) $(TARGETDIR)
	@echo "Cleaned up build directories."