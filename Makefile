all: intro

intro: intro.cpp
	g++ -o intro intro.cpp

clean:
	rm -rf *.o intro