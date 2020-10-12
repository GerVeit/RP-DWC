all: intro

intro: intro.cu
	nvcc -o intro intro.cu

clean:
	rm -rf *.o intro