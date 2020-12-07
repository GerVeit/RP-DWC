#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#include <vector>
#include <iostream>

#ifdef RD_WG_SIZE_0_0                                                            
#define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)                                                      
#define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)                                                        
#define BLOCK_SIZE RD_WG_SIZE
#else                                                                                    
#define BLOCK_SIZE 16
#endif                                                                                   

#define STR_SIZE 256

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD	(3.0e6)
/* required precision in degrees	*/
#define PRECISION	0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5

/* chip parameters	*/
float t_chip = 0.0005;
float chip_height = 0.016;
float chip_width = 0.016;
/* ambient temperature, assuming no package at all	*/
float amb_temp = 80.0;

void run(int argc, char** argv);

/* define timer macros */
#define pin_stats_reset()   startCycle()
#define pin_stats_pause(cycles)   stopCycle(cycles)
#define pin_stats_dump(cycles)    printf("timer: %Lu\n", cycles)

void fatal(char *s) {
	fprintf(stderr, "error: %s\n", s);

}

void _checkFrameworkErrors(cudaError_t error, int line, const char* file) {
	if (error == cudaSuccess) {
		return;
	}
	char errorDescription[250];
	snprintf(errorDescription, 250, "CUDA Framework error: %s. Bailing.",
			cudaGetErrorString(error));
#ifdef LOGS
	log_error_detail((char *)errorDescription);
	end_log_file();
#endif
	printf("%s - Line: %d at %s\n", errorDescription, line, file);
	exit (EXIT_FAILURE);
}

#define checkFrameworkErrors(error) _checkFrameworkErrors(error, __LINE__, __FILE__)

void writeoutput(float *vect, int grid_rows, int grid_cols, char *file) {

	int i, j, index = 0;
	FILE *fp;
	char str[STR_SIZE];

	if ((fp = fopen(file, "w")) == 0)
		printf("The file was not opened\n");

	for (i = 0; i < grid_rows; i++)
		for (j = 0; j < grid_cols; j++) {

			sprintf(str, "%d\t%g\n", index, vect[i * grid_cols + j]);
			fputs(str, fp);
			index++;
		}

	fclose(fp);
}

template<typename double_t, typename single_t>
void compareOutputHost(std::vector<double_t> &vectDouble,
		std::vector<single_t> &vectSingle) {
	single_t max_relative = -99999;
	single_t min_relative = 99999;

	for (int i = 0; i < vectDouble.size(); i++) {
		auto dt = vectDouble[i];
		auto st = vectSingle[i];
		auto diff = (st - single_t(dt)) / st;
		max_relative = std::max(max_relative, diff);
		min_relative = std::min(min_relative, diff);
	}

	std::cout << "Max relative error on host " << max_relative << std::endl;
	std::cout << "Min relative error on host " << min_relative << std::endl;
}

template<typename double_t, typename single_t> __global__
void compareOutputGPU(double_t *vectDouble, single_t *vectSingle, single_t *relative) {
	auto index = blockIdx.x * blockDim.x + threadIdx.x;
	single_t dt = single_t(vectDouble[index]);
	single_t st = vectSingle[index];

	relative[index] = dt / st;
}

template<typename single_t>
void checkRelative(std::vector<single_t> &vectSingle) {
	float max_relative = -99999;
	float min_relative = 99999;

	// for (auto diff : vectSingle) {
	// 	max_relative = std::max(max_relative, diff);
	// 	min_relative = std::min(min_relative, diff);
	// }
	printf("relative vector size %lu\n", vectSingle.size());
	for(int diff = 0; diff < vectSingle.size(); diff++) {
		printf("Relative Error: %f\n", (float)vectSingle[diff]);
		if(vectSingle[diff] > max_relative)
			max_relative = (float)vectSingle[diff];
		if(vectSingle[diff] < min_relative)
			min_relative = (float)vectSingle[diff];
	}

	printf("Max relative error on host: %f\n", max_relative);
	printf("Min relative error on host: %f\n", min_relative);
}

void readinput(double *vectDouble, float *vect, int grid_rows, int grid_cols, char *file_float, char *file_double) {

	FILE *fp, *dp;
	char str[STR_SIZE], str_b[STR_SIZE];
	float val;
	double val_b;

	if ((fp = fopen(file_float, "r")) == 0)
		printf("The file was not opened\n");
	if ((dp = fopen(file_double, "r")) == 0)
		printf("The file was not opened\n");

	for (int i = 0; i <= grid_rows - 1; i++)
		for (int j = 0; j <= grid_cols - 1; j++) {
			fgets(str, STR_SIZE, fp);
			if (feof(fp))
				fatal("not enough lines in file");
			
			if ((sscanf(str, "%f", &val) != 1))
                fatal("invalid file format");
                
			vect[i * grid_cols + j] = val;
			vectDouble[i * grid_cols + j] = val;
		}
	fclose(fp);

	for (int l = 0; l <= grid_rows - 1; l++)
		for (int k = 0; k <= grid_cols - 1; k++) {
			fgets(str_b, STR_SIZE, dp);
			if (feof(dp))
				fatal("not enough lines in file");
			
			if ((sscanf(str_b, "%lf", &val_b) != 1))
                fatal("invalid file format");
                
			//vectDouble[l * grid_cols + k] = val_b;
		}
	fclose(dp);

	
}

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

__global__ void calculate_temp(
    int iteration,          	//number of iteration
	double *power,          	//power input double
	float *power_reduced,		//power input float 
	double *temp_src,       	//temperature input/output double
	float *temp_src_reduced,	//temperature input/output float
	double *temp_dst,       	//temperature input/output double
	float *temp_dst_reduced,	//temperature input/output float
    int grid_cols,          	//Col of grid
    int grid_rows,          	//Row of grid
    int border_cols,        	//border offset
    int border_rows,       	 	//border offset
	double Cap,
	float Cap_reduced,             	
	double Rx, 
	float Rx_reduced,
	double Ry, 
	float Ry_reduced,
	double Rz, 
	float Rz_reduced,
	double step, 
	float step_reduced,
	double time_elapsed,
	float time_elapsed_reduced,
	float *relative_temp) {

	__shared__ double temp_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ double power_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ double temp_t[BLOCK_SIZE][BLOCK_SIZE]; // saving temparary temperature result

	__shared__ float temp_on_cuda_reduced[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float power_on_cuda_reduced[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float temp_t_reduced[BLOCK_SIZE][BLOCK_SIZE]; // saving temparary temperature result

	double amb_temp = double(80.0f);
	double step_div_Cap;
	double Rx_1, Ry_1, Rz_1;

	float amb_temp_reduced = float(80.0f);
	float step_div_Cap_reduced;
	float Rx_1_reduced, Ry_1_reduced, Rz_1_reduced;

	int bx = blockIdx.x;
	int by = blockIdx.y;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	step_div_Cap = step / Cap;
	step_div_Cap_reduced = step_reduced / Cap_reduced;

	Rx_1 = 1 / Rx;
	Ry_1 = 1 / Ry;
	Rz_1 = 1 / Rz;

	Rx_1_reduced = 1 / Rx_reduced;
	Ry_1_reduced = 1 / Ry_reduced;
	Rz_1_reduced = 1 / Rz_reduced;

	// each block finally computes result for a small block
	// after N iterations.
	// it is the non-overlapping small blocks that cover
	// all the input data

	// calculate the small block size
	int small_block_rows = BLOCK_SIZE - iteration * 2;        //EXPAND_RATE
	int small_block_cols = BLOCK_SIZE - iteration * 2;        //EXPAND_RATE

	// calculate the boundary for the block according to
	// the boundary of its small block
	int blkY = small_block_rows * by - border_rows;
	int blkX = small_block_cols * bx - border_cols;
	int blkYmax = blkY + BLOCK_SIZE - 1;
	int blkXmax = blkX + BLOCK_SIZE - 1;

	// calculate the global thread coordination
	int yidx = blkY + ty;
	int xidx = blkX + tx;

	// load data if it is within the valid input range
    int loadYidx = yidx; 
    int loadXidx = xidx;
	int index = grid_cols * loadYidx + loadXidx;

	if (IN_RANGE(loadYidx, 0, grid_rows - 1) && IN_RANGE(loadXidx, 0, grid_cols - 1)) {
		temp_on_cuda[ty][tx] = temp_src[index]; // Load the temperature data from global memory to shared memory
		power_on_cuda[ty][tx] = power[index]; // Load the power data from global memory to shared memory
		temp_on_cuda_reduced[ty][tx] = temp_src_reduced[index]; // Load the temperature data from global memory to shared memory
		power_on_cuda_reduced[ty][tx] = power_reduced[index]; // Load the power data from global memory to shared memory

	}
	__syncthreads();

	// effective range within this block that falls within
	// the valid range of the input data
	// used to rule out computation outside the boundary.
	int validYmin = (blkY < 0) ? -blkY : 0;
	int validYmax = (blkYmax > grid_rows - 1) ? BLOCK_SIZE - 1 - (blkYmax - grid_rows + 1) : BLOCK_SIZE - 1;
	int validXmin = (blkX < 0) ? -blkX : 0;
	int validXmax = (blkXmax > grid_cols - 1) ? BLOCK_SIZE - 1 - (blkXmax - grid_cols + 1) : BLOCK_SIZE - 1;

	int N = ty - 1;
	int S = ty + 1;
	int W = tx - 1;
	int E = tx + 1;

	N = (N < validYmin) ? validYmin : N;
	S = (S > validYmax) ? validYmax : S;
	W = (W < validXmin) ? validXmin : W;
	E = (E > validXmax) ? validXmax : E;

	bool computed;
	float relative = 0;
	for (int i = 0; i < iteration; i++) {
		computed = false;
		if ( IN_RANGE(tx, i + 1, BLOCK_SIZE-i-2) &&
		IN_RANGE(ty, i+1, BLOCK_SIZE-i-2) &&
		IN_RANGE(tx, validXmin, validXmax) &&
		IN_RANGE(ty, validYmin, validYmax)) {
			computed = true;

			//Compute with full precision
            temp_t[ty][tx] = 
            temp_on_cuda[ty][tx] + 
            step_div_Cap * (
                power_on_cuda[ty][tx] + 
                ( 
                    temp_on_cuda[S][tx] + 
                    temp_on_cuda[N][tx] -
                    double(2.0) * 
                    temp_on_cuda[ty][tx]
                ) * Ry_1 +
                (
                    temp_on_cuda[ty][E] + 
                    temp_on_cuda[ty][W]
                    - double(2.0) * 
                    temp_on_cuda[ty][tx]
                ) * Rx_1 +
                (
                    amb_temp - 
                    temp_on_cuda[ty][tx]
                ) * Rz_1 
			);
			
			//Compute with reduced precision
			temp_t_reduced[ty][tx] = 
            temp_on_cuda_reduced[ty][tx] + 
            step_div_Cap_reduced * (
                power_on_cuda_reduced[ty][tx] + 
                ( 
                    temp_on_cuda_reduced[S][tx] + 
                    temp_on_cuda_reduced[N][tx] -
                    float(2.0) * 
                    temp_on_cuda_reduced[ty][tx]
                ) * Ry_1_reduced +
                (
                    temp_on_cuda_reduced[ty][E] + 
                    temp_on_cuda_reduced[ty][W]
                    - float(2.0) * 
                    temp_on_cuda_reduced[ty][tx]
                ) * Rx_1_reduced +
                (
                    amb_temp_reduced - 
                    temp_on_cuda_reduced[ty][tx]
                ) * Rz_1_reduced 
			);
			
			if( ((float)temp_t[ty][tx] / temp_t_reduced[ty][tx]) >= 1.002247f){
				printf("relative: %f / %f = %f\n", (float)temp_t[ty][tx], temp_t_reduced[ty][tx], (float)temp_t[ty][tx] / temp_t_reduced[ty][tx]);
				printf("absolute: %f - %f = %f\n", (float)temp_t[ty][tx], temp_t_reduced[ty][tx], (float)temp_t[ty][tx] - temp_t_reduced[ty][tx]);
				printf("inputs float: Ry = %f, Rx = %f, Rz = %f, Cap = %f\n", Ry_1_reduced, Rx_1_reduced, Rz_1_reduced, step_div_Cap_reduced);
				printf("temp float: ty x tx = %f, S x tx = %f, N x tx = %f, ty x E = %f, ty x W = %f\n", 
				temp_on_cuda_reduced[ty][tx], temp_on_cuda_reduced[S][tx], temp_on_cuda_reduced[N][tx], temp_on_cuda_reduced[ty][E], temp_on_cuda_reduced[ty][W]);
				printf("power float: %f\n\n", power_on_cuda_reduced[ty][tx]);

				printf("inputs double: Ry = %lf, Rx = %lf, Rz = %lf, Cap = %lf\n", Ry_1, Rx_1, Rz_1, step_div_Cap);
				printf("temp double: ty x tx = %lf, S x tx = %lf, N x tx = %lf, ty x E = %lf, ty x W = %lf\n", 
				temp_on_cuda[ty][tx], temp_on_cuda[S][tx], temp_on_cuda[N][tx], temp_on_cuda[ty][E], temp_on_cuda[ty][W]);
				printf("power float: %lf\n", power_on_cuda[ty][tx]);
				//relative = ((float)temp_t[ty][tx] / temp_t_reduced[ty][tx]);
			}
		}
		__syncthreads();
		if (i == iteration - 1)
			break;
		if (computed) {
			//Assign the computation range
			temp_on_cuda[ty][tx] = temp_t[ty][tx];
			temp_on_cuda_reduced[ty][tx] = temp_t_reduced[ty][tx];
			//relative_temp[index] = relative;
			//printf("%d) %f\n", index, relative_temp[index]);
		}	 
		__syncthreads();
	}

	// update the global memory
	// after the last iteration, only threads coordinated within the
	// small block perform the calculation and switch on ``computed''
	if (computed) {
		temp_dst[index] = temp_t[ty][tx];
		temp_dst_reduced[index] = temp_t_reduced[ty][tx];
	}
}

/*
 compute N time steps
 */

int compute_tran_temp(	double *MatrixPowerDouble, double *MatrixTempDouble[2],
						float *MatrixPower, float *MatrixTemp[2],
						int col, int row, int total_iterations, int num_iterations, 
						int blockCols, int blockRows, int borderCols, int borderRows, float *relative_temp) {

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(blockCols, blockRows);
	

	double grid_height = chip_height / row;
	double grid_width = chip_width / col;

	double Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
	double Rx = grid_width / (double(2.0) * K_SI * t_chip * grid_height);
	double Ry = grid_height / (double(2.0) * K_SI * t_chip * grid_width);
	double Rz = t_chip / (K_SI * grid_height * grid_width);

	double max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
	double step = PRECISION / max_slope;
	
	double time_elapsed = double(0.001);

	float grid_height_ = chip_height / row;
	float grid_width_ = chip_width / col;

	float Cap_ = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width_ * grid_height_;
	float Rx_ = grid_width_ / (float(2.0) * K_SI * t_chip * grid_height_);
	float Ry_ = grid_height_ / (float(2.0) * K_SI * t_chip * grid_width_);
	float Rz_ = t_chip / (K_SI * grid_height_ * grid_width_);

	float max_slope_ = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
	float step_ = PRECISION / max_slope_;
	
	float time_elapsed_ = float(0.001);

	int src = 1, dst = 0;
	int t;

	for (t = 0; t < total_iterations; t += num_iterations) {
		int temp = src;
		src = dst;
		dst = temp;
		calculate_temp<<<dimGrid, dimBlock>>>(
				MIN(num_iterations, total_iterations - t), 
				MatrixPowerDouble, 
				MatrixPower,
				MatrixTempDouble[src], 
				MatrixTemp[src], 
				MatrixTempDouble[dst], 
				MatrixTemp[dst], 
				col, 
				row, 
				borderCols, 
				borderRows, 
				Cap, 
				Cap_, 
				Rx, 
				Rx_, 
				Ry, 
				Ry_, 
				Rz, 
				Rz_, 
				step, 
				step_, 
				time_elapsed, 
				time_elapsed_,
				relative_temp);
	}
	return dst;
}

void usage(int argc, char **argv) {
	fprintf(stderr,
			"Usage: %s <grid_rows/grid_cols> <pyramid_height> <sim_time> <temp_file> <power_file> <output_file>\n",
			argv[0]);
	fprintf(stderr,
			"\t<grid_rows/grid_cols>  - number of rows/cols in the grid (positive integer)\n");
	fprintf(stderr, "\t<pyramid_height> - pyramid heigh(positive integer)\n");
	fprintf(stderr, "\t<sim_time>   - number of iterations\n");
	fprintf(stderr,
			"\t<temp_file>  - name of the file containing the initial temperature values of each cell\n");
	fprintf(stderr,
			"\t<power_file> - name of the file containing the dissipated power values of each cell\n");
	fprintf(stderr, "\t<output_file> - name of the output file\n");
	exit(1);
}

int main(int argc, char** argv) {
	printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);

	run(argc, argv);

	return EXIT_SUCCESS;
}

void run(int argc, char** argv) {
	int size;
	int grid_rows, grid_cols;

	std::vector<float> FilesavingTempFloat, FilesavingPowerFloat, MatrixOutFloat;
	std::vector<double> FilesavingTempDouble, FilesavingPowerDouble, MatrixOutDouble;

	char *tfile_float, *pfile_float, *tfile_double, *pfile_double;

	int total_iterations = 60;
	int pyramid_height = 1;                                             // number of iterations

	if (argc != 8)
		usage(argc, argv);
    if ((grid_rows = atoi(argv[1])) <= 0 
        || (grid_cols = atoi(argv[1])) <= 0
        || (pyramid_height = atoi(argv[2])) <= 0 
        || (total_iterations = atoi(argv[3])) <= 0) 
        {
            usage(argc, argv);
        }

	tfile_float = argv[4];
	pfile_float = argv[5];
	tfile_double = argv[6];
	pfile_double = argv[7];

    size = grid_rows * grid_cols;
    

    /* --------------- pyramid parameters --------------- */
    
    # define EXPAND_RATE 2                                              // add one iteration will extend the pyramid base by 2 per each borderline
    int borderCols = (pyramid_height) * EXPAND_RATE / 2;	
    int borderRows = (pyramid_height) * EXPAND_RATE / 2;	
    int smallBlockCol = BLOCK_SIZE - (pyramid_height) * EXPAND_RATE;	
    int smallBlockRow = BLOCK_SIZE - (pyramid_height) * EXPAND_RATE;	
    int blockCols = grid_cols / smallBlockCol + ((grid_cols % smallBlockCol == 0) ? 0 : 1);	
    int blockRows = grid_rows / smallBlockRow + ((grid_rows % smallBlockRow == 0) ? 0 : 1);

    FilesavingTempFloat.resize(size);
    FilesavingPowerFloat.resize(size);
    MatrixOutFloat.resize(size);

    FilesavingTempDouble.resize(size);
    FilesavingPowerDouble.resize(size);
    MatrixOutDouble.resize(size);

    printf(
            "pyramidHeight: %d\ngridSize: [%d, %d]\nborder:[%d, %d]\nblockGrid:[%d, %d]\ntargetBlock:[%d, %d]\n",
            pyramid_height, grid_cols, grid_rows, borderCols, borderRows, blockCols, blockRows, smallBlockCol, smallBlockRow
    );

    //READ FOR FLOAT and convert to DOUBLE
    readinput(FilesavingTempDouble.data(), FilesavingTempFloat.data(), grid_rows, grid_cols, tfile_float, tfile_double);
    readinput(FilesavingPowerDouble.data(), FilesavingPowerFloat.data(), grid_rows, grid_cols, pfile_float, pfile_double);

    float *MatrixTemp[2], *MatrixPower;
    cudaMalloc((void**) &MatrixTemp[0], sizeof(float) * size);
    cudaMalloc((void**) &MatrixTemp[1], sizeof(float) * size);
    cudaMemcpy(MatrixTemp[0], FilesavingTempFloat.data(), sizeof(float) * size, cudaMemcpyHostToDevice);

    cudaMalloc((void**) &MatrixPower, sizeof(float) * size);
    cudaMemcpy(MatrixPower, FilesavingPowerFloat.data(), sizeof(float) * size, cudaMemcpyHostToDevice);

    // -------------------------------------------------------------------------

    double *MatrixTempDouble[2], *MatrixPowerDouble;
    cudaMalloc((void**) &MatrixTempDouble[0], sizeof(double) * size);
    cudaMalloc((void**) &MatrixTempDouble[1], sizeof(double) * size);
    cudaMemcpy(MatrixTempDouble[0], FilesavingTempDouble.data(), sizeof(double) * size, cudaMemcpyHostToDevice);

    cudaMalloc((void**) &MatrixPowerDouble, sizeof(double) * size);
    cudaMemcpy(MatrixPowerDouble, FilesavingPowerDouble.data(), sizeof(double) * size, cudaMemcpyHostToDevice);

    // -------------------------------------------------------------------------

	printf("Start computing the transient temperature\n");
	
	float *relativeGpu;
	float *relativeGpu_temp;
	std::vector<float> relativeCPU(size);
	std::vector<float> relativeCPU_temp(size);
	cudaMalloc((void**) &relativeGpu, sizeof(float) * size);
	cudaMalloc((void**) &relativeGpu_temp, sizeof(float) * size);

	int ret = compute_tran_temp(MatrixPowerDouble, MatrixTempDouble,
			MatrixPower, MatrixTemp, grid_cols, grid_rows, total_iterations, 
			pyramid_height, blockCols, blockRows, borderCols, borderRows, relativeGpu_temp);
    checkFrameworkErrors(cudaDeviceSynchronize());

    printf("Ending simulation\n");
    cudaMemcpy(MatrixOutFloat.data(), MatrixTemp[ret], sizeof(float) * size, cudaMemcpyDeviceToHost);

    cudaMemcpy(MatrixOutDouble.data(), MatrixTempDouble[ret], sizeof(double) * size, cudaMemcpyDeviceToHost);
    
    
    compareOutputGPU<<<grid_cols, grid_rows>>>(MatrixTempDouble[ret], MatrixTemp[ret], relativeGpu);
    checkFrameworkErrors(cudaDeviceSynchronize());

	cudaMemcpy(relativeCPU.data(), relativeGpu, sizeof(float) * size, cudaMemcpyDeviceToHost);
	cudaMemcpy(relativeCPU_temp.data(), relativeGpu_temp, sizeof(float) * size, cudaMemcpyDeviceToHost);

	//checkRelative(relativeCPU);
	//checkRelative(relativeCPU_temp);

	cudaFree(relativeGpu);
	cudaFree(relativeGpu_temp);
    cudaFree(MatrixPower);
    cudaFree(MatrixTemp[0]);
    cudaFree(MatrixTemp[1]);

    cudaFree(MatrixPowerDouble);
    cudaFree(MatrixTempDouble[0]);
    cudaFree(MatrixTempDouble[1]);
}