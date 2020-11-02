#include <stdio.h>
#include<stdint.h>

#define HOTSPOT_STR_SIZE 256
#define ERROR 0
#define SUCCESS 1
#define SUB_ABS(lhs, rhs) ((lhs > rhs) ? (lhs - rhs) : (rhs - lhs))

__global__
void compare(float lhs, double rhs, float *dst) {
        float relative = __fdividef(lhs, float(rhs));
        *dst = relative;
}

__global__
void absolute(float lhs, double rhs, uint32_t *diff) {
        float rhs_as_float = float(rhs);
	uint32_t lhs_data = *((uint32_t*) &lhs);
	uint32_t rhs_data = *((uint32_t*) &rhs_as_float);

	*diff = SUB_ABS(lhs_data, rhs_data);
}

int main(int argc, char** argv)
{
        char *file_float, *file_double;
        float val_float=0;
        float *dst;
        double val_double=0;
        uint32_t *diff;

        FILE *pf_file, *pd_file;

        file_float = argv[1];
        file_double = argv[2];
        pf_file = fopen(file_float, "rb");
        pd_file = fopen(file_double, "rb");

        if (pf_file == 0 || pd_file == 0)
                return ERROR;
        cudaMallocManaged(&dst, sizeof(double));
        cudaMallocManaged(&diff, sizeof(double));

        while(!feof(pf_file) || !feof(pd_file)) 
        {
                fread(&val_float, sizeof(float), 1, pf_file);
                fread(&val_double, sizeof(double), 1, pd_file);

                printf("FLOAT VALUE: %f, DOUBLE VALUE: %lf, DOUBLE CASTED VALUE: %f", val_float, val_double, (float)val_double);
                compare<<<1, 1>>>(val_float, val_double, dst);
                cudaDeviceSynchronize();
                printf("RELATIVE ERROR: %f", *dst);

                absolute<<<1, 1>>>(val_float, val_double, diff);
                printf("ABSOLUTE ERROR: %u", *diff);
                cudaDeviceSynchronize();
        }
        
        return SUCCESS;
}
