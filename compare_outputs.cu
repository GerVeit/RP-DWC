#include <stdio.h>

#define HOTSPOT_STR_SIZE 256
#define ERROR 0
#define SUCCESS 1

int main(int argc, char** argv)
{
        char *file_float, *file_double;
        char str[HOTSPOT_STR_SIZE], str1[HOTSPOT_STR_SIZE];
        FILE *pf_file, *pd_file;
        file_float = argv[1];
        file_double = argv[2];
        float val_float=0;
        double val_double=0;

        pf_file = fopen(file_float, "r");
        pd_file = fopen(file_double, "r");

        if (pf_file == 0 || pd_file == 0)
                return ERROR;

        while(!feof(pf_file) || !feof(pd_file)) 
        {
                fgets(str, HOTSPOT_STR_SIZE, pf_file);
                fgets(str1, HOTSPOT_STR_SIZE, pd_file);

                if(sscanf(str, "%f", &val_float) != 1)
                        return ERROR;
                if(sscanf(str1, "%lf", &val_double) != 1)
                        return ERROR;
        }
        
        return SUCCESS;
}
