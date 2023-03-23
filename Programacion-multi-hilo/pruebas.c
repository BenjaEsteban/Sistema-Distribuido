#include <stdio.h>
#include <omp.h>

int main(){
    printf("Nuestro sistema tiene %d núcleos\n\n", omp_get_num_procs());

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nprocs = omp_get_num_threads();
        printf("Hola! Soy el hilo %d, de un total de %d.\n", tid, nprocs);
    }
}
