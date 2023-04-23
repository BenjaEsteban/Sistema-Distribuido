
// Create by Javier Peña Castro and Benjamín Rodríguez Valenzuela

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define DIM 20
#define K 3 
#define T 4 

float **ELEMS;
int *CLUSTER;
float **CENTROIDES;
int N_DB;

float distancia(float *p1, float *p2)
{
    // Calculo de distancia Euclidiana
    float dist = 0;
    for (int i = 0; i < DIM; i++)
    {
        dist += (p1[i] - p2[i]) * (p1[i] - p2[i]);
    }
    //printf("%f \n",sqrt(dist));
    return sqrt(dist);  
}

void kmeans()
{
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < K; i++)
        {
            // Inicializamos centroides al azar
            int idx = rand() % N_DB;    
            printf("%d\n", idx);          
            for (int j = 0; j < DIM; j++)
            {
                CENTROIDES[i][j] = ELEMS[idx][j];  //Asignamos los indices aleatorios a la variable CENTROIDES
            }
        }
    }

    int iter = 0;
    bool no_cambio = false; // Variable que nos identifica cuando ningún elemento cambie de centroide 

    while (!no_cambio)
    {
        no_cambio = true;           

        #pragma omp parallel // Abrimos una región paralela
        {
            bool cambio = false; // Variable que almacena el estado del cambio de elementos con repecto a los centroides 

            #pragma omp for // Dividimos el siguiente bucle for en subconjuntos y permitimos que cada subconjunto sea ejecutado en paralelo en diferentes hilos
            for (int i = 0; i < N_DB; i++)
            {
                // Encuentra el centroide más cercano
                float distancia_minima = distancia(ELEMS[i], CENTROIDES[0]); // Establecemos el primer centroide como el más cercano
                int cluster_mas_cercano = 0;
                for (int j = 1; j < K; j++)
                {
                    float dist = distancia(ELEMS[i], CENTROIDES[j]); // Calculamos la distancia de todos los elementos con respecto a todos los centroides exeptuando el primero
                    if (dist < distancia_minima)
                    {
                        distancia_minima = dist; // Actualizamos la distancia y se establece el centroide actual como el centroide más cercano
                        cluster_mas_cercano = j;
                    }
                }

                // Verificamos si los elementos asignados al cluster es diferente al cluster más cercano
                if (CLUSTER[i] != cluster_mas_cercano)
                {
                    #pragma omp atomic write // Garantizamos que ningún otro hilo escriba en la misma posición de memoria al mismo tiempo
                    CLUSTER[i] = cluster_mas_cercano; // Reasignamos el elemento al cluster más cercano
                    cambio = true; 
                }
            }
            // Consultamos el estado de la variable cambio, es decir, si hubieron elementos que cambiaron de cluster
            if (cambio)
            {
                no_cambio = false;
            }

            #pragma omp barrier // Garantizamos que todos los hilos han terminado de asignar los elementos a sus respectivos clústeres
            #pragma omp for // Dividimos nuevamente el proceso en subconjuntos 
            for (int c = 0; c < K; c++)
            {
                float suma[DIM] = {0}; // Variable que acumulará la suma de los vectores, para su posterior calculo de la media
                int num_elem = 0;  // Contador de elementos

                for (int i = 0; i < N_DB; i++)
                {
                    if (CLUSTER[i] == c) // Consultamos si el elemento pertenece al cluster 
                    {
                        for (int j = 0; j < DIM; j++)
                        {
                            suma[j] += ELEMS[i][j]; // Realizamos la suma de los vectores de cada cluster
                            //printf("%f\n", suma[j]);
                        }
                        num_elem++; // Contabilizamos la cantidad de elementos por cluster  
                        //printf("%d\n",num_elem);
                    }
                    //printf("%d\n", CLUSTER[i]);
                }

                if (num_elem > 0) // Consultamos si existe un elemento en el cluster para calculamos el nuevo centroide
                {
                    for (int j = 0; j < DIM; j++)
                    {
                        #pragma omp atomic update // Actualizamos los valores de los centroides de manera atómica
                        CENTROIDES[c][j] += suma[j];  
                        //printf("%f\n", CENTROIDES[c][j]);
                    }
                    for (int j = 0; j < DIM; j++)
                    {
                        CENTROIDES[c][j] /= num_elem;
                        //printf("%f\n", CENTROIDES[c][j]);
                    }
                }
            }
        }
        iter++;
        //printf("%d\n", iter);
    }
}

// Función encargada de eliminar la memoria
void liberar_memoria()
{
    // Liberamos la memoria de la variable referente de la base de datos 
    for (int i = 0; i < N_DB; i++)
    {
        free(ELEMS[i]);
    }
    free(ELEMS);

    // Liberamos la memoria de la variable referente a los cluster
    free(CLUSTER);

    // Liberamos la memoria de la variable referente a los centroides 
    for (int i = 0; i < K; i++)
    {
        free(CENTROIDES[i]);
    }
    free(CENTROIDES);

    //printf("Memoria liberada correctamente\n");
}

int main()
{   
    // Inicializamos el tiempo para el posterior análisis
    double start_time = omp_get_wtime();
    
    scanf("%d", &N_DB);

    // Asignamos memoria para los elementos de la base de datos
    ELEMS = (float **)malloc(sizeof(float *) * N_DB);
    for (int i = 0; i < N_DB; i++)
    {
        ELEMS[i] = (float *)malloc(sizeof(float) * DIM);
    }

    // Lectura de datos de entrada
    for (int i = 0; i < N_DB; i++)
    {
        for (int j = 0; j < DIM; j++)
        {
            scanf("%f", &(ELEMS[i][j]));
        }
    }

    // Asignamos memoria para asignaciones de clúster
    CLUSTER = (int *)malloc(sizeof(int) * N_DB);

    // Asignamos memoria para centroides
    CENTROIDES = (float **)malloc(sizeof(float *) * K);
    for (int i = 0; i < K; i++)
    {
        CENTROIDES[i] = (float *)malloc(sizeof(float) * DIM);
    }

    // Establecemos el número de hilos
    omp_set_num_threads(T);

    // Corremos algoritmo K-mean 
    kmeans();

    // Imprimir resultados
    //printf("%d\n", N_DB);
    for (int i = 0; i < N_DB; i++)
    {
        for (int j = 0; j < DIM; j++)
        {
            //printf("%.2f\t", ELEMS[i][j]);
        }
        printf("%d\n", CLUSTER[i]);
    }

    liberar_memoria();

    double end_time = omp_get_wtime(); 
    double tiempo_paralelo = end_time - start_time;

    //Tiempo final de ejecución
    //printf("%f", tiempo_paralelo);
    
    return 0;
}