// Andre Driedger 1805536
// CMPT399 A1 Matrix Product

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

//int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
//int B[] = {1, 0, 0, 0 ,1 ,0 ,0 ,0, 1};
//int AB[9];
//double mSize = sqrt(sizeof(A)/sizeof(A[0]));

int *A; 
int *B;
int *AB;
int LENGTH, THRDS;

int *makeRandSet(int size, int seed){
	int *set;
	int i;
	srand(seed);
	set = malloc(((size*size) * sizeof(int)));
	if(set == NULL)
		return NULL;
	for(i=0; i<(size*size); i++)
		set[i] = rand() % 10;
	return set;
}

void *matrixMulti(void *t){
	int tid = (long)t;
	int x, y, i;

	int regular_passes = (LENGTH*LENGTH)/THRDS;

	int A_Row = tid / LENGTH;
	int B_Column = tid % LENGTH; 
	int index = tid;

	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	for(i=0; i<LENGTH*LENGTH; i++){

		int A_Simple = (i / LENGTH) * LENGTH + tid;
		int B_Simple = (i % LENGTH) + (LENGTH * tid);

		if(tid < LENGTH){
//			printf("%d\n", ((i/LENGTH)+1)*LENGTH);
			for(x = A_Simple, y = B_Simple ; x<((i/LENGTH)+1)*LENGTH; x+=THRDS, y+=LENGTH*THRDS){
//				printf("%d ", A[x]*B[y]);
//				printf("%d ", y);	
				__sync_fetch_and_add(&AB[i], A[x]*B[y]);
			}
		}
	}
/*	
	for(i=0; i<regular_passes+1; i++){
		if(index < LENGTH*LENGTH){
	
			for(x=A_Row*LENGTH, y=B_Column; x<(A_Row+1)*LENGTH; x++, y+=LENGTH)
				AB[index] += A[x] * B[y];
			
			index += THRDS;
			A_Row = index / LENGTH;
			B_Column = index % LENGTH;
		}
	}
*/
	clock_gettime(CLOCK_REALTIME, &end);
	printf("Kernel time is %f\n", (double)((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec)*1e-9));

	pthread_exit(NULL);
}

int main(int argc, char *argv[]){
	
	int i;
	LENGTH = atoi(argv[argc-2]);
	THRDS = atoi(argv[argc-1]);

	A = makeRandSet(LENGTH, 1);	
	B = makeRandSet(LENGTH, time(NULL));
	AB = malloc((LENGTH*LENGTH) * sizeof(int));
	
	if(!strcmp(argv[argc-3], "-O")){
		for(i=0; i<LENGTH*LENGTH; i++){
			printf("%d ", A[i]);
			if(i % LENGTH == LENGTH-1)
				printf("\n");
		}
		printf("\n");
		for(i=0; i<LENGTH*LENGTH; i++){
			printf("%d ", B[i]);
			if(i % LENGTH == LENGTH-1)
				printf("\n");
		}
		printf("\n");
		
	}


	pthread_t threads[THRDS];
	pthread_attr_t attr;
	int rc;
	long t;
	void *status;

	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	for(t=0; t<THRDS; t++){
		rc=pthread_create(&threads[t], NULL, matrixMulti, (void*)t);
		if(rc){
			printf("ERROR: RETURN CODE FROM PTHREAD_CREATE() IS %d\n", rc);
			exit(-1);
		}
	}
	pthread_attr_destroy(&attr);
	for(t=0; t<THRDS; t++){
		rc = pthread_join(threads[t], &status);
		if(rc){
			printf("ERROR: RETURN CODE FROM PTHREAD_JOIN() IS %d\n", rc);
			exit(-1);
		}
	}
	
	printf("\n");
	if(!strcmp(argv[argc-3], "-O")){
		for(i=0; i<LENGTH*LENGTH; i++){
			printf("%d ", AB[i]);
			if(i % LENGTH == LENGTH-1)
				printf("\n");
		}
		printf("\n");
	}
	
	free(A);
	free(B);
	free(AB);

	pthread_exit(NULL);
}
