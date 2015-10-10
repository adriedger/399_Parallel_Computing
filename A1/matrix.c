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

int *makeRandSet(int size){
	int *set;
	int i;
	srand(time(NULL));
	set = malloc(((size*size) * sizeof(int)));
	if(set == NULL)
		return NULL;
	for(i=0; i<(size*size); i++)
		set[i] = rand() % 1024;
	return set;
}

void *matrixMulti(void *t){
	int tid = (long)t;
//	printf("Hey! I'm thread #%d\n", tid);
	int x, y, i;
//	int A_Start = 0, B_Start = tid; 
//	int thread_index = tid;
//	for(i=0; i<3; i++){
//		for(x=xStart, y=yStart; x<LENGTH; x++, y+=LENGTH)
//			AB[tjob] += A[x] * B[y];
//		tjob += LENGTH;
//		printf("%d\n", tjob);  
//		xStart += LENGTH;
//		printf("%d\n", AB[tjob]);
//	}

	int regular_passes = (LENGTH*LENGTH)/THRDS;
//	int t_per_pass = THRDS;
//	int t_last_pass = (LENGTH*LENGTH) % THRDS;

	int A_Row = tid / LENGTH;
	int B_Column = tid % LENGTH; 
	int index = tid;

	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);
	
	for(i=0; i<regular_passes+1; i++){
		if(index < LENGTH*LENGTH){
//			printf("pass %d thread %d row %d column %d\n", i, tid, A_Row, B_Column);
	
			for(x=A_Row*LENGTH, y=B_Column; x<(A_Row+1)*LENGTH; x++, y+=LENGTH)
				AB[index] += A[x] * B[y];
			
//			printf("index %d AB %d\n\n", index, AB[index]);
	//		A_Row += THRDS;
	//		B_Column += index % LENGTH;
			index += THRDS;
			A_Row = index / LENGTH;
			B_Column = index % LENGTH;
		}
	}
//	AB[3] = 69;
//	printf("%d, thread id %d\n", AB[tjob], tid);
	clock_gettime(CLOCK_REALTIME, &end);
	printf("Kernel time is %f\n", (double)((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec)*1e-9));

	pthread_exit(NULL);
}

int main(int argc, char *argv[]){
	
	int i;
	LENGTH = atoi(argv[argc-2]);
	THRDS = atoi(argv[argc-1]);

	A = makeRandSet(LENGTH);	
	B = makeRandSet(LENGTH);
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
