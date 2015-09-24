// Andre Driedger 1805536
// cmpt399 Lab1 Pthreads

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
// change to 4 threads
#define THRDS 4

int counter = 0;

void *printHello(void *threadid){
//	long tid;
//	long long bleh;
//	tid = (long)threadid;
//	printf("Hey! I'm thread #%ld!\n", tid);
	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	int i;
	for(i=0; i<25000000; i++){
//		printf("Addin'\n");
//		counter ++; // where sync error is due to context switch 
		__sync_fetch_and_add(&counter, 1);
		}
//	printf("Counter Total is %d\n", counter);a
	clock_gettime(CLOCK_REALTIME, &end);
	printf("Kernel time is %f\n", (double)((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec)*1e-9));

	pthread_exit(NULL);
//	timespec *end;
	}

int main(){
//	printf("addr of main: %llx\n", (unsigned long long int)main);
	pthread_t threads[THRDS];
	pthread_attr_t attr;
	int rc; 
	long t;
	void *status;

	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	for(t=0; t<THRDS; t++){
//		printf("Creating thread #%ld\n", t);
		rc = pthread_create(&threads[t], NULL, printHello, (void*)t);
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
//		printf("Completed join with thread %ld having status %ld\n", t, (long)status);
	}
	printf("Counter Total is %d\n", counter);
	pthread_exit(NULL);
}
