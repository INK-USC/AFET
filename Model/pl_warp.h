

#ifndef pl_warp_h
#define pl_warp_h

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include "utils.h"
#include <vector>
#include <algorithm>
#define BUFFER_SIZE 512
#define SMALL_BUFFER_SIZE 16

double lr;
double alpha;
int max_iter;
int num_threads;
int distance;
int ocsample; // 0: negative sampling, 1: all negative sampling
int onsample; // 0: best negative sampling, 1: all negative sampling
int K; // the number of negative sampling
double C; // regularization termf
double WA; // weight 
char** feautre_name;
char** type_name;
int feature_count;
int type_count;
int mention_count;
int embed_size;
double *A, *Ag;
double *B, *Bg;
int *An, *Bn;
int** train_x;
int* x_count;
int** train_y;
int* y_count;
int iter;
double error;
std::vector<int> mention_set; // clean instances
std::vector<int> mention_set_noise; // noise instances
double ** weights; // edit distances between types
Hierarchy* hierarchy;


double* malloc_matrix_double(int, int);
void free_matrix_double(double *);
void get_negatives(int *, int *, int, int);
double gradient_warp(double *, double *, double *, int *, int *, int* , int, int, int);
double gradient_partial(double *, double *, double *, int *, int *, int* , int, int, int);
void update_embedding();
void *train_BCD_thread(void *id);
void *train_BCD_thread_noise(void *id);


#endif /* pl_warp_h */
