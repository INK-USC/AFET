
#ifndef utils_h
#define utils_h

#include <stdlib.h>
#include <stdio.h>
#include "hierarchy.h"
#include <vector>

#define BUFFER_SIZE 512
#define SMALL_BUFFER_SIZE 16

double rank(int);
double dot(double *, double *, int);
void print_matrix(char *, double *, int, int, char **);
void load_data(char *, int **, int *);
void load_data_path(char *, int **, int *, Hierarchy*, int);
std::vector<std::vector<int> > load_data_noise(char *, int **, int *, Hierarchy*, int);
std::vector<std::vector<int> > load_data_noise_path(char *, int **, int *, Hierarchy*, int);
int count_lines(char *);
char ** read_names(char *, int);
double ** load_weights(char * filename, int count, double alpha, int iscorr, Hierarchy* hierarchy);
int max_score(double *Ax, double* B, int* types, int pt_number, int embed_size);
std::vector<int> get_rank(double *Ax, double* B, double** weight, int p_sample, int* types, int pt_number, int embed_size);
std::vector<int> get_rank(double *Ax, double* B, int p_sample, int* types, int pt_number, int embed_size);
#endif /* utils_h */
