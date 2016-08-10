

#include "utils.h"

double rank(int k){
    double loss = 0;
    int i ;
    for (i=1; i<=k; ++i) {
        loss += 1.0/i;
    }
    return loss;
}

double dot(double *Ax , double *Bi, int len){
    double result = 0;
    for (int i = 0; i != len; i++) {
        result += Ax[i] * Bi[i];
    }
    return result;
}

void print_matrix(char *filename, double *array, int nrows, int ncolumns, char** names) {
    FILE *file = fopen( filename, "w" );
    if ( file == NULL ){
        printf( "Could not open file %s\n", filename);
        exit(EXIT_FAILURE);
    }
    fprintf(file,"%d %d\n", nrows, ncolumns);
    for (int i = 0;i != nrows; i++){
        fprintf(file, "%s\t%d\t", names[i], i);
        for (int j = 0; j != ncolumns; j++) {
            fprintf(file, "%f ",array[i * ncolumns + j]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}


void load_data(char *filename, int** data, int* count)
{
    FILE *fi = fopen(filename, "rb");
    int mid, fid, wei;
    while (fscanf(fi, "%d %d %d", &mid, &fid, &wei) == 3)
        data[mid][count[mid]++] = fid;
    fclose(fi);
}

void load_data_path(char *filename, int** data, int* count, Hierarchy* hierarchy, int row)
{
    FILE *fi = fopen(filename, "rb");
    int mid, fid, wei;
    while (fscanf(fi, "%d %d %d", &mid, &fid, &wei) == 3)
        data[mid][count[mid]++] = fid;
    for(int i =0;i<row;++i){
        std::set<int> new_labels = hierarchy->getfinetype(data[i], count[i]);
        std::set<int>::iterator it;
        int j = 0;
        for (it = new_labels.begin(); it!=new_labels.end(); it++) {
            data[i][j] = *it;
            ++j;
        }
        count[i] = j;
    }
    fclose(fi);
}

std::vector<std::vector<int> > load_data_noise(char *filename, int **data, int *count, Hierarchy* hierarchy, int row) {
    std::vector<std::vector<int> > result;
    FILE *fi = fopen(filename, "rb");
    int mid, fid, wei;
    std::set<int> mids;
    while (fscanf(fi, "%d %d %d", &mid, &fid, &wei) == 3){
        data[mid][count[mid]++] = fid;
        mids.insert(mid);
    }
    fclose(fi);
    std::vector<int> clean;
    std::vector<int> noise;
    std::set<int>::iterator it;
    for(it = mids.begin(); it != mids.end(); ++it){
        int i = *it;
        if (hierarchy->isclean(data[i], count[i])) {
            clean.push_back(i);
        }else{
            noise.push_back(i);
        }
    }
    result.push_back(clean);
    result.push_back(noise);
    return result;
}

std::vector<std::vector<int> > load_data_noise_path(char *filename, int **data, int *count, Hierarchy* hierarchy, int row) {
    std::vector<std::vector<int> > result;
    FILE *fi = fopen(filename, "rb");
    int mid, fid, wei;
    std::set<int> mids;
    while (fscanf(fi, "%d %d %d", &mid, &fid, &wei) == 3){
        data[mid][count[mid]++] = fid;
        mids.insert(mid);
    }
    fclose(fi);
    std::vector<int> clean;
    std::vector<int> noise;
    std::set<int>::iterator it;
    for(it = mids.begin(); it != mids.end(); ++it){
        int i = *it;
        if (hierarchy->isclean(data[i], count[i])) {
            clean.push_back(i);
        }else{
            noise.push_back(i);
        }
        std::set<int> new_labels = hierarchy->getfinetype(data[i], count[i]);
        std::set<int>::iterator it2;
        int j = 0;
        for (it2 = new_labels.begin(); it2!=new_labels.end(); it2++) {
            data[i][j] = *it2;
            ++j;
        }
        count[i] = j;
    }
    result.push_back(clean);
    result.push_back(noise);
    return result;
}


int count_lines(char * filename) {
    FILE *file = fopen( filename, "r" );
    if ( file == NULL ){
        printf( "Could not open file %s\n", filename);
        exit(EXIT_FAILURE);
    }
    else{
        char line[BUFFER_SIZE];
        int count = 0;
        while(fgets(line, sizeof(line), file) != NULL){
            sscanf(line, "%*s\t%d", &count);
        }
        fclose(file);
        return count+1;
    }
}

char ** read_names(char * filename, int count) {
    char ** array;
    array = (char **)malloc(count * sizeof(char **));
    if(array == NULL){
        printf("out of memory!\n");
        exit(EXIT_FAILURE);
    }
    for(int i = 0; i < count; i++){
        array[i] = (char *)malloc(BUFFER_SIZE * sizeof(char));
        if(array[i] == NULL){
            printf("out of memory!\n");
            exit(EXIT_FAILURE);
        }
    }
    FILE *file = fopen( filename, "r" );
    if ( file == NULL ){
        printf( "Could not open file %s\n", filename);
        exit(EXIT_FAILURE);
    }
    else{
        char line[BUFFER_SIZE];
        int i = 0;
        while(fgets(line, sizeof(line), file) != NULL){
            sscanf(line, "%s\t%*d", array[i]);
            ++i;
        }
        fclose(file);
    }
    return array;
}

double ** load_weights(char * filename, int count, double alpha, int iscorr, Hierarchy* hierarchy) {
    double ** corr = (double **)malloc(count * sizeof(double *));
    for(int i = 0; i < count; i++){
        corr[i] = (double *)calloc(count, sizeof(double));
        if(corr[i] == NULL){
            printf("out of memory!\n");
            exit(EXIT_FAILURE);
        }
        for(int j = 0;j < count; j++) {
            corr[i][j] = 1.0/(alpha);
        }
    }
    FILE *fi = fopen(filename, "rb");
    int mid, fid;
    double wei;
    while (fscanf(fi, "%d %d %lf", &mid, &fid, &wei) == 3)
        if(iscorr==2) corr[mid][fid] = wei;
        else corr[mid][fid] = 1.0/(alpha+wei);
    fclose(fi);
    if (iscorr==0 ) { // 1/alpha+corr or shortes path length
        return corr;
    }else if (iscorr==3|| iscorr==2){
        for(int i = 0; i < count; i++){
            for(int j = 0;j < count; j++) {
                corr[i][j] += 1.0;
            }
        }
    }else{  // edit distance
        double ** distance = (double **)malloc(count * sizeof(double *));
        for(int i = 0; i < count; i++){
            distance[i] = (double *)calloc(count, sizeof(double));
            if(distance[i] == NULL){
                printf("out of memory!\n");
                exit(EXIT_FAILURE);
            }
            for(int j = 0;j < count; j++) {
                if(i==j){
                    distance[i][j] = 0;
                }else{
                    distance[i][j] =hierarchy->edit_distance(i, j, corr);
                }
            }
        }
        return distance;
    }
    return corr;
}

int max_score(double *Ax, double* B, int* types, int pt_number, int embed_size){
    int p_sample = types[0];
    double s1_max = dot(Ax, B + p_sample * embed_size, embed_size);
    for (int i = 1; i<pt_number; ++i) {
        int temp = types[i];
        double s1  = dot(Ax, B + temp * embed_size, embed_size);
        if(s1 > s1_max) {
            p_sample = temp;
            s1_max = s1;
        }
    }
    return p_sample;
}

std::vector<int> get_rank(double *Ax, double* B, double** weight, int p_sample, int* types, int pt_number, int embed_size){
    double s1 = dot(Ax, B + p_sample * embed_size, embed_size);
    int n_sample = types[0];
    std::vector<int> result;
    double s2_max = dot(Ax, B + n_sample * embed_size, embed_size);
    for (int i = 0; i<pt_number; ++i) {
        double s2 = dot(Ax, B + types[i] * embed_size, embed_size);
        if (s1-s2<weight[p_sample][types[i]]) {
            result.push_back(types[i]);
        }
        if(s2 > s2_max) {
            n_sample = types[i];
            s2_max = s2;
        }
    }
    result.push_back(n_sample);
    return result;
}

std::vector<int> get_rank(double *Ax, double* B, int p_sample, int* types, int pt_number, int embed_size){
    double s1 = dot(Ax, B + p_sample * embed_size, embed_size);
    int n_sample = types[0];
    std::vector<int> result;
    double s2_max = dot(Ax, B + n_sample * embed_size, embed_size);
    for (int i = 0; i<pt_number; ++i) {
        double s2 = dot(Ax, B + types[i] * embed_size, embed_size);
        if (s1-s2<1) {
            result.push_back(types[i]);
        }
        if(s2 > s2_max) {
            n_sample = types[i];
            s2_max = s2;
        }
    }
    result.push_back(n_sample);
    return result;
}
