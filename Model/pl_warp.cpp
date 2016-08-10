

#include "pl_warp.h"

double* malloc_matrix_double_zero(int, int);

int main(int argc, const char * argv[]) {
    if ( argc != 12 ) {
        /* We print argv[0] assuming it is the program name */
        printf( "usage: %s -DATA -EMBED_SIZE -LR -MAX_ITER -NUM_THREAD -ALPHA -DISTANCE -OCSAMPLE -ONSAMPLE -K -C\n", argv[0] );
    }
    else {
        const char *indir = argv[1];
        embed_size = atoi(argv[2]);
        lr = atof(argv[3]);
        max_iter = atoi(argv[4]);
        num_threads = atoi(argv[5]);
        alpha = atof(argv[6]);
        distance = atoi(argv[7]);
        ocsample = atoi(argv[8]);  
        onsample = atoi(argv[9]);
        K = atoi(argv[10]);
        C = atof(argv[11]);
        char filename[BUFFER_SIZE];
        
        time_t t;
        srand((unsigned) time(&t));
        
        /* Load number of features, types, and mentions */
        snprintf(filename, sizeof(filename), "Intermediate/%s/feature.txt", indir);
        feature_count = count_lines(filename);
        feautre_name = read_names(filename, feature_count);
        snprintf(filename, sizeof(filename), "Intermediate/%s/type.txt", indir);
        type_count = count_lines(filename);
        type_name = read_names(filename, type_count);
        snprintf(filename, sizeof(filename), "Intermediate/%s/mention.txt", indir);
        mention_count = count_lines(filename);
        printf("type: %d, feature: %d, mention: %d\n",type_count, feature_count, mention_count);
        
        /* Load type hierarchy */
        snprintf(filename, sizeof(filename), "Intermediate/%s/supertype.txt", indir);
        hierarchy = new Hierarchy(filename);
        
        /* Load edit distances between types */
        if (distance == 2) { // shortest path length
            snprintf(filename, sizeof(filename), "Intermediate/%s/type_type_sp.txt", indir);
        }else{
            snprintf(filename, sizeof(filename), "Intermediate/%s/type_type_kb.txt", indir);
        }
        weights = load_weights(filename, type_count, alpha, distance, hierarchy);
        
        /* Initialize matrix A and B*/
        A = malloc_matrix_double(feature_count, embed_size);
        B = malloc_matrix_double(type_count, embed_size);
        printf("Fininsh initialize matrix A and B\n");
        Ag = (double *)calloc(feature_count * embed_size, sizeof(double));
        Bg = (double *)calloc(type_count * embed_size, sizeof(double));
        An = (int *)calloc(feature_count, sizeof(int));
        Bn = (int *)calloc(type_count, sizeof(int));
        
        snprintf(filename, sizeof(filename), "Intermediate/%s/mention_feature.txt", indir);
        train_x = (int **)malloc(mention_count * sizeof(int *));
        x_count = (int *)calloc(mention_count, sizeof(int));
        for(int i = 0; i < mention_count; i++){
            train_x[i] = (int *)calloc(BUFFER_SIZE, sizeof(int));
            if(train_x[i] == NULL){
                printf("out of memory!\n");
                exit(EXIT_FAILURE);
            }
        }
        load_data(filename, train_x, x_count);
        snprintf(filename, sizeof(filename), "Intermediate/%s/mention_type.txt", indir);
        
        train_y = (int **)malloc(mention_count * sizeof(int *));
        y_count = (int *)calloc(mention_count, sizeof(int));
        for(int i = 0; i < mention_count; i++){
            train_y[i] = (int *)calloc(SMALL_BUFFER_SIZE, sizeof(int));
            if(train_y[i] == NULL){
                printf("out of memory!\n");
                exit(EXIT_FAILURE);
            }
        }
        std::vector<std::vector<int> >clean_and_noise=load_data_noise(filename, train_y, y_count, hierarchy, mention_count);
        mention_set = clean_and_noise[0];
        mention_set_noise  = clean_and_noise[1];
        
        printf("Start training process\n");
        printf("Clean examples: %d, noise examples: %d\n", (int)mention_set.size(), (int)mention_set_noise.size());
        long a;
        pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
        for (iter = 0; iter != max_iter; iter++)
        {
            error = 0;
            std::random_shuffle(mention_set.begin(), mention_set.end());
            std::random_shuffle(mention_set_noise.begin(), mention_set_noise.end());
            for (a = 0; a < num_threads; a++)
                pthread_create(&pt[a], NULL, train_BCD_thread, (void *)a);
            
            for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
            printf("Iter:%d, error:%f\n",iter, error);
            
            update_embedding();
        }
        /* Save matrix A and B*/
        snprintf(filename, sizeof(filename), "Results/%s/emb_pl_warp_bipartite_feature.txt", indir);
        print_matrix(filename, A, feature_count, embed_size, feautre_name);
        snprintf(filename, sizeof(filename), "Results/%s/emb_pl_warp_bipartite_type.txt", indir);
        print_matrix(filename, B, type_count, embed_size, type_name);
        free_matrix_double(A);
        return 0;
    }
}

void *train_BCD_thread(void *id) {
    int *MF, *MT, *NT;
    int f_number, t_number;
    NT = (int *)malloc(type_count * sizeof(int));
    
    double *dA = (double *)malloc(embed_size*sizeof(double));
    double *Ax = (double *)malloc(embed_size*sizeof(double));
    double *dB = (double *)malloc(type_count*embed_size*sizeof(double));
    
    long long tid = (long long)id;
    std::vector<int> dataset;
    if (tid%2==0) { // clean instances
        dataset = mention_set;
    }else{  // noise instances
        dataset = mention_set_noise;
    }
    int begin = (int) (dataset.size() / (num_threads/2)*(tid/2));
    int end = (int)(dataset.size() / (num_threads/2)*(tid/2+1));
    if (end > (int)dataset.size()) end = (int)dataset.size();
    double thread_error = 0;
    for (int T = begin; T != end; T++){
        int i = dataset[T];
        
        MF =train_x[i];
        MT = train_y[i];
        f_number = x_count[i];
        t_number = y_count[i];
        get_negatives(NT, MT, type_count, type_count-t_number);
        if (tid%2==0) {
            thread_error += gradient_warp(dA, dB, Ax, MF, MT, NT, f_number, t_number, type_count-t_number);
        }else{
            thread_error += gradient_partial(dA, dB, Ax, MF, MT, NT, f_number, t_number, type_count-t_number);
        }

    }
    printf("Thread:%lld, Iter:%d, DONE\n", tid, iter);
    error += thread_error;
    
    free(NT);
    free(dA);
    free(dB);
    free(Ax);
    pthread_exit(NULL);
}

double* malloc_matrix_double(int nrows, int ncolumns) {
    double *array;
    array = (double *)malloc(nrows * ncolumns * sizeof(double *));
    if(array == NULL){
        printf("out of memory!\n");
        exit(EXIT_FAILURE);
    }
    for (int k = 0; k != nrows * ncolumns; k++)
        array[k] =((double) rand() / (RAND_MAX) - 0.5);
    return array;
}

double* malloc_matrix_double_zero(int nrows, int ncolumns) {
    double *array;
    array = (double *)malloc(nrows * ncolumns * sizeof(double *));
    if(array == NULL){
        printf("out of memory!\n");
        exit(EXIT_FAILURE);
    }
    for (int k = 0; k != nrows * ncolumns; k++)
        array[k] =1/sqrt(ncolumns);;
    return array;
}

void free_matrix_double(double* array) {
    free(array);
}

int compare (const void * a, const void * b){
    return ( *(int*)a - *(int*)b );
}

inline void get_negatives(int *negative, int* positive_types, int type_count, int nt_number){
    qsort(positive_types, type_count-nt_number, sizeof(int), compare);
    int i = 0;int j = 0;int p = 0;
    for (; i<type_count; ++i) {
        if (positive_types[p] == i) {
            ++p;
            if (p>=type_count-nt_number) {
                ++i;
                break;
            }
        }else{
            negative[j] = i;
            ++j;
        }
        
    }
    while (i<type_count) {
        negative[j] = i;
        ++j;
        ++i;
    }
}

double gradient_warp(double *dA, double *dB, double *Ax, int *features, int *positive_types, int *negative_types, int f_number, int pt_number, int nt_number){
    double cur_error = 0;
    int i,j;
    
    for (i = 0; i != embed_size; i++) dA[i] = 0;
    for (i = 0; i != pt_number + nt_number; i++) for (j = 0; j != embed_size; j++) dB[i * embed_size + j] = 0;
    for (i = 0; i != embed_size; i++) Ax[i]=0;
    
    for (i = 0; i<f_number; ++i) {
        int f = features[i];
        for (j = 0;j<embed_size;++j){
            Ax[j] += A[f * embed_size + j];
        }
    }
    for (i = 0; i<pt_number; ++i) {
        int p_sample = positive_types[i];
        std::vector<int> result = get_rank(Ax, B, weights, p_sample, negative_types, nt_number, embed_size);
        double s1 = dot(Ax, B + p_sample * embed_size, embed_size);
        int r = (int)result.size()-1;
        if (r>0) { // find negative violations
            if (ocsample == 0) { /* K negative sampling */
                result.pop_back();
                std::random_shuffle(result.begin(), result.end());
                int min_k = std::min(K,(int)result.size());
                for(j = 0;j<min_k ; ++j){
                    int n_sample = result[j];
                    double s2= dot(Ax, B + n_sample * embed_size, embed_size);
                    double L = rank(r);
                    cur_error += (weights[p_sample][n_sample]+s2-s1)*L;
                    int k;
                    for(k = 0;k<embed_size;++k){
                        dA[k] += L * (B[p_sample * embed_size + k] - B[n_sample * embed_size + k]);
                        dB[p_sample * embed_size + k] += L * Ax[k];
                        dB[n_sample * embed_size + k] -= L * Ax[k];
                    }
                }
            }else if(ocsample == 1){ /* all negative sampling */
                result.pop_back();
                std::vector<int>::iterator it;
                for (it = result.begin();it<result.end(); ++it){
                    int n_sample = *it;
                    double s2= dot(Ax, B + n_sample * embed_size, embed_size);
                    double L = rank(r);
                    cur_error += (weights[p_sample][n_sample]+s2-s1)*L;
                    int k;
                    for(k = 0;k<embed_size;++k){
                        dA[k] += L * (B[p_sample * embed_size + k] - B[n_sample * embed_size + k]);
                        dB[p_sample * embed_size + k] += L * Ax[k];
                        dB[n_sample * embed_size + k] -= L * Ax[k];
                    }
                }
            }
        }
        
    }
    
    /* Update A and B , and normalize */
    for (i = 0; i<pt_number+nt_number; ++i) {
        for (j = 0; j<embed_size; ++j)
            Bg[i * embed_size + j] += lr * dB[i * embed_size + j];
        Bn[i] += 1;
    }
    for (i = 0; i<f_number; ++i) {
        int f = features[i];
        for (j = 0;j<embed_size;++j)
            Ag[f * embed_size + j] += lr * dA[j];
        An[f] += 1;
    }
    
    return cur_error;
}

double gradient_partial(double *dA, double *dB, double *Ax, int *features, int *positive_types, int *negative_types, int f_number, int pt_number, int nt_number){
    double cur_error = 0;
    int i,j;
    
    for (i = 0; i != embed_size; i++) dA[i] = 0;
    for (i = 0; i != pt_number + nt_number; i++) for (j = 0; j != embed_size; j++) dB[i * embed_size + j] = 0;
    for (i = 0; i != embed_size; i++) Ax[i]=0;
    
    for (i = 0; i<f_number; ++i) {
        int f = features[i];
        for (j = 0;j<embed_size;++j){
            Ax[j] += A[f * embed_size + j];
        }
    }

    int p_sample = max_score(Ax, B, positive_types, pt_number, embed_size);
    std::vector<int> result = get_rank(Ax, B, weights, p_sample, negative_types, nt_number, embed_size);
    double s1 = dot(Ax, B + p_sample * embed_size, embed_size);
    int r = (int)result.size()-1;
    if (r>0) { // find negative violations
        if (onsample == 0) { /* best negative sampling */
            int n_sample = result.back();
            double s2= dot(Ax, B + n_sample * embed_size, embed_size);
            double L = rank(r);
            cur_error += (weights[p_sample][n_sample]+s2-s1)*L;
            int k;
            for(k = 0;k<embed_size;++k){
                dA[k] += L * (B[p_sample * embed_size + k] - B[n_sample * embed_size + k]);
                dB[p_sample * embed_size + k] += L * Ax[k];
                dB[n_sample * embed_size + k] -= L * Ax[k];
            }
        }
        if(onsample == 1){ /* all negative sampling */
            std::vector<int>::iterator it;
            for (it = result.begin();it<result.end()-1; ++it){
                int n_sample = *it;
                double s2= dot(Ax, B + n_sample * embed_size, embed_size);
                double L = rank(r);
                cur_error += (weights[p_sample][n_sample]+s2-s1)*L;
                int k;
                for(k = 0;k<embed_size;++k){
                    dA[k] += L * (B[p_sample * embed_size + k] - B[n_sample * embed_size + k]);
                    dB[p_sample * embed_size + k] += L * Ax[k];
                    dB[n_sample * embed_size + k] -= L * Ax[k];
                }
            }
        }
    }

    /* Update A and B , and normalize */
    for (i = 0; i<pt_number+nt_number; ++i) {
        for (j = 0; j<embed_size; ++j)
            Bg[i * embed_size + j] += lr * dB[i * embed_size + j];
        Bn[i] += 1;
    }
    for (i = 0; i<f_number; ++i) {
        int f = features[i];
        for (j = 0;j<embed_size;++j)
            Ag[f * embed_size + j] += lr * dA[j];
        An[f] += 1;
    }
    return cur_error;
}

void update_embedding(){
    double len = 0;
    
    for (int k = 0; k != feature_count; k++){
        for (int c = 0; c != embed_size; c++)
            A[k * embed_size + c] += Ag[k * embed_size + c] / (An[k] + 1);
        len = 0;
        for (int c = 0; c != embed_size; c++)
            len += A[k * embed_size + c] * A[k * embed_size + c];
        if (len > C) for (int c = 0; c != embed_size; c++)
            A[k * embed_size + c] /= sqrt(len/C);
    }
    
    for (int k = 0; k != type_count; k++){
        for (int c = 0; c != embed_size; c++)
            B[k * embed_size + c] += Bg[k * embed_size + c] / (Bn[k] + 1);
        len = 0;
        for (int c = 0; c != embed_size; c++)
            len += B[k * embed_size + c] * B[k * embed_size + c];
        if (len > C) for (int c = 0; c != embed_size; c++)
            B[k * embed_size + c] /= sqrt(len/C);
    }
    
    for (int k = 0; k != feature_count; k++){
        for (int c = 0; c != embed_size; c++)
            Ag[k * embed_size + c] = 0;
        An[k] = 0;
    }
    
    for (int k = 0; k != type_count; k++){
        for (int c = 0; c != embed_size; c++)
            Bg[k * embed_size + c] = 0;
        Bn[k] = 0;
    }
}
