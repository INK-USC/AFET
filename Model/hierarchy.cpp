

#include "hierarchy.h"

Hierarchy::Hierarchy(char *filename){
    FILE *fi = fopen(filename, "rb");
    printf("Fininsh initialize matrix A and B\n");


    int mid, fid;
    while (fscanf(fi, "%d\t%d", &mid, &fid) == 2)
        this->supertype[mid] = fid;
    fclose(fi);
}

bool Hierarchy::isclean(int *types, int count) {
    std::set<int> new_labels = getfinetype(types, count);
    return new_labels.size()==1; // if the size of fine-grained type is 1, it's clean.
}

std::vector<int> Hierarchy::getpath(int label){
    std::vector<int> result;
    result.push_back(label);
    while (supertype.find(label)!=supertype.end()) {
        label = supertype[label];
        result.push_back(label);
    }
    std::reverse(result.begin(), result.end());
    return result;
}

std::set<int> Hierarchy::getfinetype(int* labels, int count) {
    std::set<int> prefix;
    std::set<int> new_labels;
    for(int i =0; i<count; ++i) {
        int l = labels[i];
        if(prefix.find(l)!=prefix.end()) {
            continue;
        }
        std::vector<int> path = getpath(l);
        std::vector<int>::iterator it;
        for (it=path.begin(); it!=path.end()-1; ++it){
            std::set<int>::iterator index = new_labels.find(*it);
            if(index!=new_labels.end()){
                new_labels.erase(index);       
            }
            prefix.insert(*it);
        }
        new_labels.insert(l);
    }
    return new_labels;
}

double Hierarchy::edit_distance(int label1, int label2, double** W){
    std::vector<int> path1 = getpath(label1);
    std::vector<int> path2 = getpath(label2);
    int m = (int)(path1.size());
    int n = (int)(path2.size());
    double d[m][n];
    if (path1[0]!=path2[0]) {
        d[0][0] = W[path1[0]][path2[0]];
    }
    for(int i = 1; i<m; ++i){
        d[i][0] = d[i-1][0] + W[path1[i-1]][path1[i]];
    }
    for(int j = 1; j<n; ++j){
        d[0][j] = d[0][j-1] + W[path2[j-1]][path2[j]];
    }
    for (int i = 1; i<m; ++i) {
        for(int j = 1; j<n; ++j){
            if (path1[i] == path2[j]){
                d[i][j] = d[i-1][j-1];
            }
            else{
                double temp1 = d[i-1][j]+W[path1[i-1]][path1[i]];
                double temp2 = d[i][j-1]+W[path2[j-1]][path2[j]];
                double temp3 = d[i-1][j-1]+W[path1[i-1]][path2[j-1]];
                d[i][j] = std::min(std::min(temp1, temp2), temp3);
            }
        }
    }
    return d[m-1][n-1];
}


