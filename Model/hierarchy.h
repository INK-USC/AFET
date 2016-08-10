

#ifndef hierarchy_h
#define hierarchy_h


#include <stdio.h>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <iostream>

class Hierarchy{
public:
    Hierarchy(char *filename);
    bool isclean(int *types, int count);
    std::vector<int> getpath(int label);
    std::set<int> getfinetype(int* labels, int count);
    double edit_distance(int label1, int label2, double** W);
private:
    std::map<int, int> supertype;
};

#endif /* hierarchy_h */
