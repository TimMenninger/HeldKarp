#ifndef HELDKARP
#define HELDKARP

/**---------------------------------------------------------------------------+
|                                                                             |
| HeldKarp.cuh                                                                |
|                                                                             |
| Description:  This is the header file for the HeldKarp project, which       |
|               applies the Held-Karp algorithm to solve the traveling        |
|               salesman problem.                                             |
|                                                                             |
| Authors:      Tim Menninger                                                 |
|               Jared Reed                                                    |
|                                                                             |
| Revisions:    05/14/16 - Tim Menninger: Created                             |
|                                                                             |
+----------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "ta_utilities.hpp"

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif // ifdef __CUDACC__


// Maximum characters allowed in a name.
#define NAME_LEN        12


class Point2D {
public:
    // Name of the point
    float   name;
    
    // Coordinates of the point
    float   x;
    float   y;
    

    // Member functions
    CUDA_CALLABLE Point2D();
    CUDA_CALLABLE Point2D(float x0, float y0);
    CUDA_CALLABLE ~Point2D();
    
    // Cartesian distance between two points
    CUDA_CALLABLE float distanceTo(Point2D point);
};


class Set {
public:
    // Values in the set
    int *values;
    // Number of values in the set
    int nValues;
    
    // Member functions
    CUDA_CALLABLE Set(int *setValues, int numVals);
    CUDA_CALLABLE ~Set();
    
    // Determines whether two sets are equivalent
    CUDA_CALLABLE bool operator ==(const Set& otherSet);
    
    // Subtracts an element from the set
    CUDA_CALLABLE Set operator -(const int& toSub);
    
    // Sorts the values in nlog(n) time
    CUDA_CALLABLE void quickSort(int lowIndex, int highIndex);
    
    // Sorts the values in nlog(n) time
    CUDA_CALLABLE void sort();
};


class HeldKarpMemo {
public:
    // Shortest distance from point 1 to this point
    float   dist;
    
    // The second to last point index in the shortest-known path
    int     prev;
    
    //Member functions
    CUDA_CALLABLE HeldKarpMemo() : dist(0.0), prev(0) {};
    CUDA_CALLABLE ~HeldKarpMemo() {};
};


class HeldKarpMemoRow {
    // Sorted set of points the row represents
    Set subset;
public:
    // Array of actual cells in the row
    HeldKarpMemo *row;
    
    
    // Member functions
    CUDA_CALLABLE HeldKarpMemoRow(Set set, HeldKarpMemo *initRow);
    CUDA_CALLABLE ~HeldKarpMemoRow();
    
    // Updates a value in the row
    CUDA_CALLABLE void updateRow(int col, float dist, int prev);
};


#endif // ifndef HELDKARP
