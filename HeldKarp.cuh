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

// Number of points
#define NUM_POINTS      10




/**---------------------------------------------------------------------------+
|                                                                             |
|                                   CLASSES                                   |
|                                                                             |
+----------------------------------------------------------------------------*/


class Point2D {
public:
    // Name of the point
    float   name;

    // Coordinates of the point
    float   x;
    float   y;


    // Member functions
    CUDA_CALLABLE Point2D() : name(0.), x(0.), y(0.) {};
    CUDA_CALLABLE Point2D(float x0, float y0) : name(0.), x(x0), y(y0) {};
    CUDA_CALLABLE ~Point2D() {};

    // Cartesian distance between two points
    CUDA_CALLABLE float distanceTo(Point2D point) {
        double dx = x - point.x;
        double dy = y - point.y;
        return sqrt(dx * dx + dy * dy);
    };
};


class Set {
public:
    // Values in the set
    int values[NUM_POINTS];
    // Number of values in the set
    int nValues;

    // Member functions
    CUDA_CALLABLE Set(int *setValues, int numVals) : nValues(numVals) {
        for (int i = 0; i < numVals; i++) {
            values[i] = setValues[i];
        }
    };
    CUDA_CALLABLE Set() {};
    CUDA_CALLABLE ~Set() {};

    // Determines whether two sets are equivalent
    CUDA_CALLABLE bool operator ==(const Set& otherSet) {
        //  Not the same if the two are different sizes
        if (nValues != otherSet.nValues)
            return false;
        // Otherwise, check that they have the same values
        Set *otherSetCopy = new Set(NULL, 0);
        memcpy(otherSetCopy, &otherSet, sizeof(Set));
        otherSetCopy->sort();
        for (int i = 0; i < nValues; i++)
            if (values[i] != otherSetCopy->values[i])
                return false;
        // All values were equal
        return true;
    };

    // Subtracts an element from the set
    CUDA_CALLABLE Set operator -(const int& toSub) {
        // Create new array of values for new set
        int *newValues = (int *) malloc((nValues - 1) * sizeof(int));
        assert(newValues);
        
        int index = 0;
        for (int i = 0; i < nValues && index < nValues; i++) {
            if (toSub != values[i]) {
                newValues[index] = values[i];
                index++;
            }
        }

        // Create new set and return it
        Set returnSet = Set(newValues, nValues - 1);
        free(newValues);
        return returnSet;
    };

    // Adds an element to the set
    CUDA_CALLABLE Set operator +(const int& toAdd) {
        // Create a new set from the old one, but with the new integer
        int *newValues = (int *) malloc((nValues + 1) * sizeof(int));
        for (int i = 0; i < nValues; i++) {
            newValues[i] = values[i];
        }
        newValues[nValues] = toAdd;

        // Return the new set
        Set returnSet = Set(newValues, nValues + 1);
        free (newValues);
        return returnSet;
    };

    // Allows accessing elements of values with just square brackets
    CUDA_CALLABLE int operator [](const int& i) const { return values[i]; };

    // Sorts the values
    CUDA_CALLABLE void sort() {
        for (int i = 0; i < nValues; i++) {
            for (int j = 1; j < nValues - i; j++) {
                if (values[j - 1] > values[j]) {
                    int temp = values[j];
                    values[j] = values[j - 1];
                    values[j - 1] = temp;
                }
            }
        }
    };
};


class HeldKarpMemo {
public:
    // Shortest distance from point 1 to this point
    float dist;

    // Previous point in shortest path from 1 to this point
    int prev;


    //Member functions
    CUDA_CALLABLE HeldKarpMemo() : dist(0.0) {};
    CUDA_CALLABLE ~HeldKarpMemo() {};
};


class HeldKarpMemoRow {
public:
    // Array of actual cells in the row
    HeldKarpMemo row[NUM_POINTS];


    // Member functions
    CUDA_CALLABLE HeldKarpMemoRow() {};
    CUDA_CALLABLE ~HeldKarpMemoRow() {};

    // Updates a value in the row
    CUDA_CALLABLE void updateRow(int col, float dist, int prev) {
        row[col].dist = dist;
        row[col].prev = prev;
    };

    // Allows us to call row[] with just the [] array notation
    CUDA_CALLABLE HeldKarpMemo operator [](const int& i) const { return row[i]; };
};

typedef HeldKarpMemoRow* HeldKarpMemoArray;






/**---------------------------------------------------------------------------+
|                                                                             |
|                             CUDA FUNCTION STUBS                             |
|                                                                             |
+----------------------------------------------------------------------------*/

void cudaCallGetDistances
    (int                nBlocks,
     int                threadsPerBlock,
     Point2D            *points,
     int                nPoints,
     float              *distances);

void cudaCallInitializeMemoArray
    (int                nBlocks,
     int                threadsPerBlock,
     HeldKarpMemoArray  memoArray,
     Point2D            *points,
     int                nPoints,
     float              *distances);
     
void cudaCallHeldKarpKernel
    (int nBlocks,
     int threadsPerBlock,
     Set set,
     HeldKarpMemoArray memoArray,
     float *distances,
     int nPoints,
     HeldKarpMemo *mins);





#endif // ifndef HELDKARP
