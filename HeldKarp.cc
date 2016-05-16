/**---------------------------------------------------------------------------+
|                                                                             |
| HeldKarp.cc                                                                 |
|                                                                             |
| Description:  This contains the main loop which runs the Held-Karp algorithm|
|               to solve the traveling salesman problem.  It will do this     |
|               twice: once on the CPU and once on a GPU.  It records the time|
|               for each to demonstrate the GPU speedup.                      |
|                                                                             |
| Inputs:       argv[1]: number of blocks to use                              |
|               argv[2]: threads per block                                    |
|               argv[3]: file of (x, y) coordinates and names for each        |
|                                                                             |
| Outputs:      A file containing the sequence of points that produce the     |
|               shortest path, the distance to take that path, and the time   |
|               taken on the CPU and GPU.                                     |
|                                                                             |
| Authors:      Tim Menninger                                                 |
|               Jared Reed                                                    |
|                                                                             |
| Revisions:    05/14/16 - Tim Menninger: First attempt at CPU implementation |
|                   of Held-Karp algorithm                                    |
|                                                                             |
+----------------------------------------------------------------------------*/





#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <assert.h>

#include <cuda.h>

#include "CudaGillespie.cuh"
#include "ta_utilities.hpp"




#define START_TIMER() {                                                    \
        gpuErrChk(cudaEventCreate(&start));                                \
        gpuErrChk(cudaEventCreate(&stop));                                 \
        gpuErrChk(cudaEventRecord(start));                                 \
    }

#define STOP_RECORD_TIMER(name) {                                          \
        gpuErrChk(cudaEventRecord(stop));                                  \
        gpuErrChk(cudaEventSynchronize(stop));                             \
        gpuErrChk(cudaEventElapsedTime(&name, start, stop));               \
        gpuErrChk(cudaEventDestroy(start));                                \
        gpuErrChk(cudaEventDestroy(stop));                                 \
    }
    

    
    
    
/**---------------------------------------------------------------------------+
|                                                                             |
|                                POINT2D CLASS                                |
|                                                                             |
+----------------------------------------------------------------------------*/

//Constructors
Point2D::Point2D() : x(0.0) y(0.0) {
    memset(name, 0, NAME_LEN * sizeof(char)
}
Point2D::Point2D(float x0, float y0) : x(x0), y(y0) {
    memset(name, 0, NAME_LEN * sizeof(char)
}

// Destructor
Point2D::~Point2D() {}

// Returns the Cartesian distance between two points
float Point2D::distanceTo(Point2D point) {
    float dx = x - point.x, dy = y - point.y;
    return sqrt(dx * dx + dy * dy);
}



/**---------------------------------------------------------------------------+
|                                                                             |
|                                  SET CLASS                                  |
|                                                                             |
+----------------------------------------------------------------------------*/

Set::Set(int *setValues, int numVals) : values(setValues), nValues(numVals) {}
Set::~Set() {}

// Determines whether two sets are equivalent
bool Set::operator ==(const Set& otherSet) {
    //  Not the same if the two are different sizes
    if (nValues != otherSet.nValues)
        return false;
    // Otherwise, check that they have the same values
    otherSet.sort();
    for (int i = 0; i < nValues; i++)
        if (values[i] != otherSet.values[i])
            return false;
    // All values were equal
    return true;
}

// Subtracts a value from the set
Set Set::operator -(int toSub) {
    // Create new array of values for new set
    int *newValues = (int *) malloc((nValues - 1) * sizeof(int));
    int index = 0;
    for (int i = 0; i < nValues; i++) {
        if (toSub != values[i]) {
            newValues[index] = values[i];
            index++;
        }
    }
    
    // Create new set and return it
    return Set(newValues, nValues - 1);
}
    

// Bubblesorts in place.  O(n^2) but that's fast compared to TSP
void Set::sort() {
    for (int i = 0; i < nValues; i++) {
        for (int j = 1; j < nValues - i; j++) {
            if (values[j - 1] > values[j]) {
                int temp = values[j];
                values[j] = values[j - 1];
                values[j - 1] = temp;
            }
        }
    }
}



/**---------------------------------------------------------------------------+
|                                                                             |
|                             HELPER FUNCTIONS                                |
|                                                                             |
+----------------------------------------------------------------------------*/

/* Check errors on CUDA kernel calls */
void checkCUDAKernelError()
{
    cudaError_t err = cudaGetLastError();
    if  (cudaSuccess != err){
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
    }
}


/* Returns the index of the subset in the memo array. 
 * 
 * Parameters:
 *      set - the set of numbers to get an index for; the source point should
 *            be 0
 *      size - the number of points in the whole TSP algorithm
 */
int getSetIndex(Set *set, int size) {
    /*!
     * Unfortunately it is hard to explain how this arithmetic works.
     * Basically, we know that in a set S with |S| = n, there are 2^n
     * subsets.  Thus, there are 2^(n - 1) subsets whose first number is
     * fixed.  However, if we put the constraint that it is only considered
     * a subset if it is sorted, then we find that there are always
     * 2 ^ (n - m) subsets when we fix the first two digits (remember the
     * subsets are not repeated, so these first two digits are smaller than
     * any other value in the set).  This was found purely by looking for
     * a pattern.  Using this, we can find a unique index for any subset in
     * O(log(n)) time.
     */
     
    // Sort the list so we can find its index.
    set->sort();
    
    // We will continually add to the returned index
    int memoIndex = 0;
    
    // Remember the lowest value we havent seen.  We start at 1 because the
    //    smallest subset that makes sense in this problem has two elements.
    //    Thus, every set must have at least one value 1 or greater.
    lowest = 1;
    
    // This is the index in the set we are currently iterating over.  We start
    //    at 1 because the first element will always be the same (because we
    //    have a fixed first point in the problem)
    int setIndex = 1;
    
    while (1) {
        // Add in values for every subset of this subset we skip over
        for (; lowest < set->values[setIndex]; lowest++)
            memoIndex += pow(2, size - i - 1);
        
        // Increment the lowest value so that we don't double-check it.
        lowest++;
        setIndex++;
        
        // Break if we have seen every index
        if (set->nValues == setIndex)
            return memoIndex;
            
        // Increment the memo index because of a zero case that occurs if the
        //    next iteration is what was guessed.
        memoIndex++;
    }
}
        



/**---------------------------------------------------------------------------+
|                                                                             |
|                               IMPLEMENTATION                                |
|                                                                             |
+----------------------------------------------------------------------------*/

int main(int argc, char *argv[]) {
    TA_Utilities::select_least_utilized_GPU();
    int max_time_allowed_in_seconds = 300;
    TA_Utilities::enforce_time_limit(max_time_allowed_in_seconds);
    
    cudaEvent_t start;
    cudaEvent_t stop;
    
    
    /********************************Read Points******************************/
    

    FILE *dataFile = fopen(argv[1],"r");
    if (dataFile == NULL){
        fprintf(stderr, "Datapoints file missing\n");
        exit(EXIT_FAILURE);
    }
    
    // Counts how many points are processed
    int numPoints = 0;
    // Array of all points in list
    Point2D *allPoints = NULL;
    
    /* FOR line IN dataFile:
     *      Point2D nextPoint(line[1], line[2])
     *      nextPoint.name = line[0]
     *      allPoints = realloc((numPoints + 1) * sizeof(Point2D))
     *      allPoints[numPoints] = nextPoint
     *      numPoints++
     * END FOR
     */

    
    
    /****************************CPU Implementation***************************/
    
    float cpu_ms = -1;
    START_TIMER();
    
    /*! Apply Held-Karp algorithm.  For this part, we referred to Wikipedia's
     *     page on Held-Karp to help us learn the algorithm.
     */
    
    // Create a numPoints square array of distances between each other, and
    //     initialize it to zero
    float **allDistances = (float **) malloc(numPoints * sizeof(float *));
    for (int i = 0; i < numPoints; i++)
        allDistances[i] = (float *) calloc(numPoints, sizeof(float));
    
    // Find the distance between each set of two points.  For this, only find
    //     the upper triangle, and copy to the lower triangle
    for (int i = 0; i < numPoints; i++) {
        for (int j = numPoints - 1; j != i; j++) {
            // Get distance between point i and point j
            allDistances[i][j] = allPoints[i].distanceTo(allPoints[j]);
            // Distance is same in either direction
            allDistances[j][i] = allDistances[i][j];
        }
    }
    
    // We are creating a 2D array for our memoization where each column is
    // an endpoint and each row is a subset of all of the points whose
    // cardinality is greater than 2.  The value at each index [i][j] is 
    // two values: the first is the shortest distance from point 1 to point
    // j through the points in set i and the second is the index of the
    // previous point (before j) that created that shortest distance.
    // The number of distinct subsets with cardinality >= 2 is 2 ^ (numPoints
    // - 1) - 1
    int numSubsets = pow(2, numPoints - 1) - 1;
    HeldKarpMemoRow *memoArray = (HeldKarpMemoRow *) malloc(numSubsets * sizeof(HeldKarpMemoRow));
    memset(memoArray, 0, numSubsets * sizeof(HeldKarpMemoRow));
    
    
    // Initialize by setting all sets {0, n} to the distance from 1 to n.
    for (int i = 1; i < numPoints; i++) {
        int set[2] = {0, i};
        int index = getSetIndex(Set(set, 2), numPoints);
        memoArray[index].updateRow(i, allDistances[0][i], 0);
    }
    
    
    // Continue with rest of algorithm.
    for (int i = 3; i < numPoints; i++) {
        /*  for all subsets, S, of {1, 2, ..., numPoints} of size i:
         *      for each k in S:
         *          unsigned int minVal = -1
         *          int minValPrev = 0;
         *          HeldKarpMemo *memo = memoArray[getSetIndex(S - k)].row;
         *          for each m in S:
         *              if (1 == m || k == m):
         *                  continue
         *              minVal = min(minVal, memo[m].dist + allDistances[m][k]);
         *              minValPrev = m;
         *          memoArray[getSetIndex(S)].row[k].dist = minVal;
         *          memoArray[getSetIndex(S)].row[k].prev = minValPrev;
         *  Trace from dest to source using prev values
         */
    
    
    
    STOP_TIMER(&cpu_ms);
    printf("CPU runtime: %.3f\n", cpu_ms / 1000);
    
    
    
    /****************************GPU Implementation***************************/
    
    float gpu_ms = -1;
    START_TIMER();
    
    
    STOP_TIMER(&gpu_ms);
    printf("GPU runtime: %.3f\n", gpu_ms / 1000);
    printf("GPU speedup: %d%\n", (int) (gpu_ms / cpu_ms));
    
    
    
    return 0;
}
