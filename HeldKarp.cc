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
+----------------------------------------------------------------------------*/



#include <math.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include "HeldKarp.cuh"

using namespace std;



/**---------------------------------------------------------------------------+
|                                                                             |
|                                    MACROS                                   |
|                                                                             |
+----------------------------------------------------------------------------*/


/*
NOTE: You can use this macro to easily check cuda error codes
and get more information.

Modified from:
http://stackoverflow.com/questions/14038589/
what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
*/
#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(
    cudaError_t code,
    const char *file,
    int line,
    bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n",
            cudaGetErrorString(code), file, line);
        exit(code);
    }
}



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
int getSetIndex(Set set, int size) {
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
    set.sort();

    // We will continually add to the returned index
    int memoIndex = 0;

    // Remember the lowest value we havent seen.  We start at 1 because the
    //    smallest subset that makes sense in this problem has two elements.
    //    Thus, every set must have at least one value 1 or greater.
    int lowest = 1;

    // This is the index in the set we are currently iterating over.  We start
    //    at 1 because the first element will always be the same (because we
    //    have a fixed first point in the problem)
    int setIndex = 1;

    while (1) {
        // Add in values for every subset of this subset we skip over
        for (; lowest < set[setIndex]; lowest++)
            memoIndex += pow(2, size - lowest - 1);

        // Increment the lowest value so that we don't double-check it.
        lowest++;
        setIndex++;

        // Break if we have seen every index
        if (set.nValues == setIndex)
            return memoIndex;

        // Increment the memo index because of a zero case that occurs if the
        //    next iteration is what was guessed.
        memoIndex++;
    }
}

void setOfAllSubsets(Set set, int largestInSet, int largestPossibleInSet,
    HeldKarpMemoArray memoArray, float** allDistances, int curSize) {

    /* Return if set length is greater than currant because this is irrelvant
     * since the recursive calls only call on sets with length greater
     * than the current set */
    if (set.nValues > curSize) {
        return;
    }

    /* Only updating memoization array for lists of a given size */
    if (set.nValues == curSize) {

        /* For all subsets of the set minus one elements */
        for (int k = 0; k < set.nValues; k++) {
            if (set[k] ==  0)
                continue;

            float minVal = (unsigned int) -1;
            int minPrev = -1;
            Set newset = set - set[k];

            /* Iterate over this new subset */
            for (int m = 0; m < newset.nValues; m++) {
                if (newset[m] == 0)
                    continue;

                /* Calculate values to update memoization array */
                HeldKarpMemoRow memo = memoArray[getSetIndex(newset, largestPossibleInSet + 1)];
                if (minVal > memo[newset[m]].dist + allDistances[newset[m]][set[k]]) {
                    minVal = memo[newset[m]].dist + allDistances[newset[m]][set[k]];
                    minPrev = newset[m];
                }
            }
            /* Update memoization array */
            memoArray[getSetIndex(set, largestPossibleInSet + 1)].updateRow(set[k], minVal, minPrev);
        }
    }
    /* If we have reached largest set size then recursion has finished so break */
    if (largestInSet == largestPossibleInSet) {
        return;
    }

    /* Recursive call for all sets with a length of one more than current */
    for (int i = largestInSet + 1; i <= largestPossibleInSet; i++) {
        setOfAllSubsets(set + i, i, largestPossibleInSet, memoArray, allDistances, curSize);
    }
}


void cudaSetOfAllSubsets(Set set, int largestInSet, int largestPossibleInSet,
     int curSize, float *dev_allDistances, HeldKarpMemoArray dev_memoArray, 
     HeldKarpMemo *dev_mins, int nBlocks, int threadsPerBlock) {

    cudaMemset(dev_mins, 0, largestPossibleInSet + 1 * sizeof(HeldKarpMemo));
    
    
    /* Return if set length is greater than currant because this is irrelvant
     * since the recursive calls only call on sets with length greater
     * than the current set */
    if (set.nValues > curSize) {
        return;
    }

    /* Only updating memoization array for lists of a given size */
   
    if (set.nValues == curSize) {
        
        cudaCallHeldKarpKernel(nBlocks, threadsPerBlock, set, dev_memoArray, 
                dev_allDistances, largestPossibleInSet, dev_mins);              

    }
     

    /* If we have reached largest set size then recursion has finished so break */
    if (largestInSet == largestPossibleInSet) {
        return;
    }

    /* Recursive call for all sets with a length of one more than current */
    for (int i = largestInSet + 1; i <= largestPossibleInSet; i++) {
        cudaSetOfAllSubsets(set + i, i, largestPossibleInSet, curSize, 
            dev_allDistances, dev_memoArray, dev_mins, nBlocks, threadsPerBlock);
    }
    
}



/**---------------------------------------------------------------------------+
|                                                                             |
|                               IMPLEMENTATION                                |
|                                                                             |
+----------------------------------------------------------------------------*/

int main(int argc, char *argv[]) {
    TA_Utilities::select_least_utilized_GPU();
    int max_time_allowed_in_seconds = 3000;
    TA_Utilities::enforce_time_limit(max_time_allowed_in_seconds);

    cudaEvent_t start;
    cudaEvent_t stop;


    if (argc != 4) {
        fprintf(stderr, "usage: ./HeldKarp <dataPointFile> <threadsPerBlock> <maxBlocks>");
        exit(EXIT_FAILURE);
    }


    /********************************Read Points******************************/

    // Actual code
    ifstream dataFile;
    dataFile.open(argv[1]);
    if (dataFile == NULL){
        fprintf(stderr, "Datapoints file missing\n");
        exit(EXIT_FAILURE);
    }

    // Place to store name, x value, and y value which will be extracted
    // from the file
    float name;
    float x_val;
    float y_val;

    // Counts how many points are processed
    int numPoints = 0;
    // Array of all points in list
    Point2D *allPoints = (Point2D *)malloc((NUM_POINTS) * sizeof(Point2D));

    while(numPoints < NUM_POINTS) {
        dataFile >> name >> x_val >> y_val;

        Point2D nextPoint(x_val, y_val);
        nextPoint.name = name;
        allPoints[numPoints] = nextPoint;

        numPoints++;
    }
    dataFile.close();


    printf("Values: \n");
    for (int i = 0; i < numPoints; i++) {
        printf("Point%d (%.3f, %.3f) \n", i, allPoints[i].x, allPoints[i].y);
    }
    printf("\n");


    /****************************CPU Implementation***************************/

    float cpu_ms = -1;
    START_TIMER();

    /*! Apply Held-Karp algorithm.  For this part, we referred to Wikipedia's
     *     page on Held-Karp to help us learn the algorithm.
     */

    // Create a numPoints square array of distances between each other, and
    //     initialize it to zero
    float **allDistances = (float **) malloc(numPoints * sizeof(float *));
    for (int i = 0; i < numPoints; i++) {
        allDistances[i] = (float *) calloc(numPoints, sizeof(float));
    }
    // Find the distance between each set of two points.  For this, only find
    //     the upper triangle, and copy to the lower triangle
    for (int i = 0; i < numPoints; i++) {
        for (int j = numPoints - 1; j != i; j--) {
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
    HeldKarpMemoArray memoArray = (HeldKarpMemoArray) malloc(numSubsets * sizeof(HeldKarpMemoRow));
    assert (memoArray != NULL);
    memset(memoArray, 0, numSubsets * sizeof(HeldKarpMemoRow));

    // Initialize by setting all sets {0, n} to the distance from 1 to n.
    for (int i = 1; i < numPoints; i++) {
        int setIndices[2] = {0, i};
        int index = getSetIndex(Set(setIndices, 2), numPoints);
        memoArray[index].updateRow(i, allDistances[0][i], 0);
    }

    // Continue with rest of algorithm.
    for (int j = 3; j < numPoints + 1; j++) {
        for (int i = 1; i < numPoints; i++) {
            int setIndices[2] = {0, i};
            setOfAllSubsets(Set(setIndices, 2), i, numPoints - 1, memoArray, allDistances, j);
        }
    }

    int fullSetList[numPoints];
    for (int k = 0; k < numPoints; k++) {
        fullSetList[k] = k;
    }

    // Define some variables that we will use to reconstruct the path
    Set fullSet = Set(fullSetList, numPoints);
    int fullSetIndex = getSetIndex(fullSet, numPoints);
    float currdist;
    int next = 0;
    int *path = (int *) malloc(numPoints + 1 * sizeof(int));
    path[0] = 0; // First point is always the source
    float distance = (unsigned int) -1;

    // Find the last point in the minimum distance path
    for (int j = 1; j < numPoints; j++) {
        currdist = memoArray[fullSetIndex][j].dist;
        // Update only if we found a shorter distance
        if (currdist + allDistances[j][0] < distance) {
            next = memoArray[fullSetIndex][j].prev;
            path[1] = j;
            path[2] = next;
            distance = currdist + allDistances[j][0];
        }
    }

    // Follow the trail of prev indices to get the rest of the path
    for (int i = 2; i < numPoints; i++) {
        fullSet = fullSet - path[i - 1];
        next = memoArray[getSetIndex(fullSet, numPoints)][path[i]].prev;
        path[i + 1] = next;
    }
    
    STOP_RECORD_TIMER(cpu_ms);

    /* Results */
    printf("CPU Final Path: ");
    for (int i = 0; i< numPoints + 1; i++)
        printf("%d ", path[i]);
    printf("\nCPU Final Path Length: %.3f", distance);
    printf("\n\n");


    // Free all allocated memory
    for (int i = 0; i < numPoints; i++) {
         delete allDistances[i];
    }
    delete allDistances;
    delete memoArray;


    printf("CPU runtime: %.3f seconds\n\n\n", cpu_ms / 1000);


exit(0);
    /****************************GPU Implementation***************************/

    float gpu_ms = -1;
    START_TIMER();


    /*=========================== ALLOCATE MEMORY ============================*/

    // Use command line arguments to define how many blocks and threads the GPU
    // will use in its kernel
    const unsigned int threadsPerBlock = atoi(argv[2]);
    const unsigned int maxBlocks = atoi(argv[3]);

    // Define the number of blocks and threads per block that the GPU will use.
    // This will differ from kernel to kernel
    unsigned int nBlocks;
    
    // Copy the list of points
    Point2D *dev_allPoints;
    assert(cudaSuccess == cudaMalloc((void **) &dev_allPoints, 
                numPoints * sizeof(Point2D)));
    assert(cudaSuccess == cudaMemcpy(dev_allPoints, allPoints, 
                numPoints * sizeof(Point2D), cudaMemcpyHostToDevice));

    // Create space for a list of distances between any two points
    float *dev_allDistances;
    cudaMalloc((void **) &dev_allDistances, numPoints * numPoints * sizeof(float));

    HeldKarpMemo* dev_mins;
    cudaMalloc((void **) &dev_mins, numPoints * sizeof(HeldKarpMemo));

    // Create space for the memoization array
    HeldKarpMemoArray dev_memoArray;
    cudaMalloc((void **) &dev_memoArray, numSubsets * sizeof(HeldKarpMemoRow));




    /*============================ GET DISTANCES ============================*/

    // Get distances kernel will run on a square matrix with side numPoints
    nBlocks = min(maxBlocks, (unsigned int) ceil((numPoints * numPoints) / float(threadsPerBlock)));

    // Fill in the distances array
    cudaCallGetDistances(nBlocks, threadsPerBlock, dev_allPoints, numPoints, dev_allDistances);
    
    float *host_allDistances = (float *) malloc(numPoints * numPoints * sizeof(float));
    cudaMemcpy(host_allDistances, dev_allDistances, numPoints * numPoints * sizeof(float), \
                cudaMemcpyDeviceToHost);

    checkCUDAKernelError();


    /*=========================== INIT MEMO ARRAY ===========================*/

    // Initializing memo will run once for each point
    nBlocks = min(maxBlocks, (unsigned int) ceil(numPoints / float(threadsPerBlock)));

    // Fill in the memo array for al subsets of length 2
    cudaCallInitializeMemoArray(nBlocks, threadsPerBlock, dev_memoArray, dev_allPoints,
        numPoints, dev_allDistances);

    checkCUDAKernelError();

    /*========================== FILL MEMO ARRAY ============================*/
                
    for (int j = 3; j < numPoints + 1; j++) {
        for (int i = 1; i < numPoints; i++) {
            int setIndices[2] = {0, i};
            cudaSetOfAllSubsets(Set(setIndices, 2), i, numPoints - 1,
                j, dev_allDistances, dev_memoArray, dev_mins, nBlocks, threadsPerBlock);
        }
    }
    
    memoArray = (HeldKarpMemoArray) malloc(numSubsets * sizeof(HeldKarpMemoRow));
    cudaMemcpy(memoArray, dev_memoArray, numSubsets * sizeof(HeldKarpMemoRow),
                cudaMemcpyDeviceToHost);
                
    /*========================== FIND SHORTEST PATH =========================*/


    checkCUDAKernelError();
    
    distance = (unsigned int) -1;
    // Find the last point in the minimum distance path
    for (int j = 1; j < numPoints; j++) {
        currdist = memoArray[fullSetIndex][j].dist;
        // Update only if we found a shorter distance
        if (currdist + host_allDistances[j * numPoints + 0] < distance) {
            next = memoArray[fullSetIndex][j].prev;
            path[1] = j;
            path[2] = next;
            distance = currdist + host_allDistances[j * numPoints + 0];
        }
    }

    // Follow the trail of prev indices to get the rest of the path
    for (int i = 2; i < numPoints; i++) {
        fullSet = fullSet - path[i - 1];
        next = memoArray[getSetIndex(fullSet, numPoints)][path[i]].prev;
        path[i + 1] = next;
    }


    /*============================== FREE MEMORY ============================*/

    delete memoArray;
    delete host_allDistances;

    cudaFree(dev_allPoints);
    cudaFree(dev_allDistances);
    cudaFree(dev_memoArray);
    cudaFree(dev_mins);
    
    STOP_RECORD_TIMER(gpu_ms);


    /* Results */
    printf("GPU Final Path: ");
    for (int i = 0; i< numPoints + 1; i++)
        printf("%d ", path[i]);
    printf("\nGPU Final Path Length: %.3f", distance);
    printf("\n\n");
    
    
    printf("GPU runtime: %.3f seconds\n", gpu_ms / 1000);
    printf("GPU took %d%% of the time the CPU did.\n",
            (int) (gpu_ms / cpu_ms * 100));



    return 0;
}


