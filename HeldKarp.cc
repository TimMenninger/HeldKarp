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
#include <setjmp.h>

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
        return;
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
                
        checkCUDAKernelError();
        return;

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
    
    // Returned by main.  Set to nonzero if error occurs.
    int status = 0;
    
    // Instantiate what we're using.  Will be checked for NULL before freeing
    // in the case of an error
    float               **allDistances      = NULL;
    Point2D             *allPoints          = NULL;
    HeldKarpMemoArray   memoArray           = NULL;
    int                 *path               = NULL;
    Point2D             *dev_allPoints      = NULL;
    HeldKarpMemoArray   dev_memoArray       = NULL;
    float               *host_allDistances  = NULL;
    HeldKarpMemo        *dev_mins           = NULL;
    float               *dev_allDistances   = NULL;
    
    // Other variables used so we can use goto function
    float cpu_ms, gpu_ms, distance, currdist;
    int numSubsets, fullSetList[NUM_POINTS], fullSetIndex, next;
    Set fullSet;
    unsigned int threadsPerBlock, maxBlocks, nBlocks;
    ofstream outputFile;


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
    allPoints = (Point2D *)malloc((NUM_POINTS) * sizeof(Point2D));

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

    cpu_ms = -1;
    START_TIMER();

    /*! Apply Held-Karp algorithm.  For this part, we referred to Wikipedia's
     *     page on Held-Karp to help us learn the algorithm.
     */

    // Create a numPoints square array of distances between each other, and
    //     initialize it to zero
    allDistances = (float **) malloc(numPoints * sizeof(float *));
    if (!allDistances) {
        fprintf(stderr, "Failed to allocate %lu bytes for allDistances.\n",
            numPoints * sizeof(float *));
        status = 1;
        goto free_memory;
    }
    for (int i = 0; i < numPoints; i++) {
        allDistances[i] = (float *) calloc(numPoints, sizeof(float));
        if (!allDistances[i]) {
            fprintf(stderr, "Failed to allocate %lu bytes for allDistances[%d].\n",
                numPoints * sizeof(float), i);
            status = 1;
            goto free_memory;
        }
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
    numSubsets = pow(2, numPoints - 1) - 1;
    memoArray = (HeldKarpMemoArray) malloc(numSubsets * sizeof(HeldKarpMemoRow));
    if (!memoArray) {
        fprintf(stderr, "Failed to allocate %lu bytes for memoArray.\n", 
            numSubsets * sizeof(HeldKarpMemoRow));
        status = 1;
        goto free_memory;
    }
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

    for (int k = 0; k < NUM_POINTS; k++) {
        fullSetList[k] = k;
    }

    // Define some variables that we will use to reconstruct the path
    fullSet = Set(fullSetList, numPoints);
    fullSetIndex = getSetIndex(fullSet, numPoints);
    next = 0;
    path = (int *) malloc((numPoints + 1) * sizeof(int));
    if (!path) {
        fprintf(stderr, "Failed to allocate %lu bytes for path.\n",
            (numPoints + 1) * sizeof(int));
        status = 1;
        goto free_memory;
    }
    path[0] = 0; // First point is always the source
    distance = (unsigned int) -1;

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

	outputFile.open ("out.txt");
	
    /* Results */
    printf("CPU Final Path: ");
    for (int i = 0; i < numPoints; i++) {
        printf("%d ", path[i]);
		outputFile <<  path[i];
		outputFile << " ";
		outputFile <<  path[i + 1];
		outputFile << "\n";
	}
    printf("%d", path[0]);

	
    printf("\nCPU Final Path Length: %.3f", distance);
    printf("\n\n");

	outputFile.close();



    /****************************GPU Implementation***************************/

    gpu_ms = -1;
    START_TIMER();


    /*=========================== ALLOCATE MEMORY ============================*/

    // Use command line arguments to define how many blocks and threads the GPU
    // will use in its kernel
    threadsPerBlock = atoi(argv[2]);
    maxBlocks = atoi(argv[3]);
    
    // Copy the list of points
    if (cudaSuccess != cudaMalloc((void **) &dev_allPoints, 
        numPoints * sizeof(Point2D))) {
            fprintf(stderr, "Failed to allocate %lu bytes for dev_allPoints.\n",
                numPoints * sizeof(Point2D));
            status = 1;
            goto free_memory;
    }
    if (cudaSuccess != cudaMemcpy(dev_allPoints, allPoints, 
        numPoints * sizeof(Point2D), cudaMemcpyHostToDevice)) {
            fprintf(stderr, "Failed to copy %lu bytes from host to dev_allPoints.\n",
                numPoints * sizeof(Point2D));
            status = 1;
            goto free_memory;
    }

    // Create space for a list of distances between any two points
    if (cudaSuccess != cudaMalloc((void **) &dev_allDistances, 
        numPoints * numPoints * sizeof(float))) {
            fprintf(stderr, "Failed to allocate %lu bytes for dev_allDistances.\n",
                numPoints * numPoints * sizeof(float));
            status = 1;
            goto free_memory;
    }

    if (cudaSuccess != cudaMalloc((void **) &dev_mins,
        numPoints * sizeof(HeldKarpMemo))) {
            fprintf(stderr, "Failed to allocate %lu bytes for dev_mins.\n",
                numPoints * sizeof(HeldKarpMemo));
            status = 1;
            goto free_memory;
    }

    // Create space for the memoization array
    if (cudaSuccess != cudaMalloc((void **) &dev_memoArray, 
        numSubsets * sizeof(HeldKarpMemoRow))) {
            fprintf(stderr, "Failed to allocate %lu bytes for dev_memoArray.\n",
                numSubsets * sizeof(HeldKarpMemoRow));
            status = 1;
            goto free_memory;
    }



    /*============================ GET DISTANCES ============================*/

    // Get distances kernel will run on a square matrix with side numPoints
    nBlocks = min(maxBlocks, (unsigned int) ceil((numPoints * numPoints) / float(threadsPerBlock)));

    // Fill in the distances array
    cudaCallGetDistances(nBlocks, threadsPerBlock, dev_allPoints, numPoints, dev_allDistances);
    
    host_allDistances = (float *) malloc(numPoints * numPoints * sizeof(float));
    if (cudaSuccess != cudaMemcpy(host_allDistances, dev_allDistances,
        numPoints * numPoints * sizeof(float), cudaMemcpyDeviceToHost)) {
            fprintf(stderr, "Failed to copy %lu bytes from device to host_allDistances.\n",
                numPoints * numPoints * sizeof(float));
            status = 1;
            goto free_memory;
    }

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
    
    
    if (cudaSuccess != cudaMemcpy(memoArray, dev_memoArray, 
        numSubsets * sizeof(HeldKarpMemoRow), cudaMemcpyDeviceToHost)) {
            fprintf(stderr, "Failed to copy %lu bytes from device to memoArray.\n",
                numSubsets * sizeof(HeldKarpMemoRow));
            status = 1;
            goto free_memory;
    }


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
    fullSet = Set(fullSetList, numPoints);
    for (int i = 2; i < numPoints; i++) {
        //fullSet = fullSet - path[i - 1];
        next = memoArray[getSetIndex(fullSet, numPoints)][path[i]].prev;
        path[i + 1] = next;
    }

    /*============================== FREE MEMORY ============================*/


    /* Results */
    printf("GPU Final Path: ");
    for (int i = 0; i < NUM_POINTS + 1; i++)
        printf("%d ", path[i]);
    printf("\nGPU Final Path Length: %.3f", distance);
    printf("\n\n");
    
    
    STOP_RECORD_TIMER(gpu_ms);
    
    //printf("\n");
    printf("CPU runtime: %.3f seconds\n", cpu_ms / 1000);
    printf("GPU runtime: %.3f seconds\n", gpu_ms / 1000);
    printf("GPU took %d%% of the time the CPU did.\n",
            (int) (gpu_ms / cpu_ms * 100));


// Puttin a label here so when there is an error, we can jump here and free
// everything before exiting.
free_memory:
    // Free all allocated memory
    if (allDistances) {
        for (int i = 0; i < numPoints; i++) {
             free(allDistances[i]);
        }
        free(allDistances);
    }
    
    if (memoArray)          free(memoArray);
    if (host_allDistances)  free(host_allDistances);
    if (allPoints)          free(allPoints);
    if (path)               free(path);
    
    if (dev_allPoints)      cudaFree(dev_allPoints);
    if (dev_allDistances)   cudaFree(dev_allDistances);
    if (dev_memoArray)      cudaFree(dev_memoArray);
    if (dev_mins)           cudaFree(dev_mins);

    printf("Main loop returning with");
    printf(status == 0 ? " no " : " ");
    printf("error(s).\n");

    return status;
}


