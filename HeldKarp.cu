
#include <cmath>
#include "HeldKarp.cuh"




/* Returns the index of the subset in the memo array. 
 * 
 * Parameters:
 *      set - the set of numbers to get an index for; the source point should
 *            be 0
 *      size - the number of points in the whole TSP algorithm
 */
__device__
int cudaGetSetIndex(Set set, int size) {
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
            memoIndex += powf(2, size - lowest - 1);
        
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











__global__
void cudaGetDistances(Point2D *points, int nPoints, float *distances) {
    
    // Get the index of the thread so we only iterate part of the data. 
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Variables when filling in the distances array
    int row, col;
    
    
    while (tid < (nPoints * nPoints)) {
        // The row and column can be determined from mod and division
        row = tid / nPoints;
        col = tid % nPoints;
        
        // Get Euclidean distance and put it into the array.
        distances[tid] = points[row].distanceTo(points[col]);
        
        // Advance thread index.
        tid += blockDim.x * gridDim.x;
    }
    
}




void cudaCallGetDistances(int nBlocks,
                          int threadsPerBlock,
                          Point2D *points,
                          int nPoints,
                          float *distances) {

    // Number of bytes of shared memory
    int shmem = 0;
    
    // Fill in all of the distances between two points.
    cudaGetDistances<<<nBlocks, threadsPerBlock, shmem>>>(points, nPoints, distances);
                              
}









__global__
void cudaInitializeMemoArray(HeldKarpMemoArray memoArray,
                             Point2D *points,
                             int nPoints,
                             float *distances) {

    // Get the index of the thread so we only iterate part of the data. 
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // We don't care about the 0 to 0 case, so we will skip right away if
    // tid is 0.
    tid = (tid == 0 ? blockDim.x * gridDim.x : tid);
    
    while (tid < nPoints) {
        // Create a length two subset with the source as the first point
        int setPoints[2] = { 0, tid };
        
        // Memoize the "shortest distance" as the distance between these points.
        memoArray[cudaGetSetIndex(Set(setPoints, 2), nPoints)].updateRow(tid, distances[tid], 0);
        
        // Advance thread index.
        tid += blockDim.x * gridDim.x;
    }

}





void cudaCallInitializeMemoArray(int nBlocks,
                                 int threadsPerBlock,
                                 HeldKarpMemoArray memoArray,
                                 Point2D *points,
                                 int nPoints,
                                 float *distances) {

    // Number of bytes of shared memory
    int shmem = 0;
    
    // Initialize the memo array withs subsets of length 2
    cudaInitializeMemoArray<<<nBlocks, threadsPerBlock, shmem>>>
        (memoArray, points, nPoints, distances);

}
