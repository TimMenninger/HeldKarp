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
|                                POINT2D CLASS                                |
|                                                                             |
+----------------------------------------------------------------------------*/

//Constructors
Point2D::Point2D() : x(0.0), y(0.0) {
    name = 0;
}
Point2D::Point2D(float x0, float y0) : x(x0), y(y0) {
    name = 0;
}

// Destructor
Point2D::~Point2D() {}

// Returns the Cartesian distance between two points
float Point2D::distanceTo(Point2D point) {
    double dx = x - point.x;
    double dy = y - point.y;
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
    Set *otherSetCopy = new Set(NULL, 0);
    memcpy(otherSetCopy, &otherSet, sizeof(Set));
    otherSetCopy->sort();
    for (int i = 0; i < nValues; i++)
        if (values[i] != otherSetCopy->values[i])
            return false;
    // All values were equal
    return true;
}

// Subtracts a value from the set
Set Set::operator -(const int& toSub) {
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


// Adds a value to the set
Set Set::operator +(const int& toAdd) {
    // Create a new set from the old one, but with the new integer
    int *newValues = (int *) malloc((nValues + 1) * sizeof(int));
    for (int i = 0; i < nValues; i++) {
        newValues[i] = values[i];
    }
    newValues[nValues] = toAdd;
    
    // Return the new set
    return Set(newValues, nValues + 1);
}


// Be able to access values in the set with square brackets
int Set::operator [](const int& i) const { return values[i]; }
    

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
|                           HELDKARP MEMO ROW CLASS                           |
|                                                                             |
+----------------------------------------------------------------------------*/

HeldKarpMemoRow::HeldKarpMemoRow(HeldKarpMemo *initRow) : row(initRow) {}

HeldKarpMemoRow::~HeldKarpMemoRow() {}

void HeldKarpMemoRow::updateRow(int col, float dist, int prev) {
    row[col].dist = dist;
    row[col].prev = prev;
}
    
HeldKarpMemo HeldKarpMemoRow::operator [](const int& i) const { return row[i]; }



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
        
        for (int i = 0; i < set.nValues; i++) {
            printf("%d ", set[i]);
        }
        printf("\n");

		
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



/**---------------------------------------------------------------------------+
|                                                                             |
|                               IMPLEMENTATION                                |
|                                                                             |
+----------------------------------------------------------------------------*/

int main(int argc, char *argv[]) {
    //TA_Utilities::select_least_utilized_GPU();
    int max_time_allowed_in_seconds = 3000;
    //TA_Utilities::enforce_time_limit(max_time_allowed_in_seconds);
    
    cudaEvent_t start;
    cudaEvent_t stop;
    
    
    /********************************Read Points******************************/
    
#if 1
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
	int totalPoints;
	dataFile >> totalPoints;
    // Array of all points in list
    Point2D *allPoints = NULL;
    
    
    //Point2D *allPoints = (Point2D *)malloc(totalPoints * sizeof(Point2D));
		
	while(1) {
	//for(int i = 0; i < totalPoints ; i++) {
		dataFile >> name >> x_val >> y_val;
		
		Point2D nextPoint(x_val, y_val);
		nextPoint.name = name;
		Point2D *temp = (Point2D *)realloc(allPoints, (numPoints + 1) * sizeof(Point2D));
		allPoints = temp;
		allPoints[numPoints] = nextPoint;
		
		//printf("   current point :(%f, %f)", allPoints[numPoints].x, allPoints[numPoints].y);	
		
		numPoints++;
		
		//printf("     numpoints: %d   name: %f    x_val: %f    y_val: %f\n",  numPoints, name, x_val, y_val);
		
		if (numPoints == totalPoints)
            break;
	}
		dataFile.close();
#else
    // Can use this for debugging when we don't have files
    int numPoints = 5;
    Point2D allPoints[5] = { Point2D(0.0, 0.0), Point2D(2.0, 2.0), Point2D(4.0, 4.0), Point2D(1.0, 1.0), Point2D(3.0, 3.0) };
#endif
	printf("Values: \n");
	for (int i = 0; i < numPoints; i++) {
		printf("Point%d (%.3f, %.3f) \n", i, allPoints[i].x, allPoints[i].y);
	}
	printf("\n");
    
    
    /****************************CPU Implementation***************************/
    
    float cpu_ms = -1;
    //START_TIMER();
    
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
    for (int i = 0; i < numPoints; i++) {
        for (int j = 0; j < numPoints; j++) {
            printf("%.3f ", allDistances[i][j]);
        }
        printf("\n");
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
    memset(memoArray, 0, numSubsets * sizeof(HeldKarpMemoRow));
    for (int i = 0; i < numSubsets; i++) {
        memoArray[i].row = (HeldKarpMemo *) calloc(numPoints, sizeof(HeldKarpMemo));
	}
	
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
		
        
        
        
        
    printf("\n\n");
    for (int i = 0; i < numSubsets; i++) {
        for (int j = 0; j < numPoints; j++) {
            printf("%d ", memoArray[i][j].prev);
        }
        printf("\n");
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
    path[numPoints] = 0; // Always end at the source
	float distance = (unsigned int) -1;
    
    // Find the last point in the minimum distance path
    for (int j = 1; j < numPoints; j++) {
        currdist = memoArray[fullSetIndex][j].dist;
        // Update only if we found a shorter distance
        if (currdist < distance) {
            path[1] = j;
            next = memoArray[fullSetIndex][j].prev;
            distance = currdist;
        }
    }
    
    // Follow the trail of prev indices to get the rest of the path
    for (int i = 2; i < numPoints; i++) {
        fullSet = fullSet - path[i - 1];
        path[i] = next;
        next = memoArray[getSetIndex(fullSet, numPoints)][path[i]].prev;
    }
    
    // The distance is the first iteration distance, which counts all points,
    // plus the distance back to the source
    distance += allDistances[0][path[numPoints - 1]];
	
	/* Results */
	printf("Final Path: ");
	for (int i = 0; i< numPoints + 1; i++) 
		printf("%d ", path[i]);
	printf("\nFinal Path Length: %.3f", distance);
	printf("\n\n");
	
     
    // Free all allocated memory
    for (int i = 0; i < numPoints; i++) {
         delete allDistances[i];
    }
    delete allDistances;
    delete memoArray;
 
    
    //STOP_RECORD_TIMER(cpu_ms);
    //printf("CPU runtime: %.3f seconds\n", cpu_ms / 1000);

    
    /****************************GPU Implementation***************************/
    
    float gpu_ms = -1;
    //START_TIMER();
    
    
    //STOP_RECORD_TIMER(gpu_ms);
    //printf("GPU runtime: %.3f seconds\n", gpu_ms / 1000);
    //printf("GPU took %d%% of the time the CPU did.\n", 
    //        (int) (gpu_ms / cpu_ms * 100));
    
    
    
    return 0;
}
