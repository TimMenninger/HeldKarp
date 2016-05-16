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

// Maximum characters allowed in a name.
#define	NAME_LEN		12


class Point2D {
	// Name of the point
	char	name[NAME_LEN];
	
    // Coordinates of the point
    float   x;
    float   y;
    
    
    // Member functions
    Point2D() : x(0.0), y(0.0) {
		memset(name, 0, NAME_LEN * sizeof(char)
	};
    Point2D(float x0, float y0) : x(x0), y(y0) {
		memset(name, 0, NAME_LEN * sizeof(char)
	};
    ~Point2D() {};
    
    // Cartesian distance between two points
    float distanceTo(Point2D point) {
		float dx = x - point.x, dy = y - point.y;
		return sqrt(dx * dx + dy * dy);
	};
};


class HeldKarpMemo {
	// Shortest distance from point 1 to this point
	float	dist;
	
	// The second to last point index in the shortest-known path
	int		prev;
	
	//Member functions
	HeldKarpMemo() : dist(0.0), prev(0) {};
	~HeldKarpMemo() {};
};


class Set {
	// Values in the set
	int *values;
	// Number of values in the set
	int nValues;
	
	
	// Member functions
	Set(int *setValues, int numVals);
	~Set();
	
	// Determines whether two sets are equivalent
	bool operator ==(const Set& otherSet);
	
	// Subtracts an element from the set
	Set operator -(int toSub);
	
	// Sorts the values in nlog(n) time
	void quickSort(int lowIndex, int highIndex);
	
	// Sorts the values in nlog(n) time
	void sort();
};


class HeldKarpMemoRow {
	// Sorted set of points the row represents
	Set subset;
	
	// Array of actual cells in the row
	HeldKarpMemo *row;
	
	// Next row in array
	HeldKarpMemoRow *next;
	
	
	// Member functions
	HeldKarpMemoRow(Set set, HeldKarpMemo *initRow) : subset(set), row(initRow) {};
	~HeldKarpMemoRow() {};
	
	// Updates a value in the row
	void updateRow(int col, float dist, int prev) {
		row[col].dist = dist;
		row[col].prev = prev;
		next = NULL;
	};
}


#endif // ifndef HELDKARP
