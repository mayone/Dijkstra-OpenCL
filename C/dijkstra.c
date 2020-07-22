#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define MAX_SIZE 10000
#define INF 1e9
#define NIL -1

int adj[MAX_SIZE][MAX_SIZE];	// Adjacency list stored in matrix
int weight[MAX_SIZE][MAX_SIZE];
int degree[MAX_SIZE];			// degree of vertices
int parent[MAX_SIZE];
int distance[MAX_SIZE];
int visited[MAX_SIZE];

int main(int argc, char const *argv[])
{
	int num_cases;
	int n, m;		// number of vertices, number of edges
	int u, v, w;	// vertices and the weight of the edge between
	int src;
	int msec, total;
	struct timeval t_start, t_end;

	scanf("%d", &num_cases);

	for(int c = 0; c < num_cases; c++) {
		scanf("%d%d", &n, &m);

		// Initialize
		for(int i = 0; i < n; i++) {
			degree[i] = 0;
			distance[i] = INF;
			parent[i] = NIL;
			visited[i] = 0;
		}

		// Construct adjacencey list
		for(int i = 0; i < m; i++) {
			scanf("%d%d%d", &u, &v, &w);
			adj[u][degree[u]] = v;
			weight[u][degree[u]] = w;
			adj[v][degree[v]] = u;
			weight[v][degree[v]] = w;
			degree[u]++;
			degree[v]++;
		}

		gettimeofday(&t_start, NULL);

		// Set vertex 0 as source
		src = 0;
		distance[src] = 0;

		// Dijkstra
		for(int i = 0; i < n; i++) {
			// Extract min
			int min_distance = INF;
			int min_vertex;
			for(int j = 0; j < n; j++) {
				if (!visited[j] && distance[j] < min_distance) {
					min_distance = distance[j];
					min_vertex = j;
				}
			}
			// Relaxation
			for(int j = 0; j < degree[min_vertex]; j++) {
				if (distance[adj[min_vertex][j]] > distance[min_vertex] + weight[min_vertex][j]) {
					distance[adj[min_vertex][j]] = distance[min_vertex] + weight[min_vertex][j];
					parent[adj[min_vertex][j]] = min_vertex;
				}
			}
			visited[min_vertex] = 1;
		}

		gettimeofday(&t_end, NULL);
		msec = (t_end.tv_sec-t_start.tv_sec) * 1000;
		msec += (t_end.tv_usec-t_start.tv_usec) / 1000;
		total += msec;

		// Output
		printf("Vertex\t\tDistance from source (path)\n");
		for(int i = 0; i < n; i++) {
			printf("%d", i);
			if (i == src)
				printf(" (source)\t");
			else
				printf("\t\t");
			printf("%d", distance[i]);
			printf(" (");
			int p = i;
			do {
				if (p == i)
					printf("%d", p);
				else
					printf("-%d", p);
				p = parent[p];
			}
			while(p != NIL);
			printf(")\n");
		}
		printf("Elapsed time: %d ms\n", total);
		printf("======================================\n");
	}

	return 0;
}