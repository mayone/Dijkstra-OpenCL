#include<stdio.h>
#include<stdlib.h>

int main(int argc, char **argv)
{
	if(argc != 2)
	{
		fprintf(stderr, "usage: %s <num of vertex>\n", argv[0]);
		exit(EXIT_FAILURE);
	}
	int c = 1;
	int n = atoi(argv[1]);
	int i, j, k;
	printf("%d\n", c);
	for(i = 0; i < c; i++)
	{
		printf("%d %d\n", n, n*(n-1)/2);
		for(j = 0; j < n; j++)
			for(k = 0; k < j; k++)
				printf("%d %d 10\n", j, k);
	}
	
	return 0;
}
