#include <stdio.h>
#include <sys/time.h>
#if defined(__APPLE__)
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SIZE 10000
#define INF 1e9

int max_size = MAX_SIZE;
int adj[MAX_SIZE][MAX_SIZE];
int weight[MAX_SIZE][MAX_SIZE];
int degree[MAX_SIZE];
int distance[MAX_SIZE];
int visited[MAX_SIZE];

cl_program load_program(cl_context context, const char* filename);

int main(int argc, char const *argv[])
{
	int c, num_cases;
	int n, m;		// number of vertices, number of edges
	int u, v, w;	// vertices and the weight of the edge between
	int i, j, k;
	int src;
	int msec = 0, total = 0;
	struct timeval t_start, t_end;

	// OpenCL
	cl_platform_id *platforms;
	cl_context context;
	cl_uint num_devices;
	cl_device_id *devices;
	cl_command_queue queue;
	cl_program program;
	cl_kernel init, reduce, extractMin, relax;
	char *devName;
	char *devVer;
	size_t cb;
	cl_int err;
	cl_uint num;
	size_t actual_work_size;
	size_t global_work_size;
	size_t local_work_size = 256;
	size_t reduce_work_size = 1024;
	size_t num_groups = reduce_work_size/local_work_size;
	
	// get the id of supporting OpenCL platforms
	err = clGetPlatformIDs(0, 0, &num);
	if(err != CL_SUCCESS)
	{
		perror("Unable to get platforms");
		return 0;
	}
	platforms = (cl_platform_id*) malloc(num * sizeof(cl_platform_id));
	clGetPlatformIDs(num, &platforms[0], NULL);
	printf("There are %d platform(s) on this device\n", num);
	
	// create a OpenCL context
	cl_context_properties prop[] = { CL_CONTEXT_PLATFORM, (cl_context_properties) platforms[0], 0 };
	context = clCreateContextFromType(prop, CL_DEVICE_TYPE_DEFAULT, NULL, NULL, NULL);
	if(context == 0)
	{
		perror("Can't create OpenCL context");
		return 0;
	}

	// get a list of devices
	clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &cb);
	devices = (cl_device_id*) malloc(cb / sizeof(cl_device_id)); 
	clGetContextInfo(context, CL_CONTEXT_DEVICES, cb, &devices[0], 0);

	// get the number of devices
	clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES, 0, NULL, &cb);
	clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES, cb, &num_devices, 0);
	printf("There are %d device(s) in the context\n", num_devices);

	// show devices info
	for(i = 0; i < num_devices; i++)
	{
		// get device name
		clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 0, NULL, &cb);
		devName = (char*) malloc(sizeof(char) * cb);
		clGetDeviceInfo(devices[i], CL_DEVICE_NAME, cb, &devName[0], NULL);
		devName[cb] = 0;
		printf("Device: %s", devName);
		free(devName);
		
		// get device supports version
		clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, 0, NULL, &cb);
		devVer = (char*) malloc(sizeof(char) * cb);
		clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, cb, &devVer[0], NULL);
		devVer[cb] = 0;
		printf(" (supports %s)\n", devVer);
		free(devVer);
	}

	// construct command queue
	queue = clCreateCommandQueue(context, devices[0], 0, NULL);
	if(queue == 0)
	{
		perror("Can't create command queue\n");
		clReleaseContext(context);
		return 0;
	}

	// create cl buffers
	cl_mem cl_adj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * MAX_SIZE * MAX_SIZE, NULL, NULL);
	cl_mem cl_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * MAX_SIZE * MAX_SIZE, NULL, NULL);
	cl_mem cl_degree = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * MAX_SIZE, NULL, NULL);
	cl_mem cl_distance = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * MAX_SIZE, NULL, NULL);
	cl_mem cl_visited = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * MAX_SIZE, NULL, NULL);
	cl_mem cl_group_min = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * num_groups, NULL, NULL);
	cl_mem cl_group_min_id = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * num_groups, NULL, NULL);
	cl_mem cl_min = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * 1, NULL, NULL);
	cl_mem cl_min_id = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * 1, NULL, NULL);
	if(cl_adj == 0 || cl_weight == 0 || 
		cl_degree == 0 || cl_distance == 0 || cl_visited == 0 ||
		cl_group_min == 0 || cl_group_min_id == 0 ||
		cl_min == 0 || cl_min_id == 0)
	{
		perror("Can't create OpenCL buffer");
		clReleaseMemObject(cl_adj);
		clReleaseMemObject(cl_weight);
		clReleaseMemObject(cl_degree);
		clReleaseMemObject(cl_distance);
		clReleaseMemObject(cl_visited);
		clReleaseMemObject(cl_group_min);
		clReleaseMemObject(cl_group_min_id);
		clReleaseMemObject(cl_min);
		clReleaseMemObject(cl_min_id);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}

	// create and compile the program object
	program = load_program(context, "shader.cl");
	if(program == 0)
	{
		perror("Error, can't load or build program");
		clReleaseMemObject(cl_adj);
		clReleaseMemObject(cl_weight);
		clReleaseMemObject(cl_degree);
		clReleaseMemObject(cl_distance);
		clReleaseMemObject(cl_visited);
		clReleaseMemObject(cl_group_min);
		clReleaseMemObject(cl_group_min_id);
		clReleaseMemObject(cl_min);
		clReleaseMemObject(cl_min_id);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}

	// create kernel objects from compiled program	
	reduce = clCreateKernel(program, "reduce", NULL);
	if(reduce == 0)
	{
		perror("Error, can't load kernel reduce");
		clReleaseProgram(program);
		clReleaseMemObject(cl_adj);
		clReleaseMemObject(cl_weight);
		clReleaseMemObject(cl_degree);
		clReleaseMemObject(cl_distance);
		clReleaseMemObject(cl_visited);
		clReleaseMemObject(cl_group_min);
		clReleaseMemObject(cl_group_min_id);
		clReleaseMemObject(cl_min);
		clReleaseMemObject(cl_min_id);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}
	clSetKernelArg(reduce, 0, sizeof(cl_mem), &cl_distance);
	clSetKernelArg(reduce, 1, sizeof(cl_mem), &cl_visited);
	clSetKernelArg(reduce, 2, sizeof(cl_mem), &cl_group_min);
	clSetKernelArg(reduce, 3, sizeof(cl_mem), &cl_group_min_id);
	
	extractMin = clCreateKernel(program, "extractMin", NULL);
	if(extractMin == 0)
	{
		perror("Error, can't load kernel extractMin");
		clReleaseProgram(program);
		clReleaseMemObject(cl_adj);
		clReleaseMemObject(cl_weight);
		clReleaseMemObject(cl_degree);
		clReleaseMemObject(cl_distance);
		clReleaseMemObject(cl_visited);
		clReleaseMemObject(cl_group_min);
		clReleaseMemObject(cl_group_min_id);
		clReleaseMemObject(cl_min);
		clReleaseMemObject(cl_min_id);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}
	clSetKernelArg(extractMin, 0, sizeof(cl_mem), &cl_group_min);
	clSetKernelArg(extractMin, 1, sizeof(cl_mem), &cl_group_min_id);
	clSetKernelArg(extractMin, 2, sizeof(int), &num_groups);	// can't use size_t
	clSetKernelArg(extractMin, 3, sizeof(cl_mem), &cl_min);
	clSetKernelArg(extractMin, 4, sizeof(cl_mem), &cl_min_id);
	clSetKernelArg(extractMin, 5, sizeof(cl_mem), &cl_visited);

	relax = clCreateKernel(program, "relax", NULL);
	if(relax == 0)
	{
		perror("Error, can't load kernel relax");
		clReleaseProgram(program);
		clReleaseMemObject(cl_adj);
		clReleaseMemObject(cl_weight);
		clReleaseMemObject(cl_degree);
		clReleaseMemObject(cl_distance);
		clReleaseMemObject(cl_visited);
		clReleaseMemObject(cl_group_min);
		clReleaseMemObject(cl_group_min_id);
		clReleaseMemObject(cl_min);
		clReleaseMemObject(cl_min_id);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}
	clSetKernelArg(relax, 0, sizeof(cl_mem), &cl_adj);
	clSetKernelArg(relax, 1, sizeof(cl_mem), &cl_weight);
	clSetKernelArg(relax, 2, sizeof(cl_mem), &cl_distance);
	clSetKernelArg(relax, 3, sizeof(cl_mem), &cl_degree);
	clSetKernelArg(relax, 4, sizeof(int), &max_size);
	clSetKernelArg(relax, 5, sizeof(cl_mem), &cl_min_id);

	scanf("%d", &num_cases);

	for(c = 0; c < num_cases; c++)
	{
		scanf("%d%d", &n, &m);

		// initialize
		for(i = 0; i < n; i++)
		{
			degree[i] = 0;
			distance[i] = INF;
			visited[i] = 0;
		}

		for(i = 0; i < m; i++)
		{
			scanf("%d%d%d", &u, &v, &w);
			adj[u][degree[u]] = v;
			weight[u][degree[u]] = w;
			adj[v][degree[v]] = u;
			weight[v][degree[v]] = w;
			degree[u]++;
			degree[v]++;
		}

		// Set vertex 0 as source
		src = 0;
		distance[src] = 0;

		// copy data to cl buffer
		cl_event startEvt;
		clEnqueueWriteBuffer(queue, cl_adj, CL_TRUE, 0, sizeof(int) * MAX_SIZE * MAX_SIZE, &adj[0], 0, NULL, NULL);
		clEnqueueWriteBuffer(queue, cl_weight, CL_TRUE, 0, sizeof(int) * MAX_SIZE * MAX_SIZE, &weight[0], 0, NULL, NULL);
		clEnqueueWriteBuffer(queue, cl_degree, CL_TRUE, 0, sizeof(int) * MAX_SIZE, &degree[0], 0, NULL, NULL);
		clEnqueueWriteBuffer(queue, cl_distance, CL_TRUE, 0, sizeof(int) * MAX_SIZE, &distance[0], 0, NULL, NULL);
		clEnqueueWriteBuffer(queue, cl_visited, CL_TRUE, 0, sizeof(int) * MAX_SIZE, &visited[0], 0, NULL, &startEvt);
		clWaitForEvents(1, &startEvt);

		// dijkstra
		gettimeofday(&t_start, NULL);
		for(i = 0; i < n; i++)
		{
			// invoke kernel reduce
			local_work_size = 256;
			global_work_size = reduce_work_size;
			clSetKernelArg(reduce, 4, sizeof(int), &n);
			err = clEnqueueNDRangeKernel(queue, reduce, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
			if(err == CL_SUCCESS){}
			else
			{
				printf("Error: can't enqueue kernel reduce\n");
			}

			// invoke kernel extractMin
			local_work_size = num_groups;
			global_work_size = num_groups;
			err = clEnqueueNDRangeKernel(queue, extractMin, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
			if(err == CL_SUCCESS){}
			else
			{
				printf("Error: can't enqueue kernel extractMin\n");
				printf("%d\n", err);
			}

			// invoke kernel relax
			local_work_size = 256;
			global_work_size = n + (256 - n % 256);
			err = clEnqueueNDRangeKernel(queue, relax, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
			if(err == CL_SUCCESS){}
			else
			{
				printf("Error: can't enqueue kernel relax\n");
			}		
		}
		cl_event finalEvt;
		clEnqueueReadBuffer(queue, cl_distance, CL_TRUE, 0, sizeof(int) * MAX_SIZE, &distance[0], 0, NULL, &finalEvt);
		clWaitForEvents(1, &finalEvt);

		gettimeofday(&t_end, NULL);
		msec = (t_end.tv_sec-t_start.tv_sec)*1000;
		msec += (t_end.tv_usec-t_start.tv_usec)/1000;
		total += msec;

		// Output
		printf("Vertex\t\tDistance from source (path)\n");
		for(int i = 0; i < n; i++) {
			printf("%d", i);
			if (i == src)
				printf(" (source)\t");
			else
				printf("\t\t");
			printf("%d\n", distance[i]);
		}
		printf("Elapsed time: %d ms\n", total);
		printf("======================================\n");
	}

	// release kernel
	clReleaseKernel(reduce);
	clReleaseKernel(extractMin);
	clReleaseKernel(relax);
	clReleaseProgram(program);
	clReleaseMemObject(cl_degree);
	clReleaseMemObject(cl_distance);
	clReleaseMemObject(cl_visited);
	clReleaseMemObject(cl_group_min);
	clReleaseMemObject(cl_group_min_id);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	return 0;
}

cl_program load_program(cl_context context, const char* filename)
{
	FILE *fp;
	size_t length;
	char *data;
	const char* source;
	size_t ret;

	// open file
	fp = fopen(filename, "rb");
	if(fp == NULL)
		perror("Error opening file\n");

	// get file length
	fseek (fp, 0, SEEK_END);
	length = ftell (fp);
	fseek (fp, 0, SEEK_SET);	// rewind(fp);

	// read program source
	data = (char*)malloc((length+1) * sizeof(char));
	ret = fread(data, sizeof(char), length, fp);
	if(ret != length)
		perror("Error reading file");
	data[length] = 0;

	// create and build program object
	source = &data[0];
	cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, NULL);
	if(program == 0)
		return 0;

	// compile program
	if(clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS)
		return 0;

	free(data);
	fclose(fp);

	return program;
}
