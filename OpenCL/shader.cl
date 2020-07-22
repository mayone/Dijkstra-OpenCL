__kernel void init(__global int *degree,
					__global int *distance,
					__global int *visited,
					__const int n)
{
	int tid = get_global_id(0);
	//if(tid < n)
	//{
		degree[tid] = 0;
		distance[tid] = 1e9;
		visited[tid] = 0;
	//}
}

__kernel
void reduce(__global int *distance,
			__global int *visited,
			__global int *group_min,
			__global int *group_min_id,
			__const int n)
{
	__local int fetch[256];
	__local int fetch_id[256];

	int global_id = get_global_id(0);
	int local_id = get_local_id(0);
	int element;
	int item_min = 1e9;
	int item_min_id = global_id;
	int offset;
	int this, next;
	int this_id, next_id;

	// use all work_item to scan through the distance array
	while(global_id < n)
	{
		element = distance[global_id];
		if(!visited[global_id] && (element < item_min))	// not visited
		{
			item_min = element;
			item_min_id = global_id;
		}
		global_id += get_global_size(0);
	}

	// reduce to find group_min
	fetch[local_id] = item_min;
	fetch_id[local_id] = item_min_id;
	barrier(CLK_LOCAL_MEM_FENCE);	// wait for all group_item fetched

	for(offset = get_local_size(0) >> 1; offset > 0; offset >>= 1)
	{
		if(local_id < offset)
		{
			this = fetch[local_id];
			next = fetch[local_id + offset];
			this_id = fetch_id[local_id];
			next_id = fetch_id[local_id + offset];
			if(this < next)
			{
				fetch[local_id] = this;
				fetch_id[local_id] = this_id;
			}
			else
			{
				fetch[local_id] = next;
				fetch_id[local_id] = next_id;
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);	// wait for all group_min are gotten
	}
	if(local_id == 0)
	{
		int group_id = get_group_id(0);
		group_min[group_id] = fetch[0];
		group_min_id[group_id] = fetch_id[0];
	}
}

__kernel
void extractMin(__global int *group_min,
				__global int *group_min_id,
				__const int n,
				__global int *min,
				__global int *min_id,
				__global int *visited)
{
	int global_id = get_global_id(0);
	int local_id = get_local_id(0);
	int offset;
	int this, next;
	int this_id, next_id;

	for(offset = get_local_size(0) >> 1; offset > 0; offset >>= 1)
	{
		if(local_id < offset)
		{
			next = group_min[local_id + offset];
			next_id = group_min_id[local_id + offset];
			this = group_min[local_id];
			this_id = group_min_id[local_id];
			if(this < next)
			{
				group_min[local_id] = this;
				group_min_id[local_id] = this_id;
			}
			else
			{
				group_min[local_id] = next;
				group_min_id[local_id] = next_id;
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);	// wait for all group_min are gotten
	}
	if(local_id == 0)
	{
		int group_id = get_group_id(0);
		min[group_id] = group_min[0];
		min_id[group_id] = group_min_id[0];
		visited[min_id[group_id]] = 1;
	}
}

__kernel void relax(__global int *adj,
					__global int *weight,
					__global int *distance,
					__global int *degree,
					__const int max_size,
					__global int *min_id)
{
	int global_id = get_global_id(0);
	int length = degree[min_id[0]];

	if(global_id < length)
	{
		if( distance[adj[min_id[0]*max_size + global_id]] >
			distance[min_id[0]] + weight[min_id[0]*max_size + global_id])
		{
			distance[adj[min_id[0]*max_size + global_id]] =
			distance[min_id[0]] + weight[min_id[0]*max_size + global_id];
		}
	}
}
