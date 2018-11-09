// File: a1.hpp
// FirstName
// LastName
// I AFFIRM THAT WHEN WORKING ON THIS ASSIGNMENT
// I WILL FOLLOW UB STUDENT CODE OF CONDUCT, AND
// I WILL NOT VIOLATE ACADEMIC INTEGRITY RULES

#ifndef A1_HPP
#define A1_HPP

#include <vector>
#include <mpi.h>


#include <unistd.h>
#include <iostream>
#include <random>
#include <list>
#include <set>
#include <algorithm>
#include <functional>

void set_comm(int value, int rank, MPI_Win window){
		MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, window);
		MPI_Put(&value, 1, MPI_INT, rank, 0, 1, MPI_INT, window);
		MPI_Win_unlock(rank, window);
	}

// IMPLEMENT ME!
template <typename T, typename Pred>
void mpi_extract_if(MPI_Comm comm, const std::vector<T>& in, std::vector<T>& out, Pred pred) {
	// allocate memory for flag
	int rank, size;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	int* comm_with; //if comm_with == rank - he's free
	MPI_Alloc_mem(sizeof(int), MPI_INFO_NULL, &comm_with);
	*comm_with = size+1;

	MPI_Win win_status;
	MPI_Win_create(comm_with, sizeof(int), sizeof(int),\
				   MPI_INFO_NULL, comm, &win_status);
	MPI_Win_fence(0, win_status);

	// compute extract_if and fill matrix of lists with elements
	std::mt19937 rng(rank); //try using different seed if it's gonna be uneven
	std::uniform_int_distribution<int> idx_dist(0, size-1);

	std::vector< std::list<T> > rank_elements(size); //test this
	for(auto const& value: in){
		if(pred(value)) rank_elements[idx_dist(rng)].push_back(value);
	}
	if (rank == 0) sleep(5);
	// by now in rank_elements I have extracted elements distributed per rank
	// also we can now check size of each message easily

	// create a set of indices whom we should visit
	std::set<int> to_visit;
	for(int i=0; i<size; i++) if (i != rank) to_visit.insert(i);


	std::vector<long> send_sizes(size);
	std::vector<long> recv_sizes(size);
	
	for(int i=0; i<size; i++) send_sizes[i] = rank_elements[i].size();

	long init_size = 0;
	for (auto n: send_sizes) init_size += n;
	std::cout << rank << " had " << init_size << " elements" << std::endl;

	MPI_Alltoall(
		send_sizes.data(), 1, MPI_LONG,
		recv_sizes.data(), 1, MPI_LONG,
		comm
	);
	
	long final_size = 0;
	for (auto n: recv_sizes) final_size += n;
	std::cout << rank << " received " << final_size << " elements" << std::endl;

	////check if somebody in the queue is ready to interact
	
	////say that I'm ready
	// *comm_with = rank;

	// int other_comm_with = size+2;
	// MPI_Request req_get;

	// for(int k=0; k < 3; k++){
	// 	//partner exchange phase
	// 	for(auto const& i : to_visit){
	// 		other_comm_with = size+2;
	// 		MPI_Win_lock(MPI_LOCK_EXCLUSIVE, i, 0, win_status);

	// 		MPI_Rget(&other_comm_with, 1, MPI_INT, \
	// 				i, 0, 1, MPI_INT, win_status, &req_get);
	// 		MPI_Wait(&req_get, MPI_STATUS_IGNORE);
			
	// 		std::cout << k << ":" << rank << " got " << other_comm_with << " from " << i << std::endl;
	// 		if (other_comm_with == i){
	// 			std::cout << rank << " initiates comm with " << i << std::endl;
	// 			MPI_Put(&rank, 1, MPI_INT, i, 0, 1, MPI_INT, win_status);
				
	// 			MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, win_status);
	// 			*comm_with = i;
	// 			MPI_Win_unlock(rank, win_status);

	// 			MPI_Win_unlock(i, win_status);
	// 			break;
				
	// 			// unlocking so that anyone can see this guy is busy
	// 		} else MPI_Win_unlock(i, win_status);
	// 	}
	// 	//data exchange phase
	// 	std::cout << k << ":" << rank << " now comms with " << *comm_with << std::endl;

	// 	if (*comm_with != rank) {
	// 		std::cout << rank << " exchanges data with " << *comm_with << std::endl;
	// 		to_visit.erase(*comm_with);
	// 		set_comm(rank, rank, win_status);
	// 		// MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, win_status);
	// 		// MPI_Put(&rank, 1, MPI_INT, rank, 0, 1, MPI_INT, win_status);
	// 		// MPI_Win_unlock(rank, win_status);
	// 	}


	// }

	//
	MPI_Win_fence(0, win_status);
	MPI_Win_free(&win_status);
	MPI_Free_mem(comm_with);
} // mpi_extract_if

#endif // A1_HPP