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

#include <iostream>
#include <random>
#include <list>
#include <set>
#include <algorithm>
#include <functional>


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

	// compute extract_if and fill matrix of lists with elements
	std::mt19937 rng(0); //try using different seed if it's gonna be uneven
	std::uniform_int_distribution<int> idx_dist(0, size-1);

	std::vector< std::list<T> > rank_elements(size); //test this
	for(auto const& value: in){
		if(pred(value)) rank_elements[idx_dist(rng)].push_back(value);
	}
	// by now in rank_elements I have extracted elements distributed per rank
	// also we can now check size of each message easily

	// create a set of indices whom we should visit
	std::set<int> to_visit;
	for(int i=0; i<size; i++) if (i != rank) to_visit.insert(i);

	//check if somebody in the queue is ready to interact
	*comm_with = rank;

	int other_comm_with;
	for(int k=0; k < 5; k++){
		std::cout << rank << " in here " << std::endl;
		//partner exchange phase
		for(auto const& i : to_visit){
			MPI_Win_lock(MPI_LOCK_EXCLUSIVE, i, 0, win_status);
			MPI_Get(&other_comm_with, 1, \
					MPI_INT, i, 0, 1, \
					MPI_INT, win_status);
			std::cout << rank << " got " << other_comm_with << " from " << i << std::endl;
			if(other_comm_with == i){
				std::cout << rank << " initiates comm with " << i << std::endl;
				MPI_Put(&rank, 1, MPI_INT, i, 0, 1, MPI_INT, win_status);
				MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, win_status);
				MPI_Put(&i, 1, MPI_INT, rank, 0, 1, MPI_INT, win_status);
				MPI_Win_unlock(rank, win_status);
				MPI_Win_unlock(i, win_status); // unlocking so that anyone can see this guy is busy
			} else  MPI_Win_unlock(i ,win_status);
		}
		//data exchange phase
		std::cout << rank << " now comms with " << *comm_with << std::endl;
		if (*comm_with != rank) {
			std::cout << rank << " is exchanges data with " << *comm_with << std::endl;
			to_visit.erase(*comm_with);
			MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, win_status);
			MPI_Put(&rank, 1, MPI_INT, rank, 0, 1, MPI_INT, win_status);
			MPI_Win_unlock(rank, win_status);
		}


	}

	//
	MPI_Win_free(&win_status);
	MPI_Free_mem(comm_with);
} // mpi_extract_if

#endif // A1_HPP