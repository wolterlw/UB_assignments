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
#include <algorithm>
#include <functional>

// IMPLEMENT ME!
template <typename T, typename Pred>
void mpi_extract_if(MPI_Comm comm, const std::vector<T>& in, std::vector<T>& out, Pred pred) {
	// allocate memory for flag
	int rank, size;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	bool* flag_buf;
	MPI_Alloc_mem(sizeof(bool), MPI_INFO_NULL, &flag_buf);
	*flag_buf = false;

	MPI_Win win_waiting;
	MPI_Win_create(flag_buf, sizeof(bool), sizeof(bool),\
				   MPI_INFO_NULL, comm, &win_waiting);

	// compute extract_if
	for(auto const& value: in) {
		if(pred(value)) out.push_back(value); //since this is a vector maybe it's not the best way
	}

	std::cout << rank << " finished extracting" << std::endl;

	// compute random indices
	std::vector<int> indices(out.size());
	std::mt19937 rng(0); //try using different seed if it's gonna be uneven
	std::uniform_int_distribution<int> idx_dist(0, size-1);
	std::generate(\
		std::begin(indices),\
		std::end(indices),\
		std::bind(idx_dist, rng));
	// TODO: try deleting indices that ==rank to speed up list traversal

	// create a vector of indices whom we should visit
	std::list<int> to_visit(size);
	std::iota(std::begin(to_visit), std::end(to_visit), 0);
	to_visit.remove(rank);

	int rank_from;
	// call checking if anyone is ready to receive
	while (~to_visit.empty()){
		bool is_waiting = false;
		for(auto const& i : to_visit){
			MPI_Win_lock(MPI_LOCK_EXCLUSIVE, i, 0, win_waiting);
			MPI_Get(&is_waiting, 1, \
					MPI_C_BOOL, i, 0, 1, \
					MPI_C_BOOL, win_waiting);
			if(is_waiting){
				init_transfer(rank, i);
				MPI_Win_unlock(i, win_waiting); // unlocking so that anyone can see this guy is busy
				transfer_data(rank, i, indices, out); //performing actual transfer
				to_visit.remove(i);
			}
		}
		*flag_buf = true; //telling that I'm ready to receive
		rank_from = wait_for_transfer(rank); //setting flag_buf to false; 
		//make sure this ^ guy won't wait forever
		receive_data(rank, rank_from, indices, out);
	}

	//
	MPI_Win_free(&win_waiting);
	MPI_Free_mem(flag_buf);
} // mpi_extract_if

#endif // A1_HPP

void init_transfer(int rank, int to_rank){

}

void transfer_data(int rank, int to_rank, const std::vector<int>& indices, const std::vector<int>& out){

}

int wait_for_transfer(int rank){

}

int receive_data(int rank, int rank_from, const std::vector<int>& indices, const std::vector<int>& out){
	
}