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

#define SIZE_TAG 401
#define DATA_TAG 402


// IMPLEMENT ME!
template <typename T, typename Pred>
void mpi_extract_if(MPI_Comm comm, const std::vector<T>& in, std::vector<T>& out, Pred pred){
	// allocate memory for flag
	int rank, size;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	// creating a custom datatype
	MPI_Datatype JUST_BYTES;
	MPI_Type_contiguous(sizeof(T), MPI_BYTE, &JUST_BYTES);
	MPI_Type_commit(&JUST_BYTES);

	// creating a window
	int* im_free; // try changing it to bool
	MPI_Alloc_mem(sizeof(int), MPI_INFO_NULL, &im_free);
	*im_free = 0;

	MPI_Win win_status;
	MPI_Win_create(im_free, sizeof(int), sizeof(int),\
				   MPI_INFO_NULL, comm, &win_status);
	
	// everybody should finish with that before the start

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
	std::set<int> visited;
	std::set<int> to_visit;
	for(int i=0; i<size; i++) if (i != rank) to_visit.insert(i);

	//check if somebody in the queue is ready to interact
	
	//say that I'm ready
	*im_free = 1;

	int other_free;
	MPI_Request curr_req;
	int size_tmp;
	for(int k=0; k < 3; k++){ //change it into a while loop
		if (to_visit.empty()) break;
		visited.clear();
		for(auto const& i : to_visit){
			MPI_Win_lock(MPI_LOCK_SHARED, i, 0, win_status);
			MPI_Rget(&other_free, 1, MPI_INT, \
					i, 0, 1, MPI_INT, win_status, &curr_req);
			MPI_Wait(&curr_req, MPI_STATUS_IGNORE);
			MPI_Win_unlock(i, win_status);
			
			std::cout << k << ":" << rank << " got " << other_free << " from " << i << std::endl;
			if (other_free){
				std::cout << k << ":" << rank << " sends data to " << i << std::endl;
				size_tmp = rank_elements[i].size();
				MPI_Isend(&size_tmp, 1, MPI_INT, i, SIZE_TAG, comm, &curr_req);
				//transfer data
				visited.insert(i);
			}
		}
		for (auto const& i: visited){
			MPI_Recv(&size_tmp, 1, MPI_INT, i, SIZE_TAG, comm, MPI_STATUS_IGNORE);
			std::cout << k << ":" << rank << " gets " << size_tmp << " elemets from " << i << std::endl;
			to_visit.erase(i);
		}
		std::cout << k << ":" << rank << " visited: [";
		for (auto const& i: visited) std::cout << i << ",";
		std::cout << "]" << std::endl;

		std::cout << k << ":" << rank << " has yet to visit: [";
		for (auto const& i: to_visit) std::cout << i << ",";
		std::cout << "]" << std::endl;
	}

	// finalizing setup
	// MPI_Win_fence(0, win_status);
	MPI_Type_free(&JUST_BYTES);

	MPI_Win_free(&win_status);
	MPI_Free_mem(im_free);
} // mpi_extract_if

#endif // A1_HPP