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
#include <deque>
#include <set>
#include <algorithm>
#include <functional>
#include <iterator>

#define TAG_SIZE 401
#define TAG_DATA 402

template <typename T>
class DataHandler{
	std::deque<std::vector<T>> send_data_queue;
	std::vector<std::vector<T>> recv_data_queue;
	std::deque<MPI_Request> send_req_queue;
	std::vector<MPI_Request> recv_req_queue;

	const std::vector<std::deque<T>>& data;

	public:
	DataHandler(const std::vector<std::deque<T>>& data_deque) :data(data_deque){
		recv_data_queue.resize(data_deque.size());
		recv_req_queue.resize(data_deque.size());
	}

	void send_to(const std::set<int>& send_to, MPI_Datatype dtype, MPI_Comm comm){
		for (auto const& i: send_to){ 
			send_data_queue.emplace_back();
			for (auto const& value: data[i])\
				send_data_queue[0].push_back(value);

			send_req_queue.emplace_back();
			MPI_Isend(&send_data_queue.back()[0],
					  send_data_queue.back().size(), dtype,
					  i, TAG_DATA, comm, &send_req_queue.back());
		}
	}

	void clear_sent(){
		for(int i=0; i < send_req_queue.size(); i++){
		int flag=0;
			MPI_Test(&send_req_queue[i], &flag, MPI_STATUS_IGNORE);
			if (flag) send_data_queue[i].clear();
		}
	}

	void init_receive(MPI_Datatype dtype, MPI_Comm comm){
		int size_tmp = -1;
		int flag;
		MPI_Status status;

		while (MPI_Iprobe(MPI_ANY_SOURCE, TAG_DATA, comm, &flag, &status)){
			
			MPI_Get_count(&status, MPI_INT, &size_tmp);
			recv_data_queue[status.MPI_SOURCE].resize(size_tmp);

			MPI_Irecv(&recv_data_queue[status.MPI_SOURCE][0], size_tmp, \
					  dtype, status.MPI_SOURCE, TAG_DATA, comm, \
					  &recv_req_queue[status.MPI_SOURCE]);
		}
	}

	void finalize_out(int rank, std::vector<T>& out){//, std::vector<T>& out){
		for (int i=0; i < recv_req_queue.size(); i++)\
				if (i != rank) MPI_Wait(&recv_req_queue[i], MPI_STATUS_IGNORE);
		
		long final_size = 0;
		for (auto const& part: recv_data_queue) final_size += part.size();
		out.reserve(final_size);
		for(auto && v: recv_data_queue)\
			out.insert(out.end(), v.begin(), v.end());
	}

	~DataHandler(){
		send_data_queue.clear();
		send_req_queue.clear();
		recv_data_queue.clear();
		recv_data_queue.clear();
	}
};

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

	// compute extract_if and fill matrix of deques with elements
	std::mt19937 rng(rank); //try using different seed if it's gonna be uneven
	std::uniform_int_distribution<int> idx_dist(0, size-1);

	std::vector<std::deque<T> > rank_elements(size); //test this
	for(auto const& value: in)
		if(pred(value)) rank_elements[idx_dist(rng)].push_back(value);



	if (rank < 2) {
	  double S = 0.0;
      for (int i = 1; i < 10000; ++i)
		for (int j = 1; j < 10000; ++j)
			for (int k = 1; k < 10000; ++k) S+= i * j * k;
	   std::cout << S << std::endl;
	}
	// by now in rank_elements I have extracted elements distributed per rank

	std::cout << ">" << rank << " here we go" << std::endl;


	// create a set of indices whom we should visit
	std::set<int> visited;
	std::set<int> to_visit;
	for(int i=0; i<size; i++) if (i != rank) to_visit.insert(i);
	
	//say that I'm ready
	DataHandler<T> data_handler(rank_elements);

	*im_free = 1;

	int other_free;
	MPI_Request curr_req;

	int k = 0;
	// checking who else is free to communicate
	while (not to_visit.empty()) { //change it into a while loop
		visited.clear();
		k++; 
		for(auto const& i : to_visit){
			std::cout << ">" << rank << " start" << std::endl;
			MPI_Win_lock(MPI_LOCK_SHARED, i, 0, win_status);
			MPI_Get(&other_free, 1, MPI_INT, \
					i, 0, 1, MPI_INT, win_status);
			// MPI_Wait(&curr_req, MPI_STATUS_IGNORE);
			MPI_Win_unlock(i, win_status);
			std::cout << ">" << rank << " stop" << std::endl;

			if (other_free){
	
				visited.insert(i);
			}
		}
		//sending data
		data_handler.send_to(visited, JUST_BYTES, comm);

		//receiving data
		data_handler.init_receive(JUST_BYTES, comm);
		for (auto const& i: visited) to_visit.erase(i);
		std::cout << k << ": " << rank << "visits [" << std::flush;
		for (auto const& i: visited) std::cout << i << "," << std::flush;
		std::cout << "]" << std::endl;
	}
	// finalizing setup
	// data_handler.finalize_out(rank, out);
	MPI_Type_free(&JUST_BYTES);

	MPI_Win_free(&win_status);
	MPI_Free_mem(im_free);
} // mpi_extract_if

#endif // A1_HPP 