// File: a1.hpp
// Volodymyr
// Liunda
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
#include <cstdlib>

#define TAG_DATA 402

template <typename T>
class DataHandler{
	std::deque<std::vector<T>> send_data_queue;
	std::vector<std::vector<T>> recv_data_queue;
	std::deque<MPI_Request> send_req_queue;
	std::vector<MPI_Request> recv_req_queue;
	
	std::set<int> to_recv_from;
	std::set<int> recvd;
	int rank;
	MPI_Datatype dtype;
	MPI_Comm comm;

	const std::vector<std::deque<T>>& data;

	public:
	DataHandler(int rank, const std::vector<std::deque<T>>& data_deque,\
				MPI_Datatype dtype, MPI_Comm comm\
				) :data(data_deque), rank(rank), dtype(dtype), comm(comm){
		recv_data_queue.resize(data_deque.size());
		recv_req_queue.resize(data_deque.size());
		for (int i=0; i < data_deque.size(); i++)\
			if (i != rank) to_recv_from.insert(i);
	}

	void send_to(int send_to){
		send_data_queue.emplace_back();
		for (auto const& value: data[send_to])\
			send_data_queue.back().push_back(value);

		send_req_queue.emplace_back();

 		MPI_Isend(&send_data_queue.back()[0],
				  send_data_queue.back().size(), dtype,
				  send_to, TAG_DATA, comm, &send_req_queue.back());
	}

	void clear_sent(){
		for(int i=0; i < send_req_queue.size(); i++){
		int flag=0;
			MPI_Test(&send_req_queue[i], &flag, MPI_STATUS_IGNORE);
			if (flag) send_data_queue[i].clear();
		}
	}

	void init_receive(){
		int size_tmp = 0;
		int flag = 0;
		int any = 0;
		MPI_Status status;

		recvd.clear();
		for(auto const& i: to_recv_from){
			MPI_Iprobe(i, TAG_DATA, comm, &flag, &status);
			MPI_Get_count(&status, MPI_INT, &size_tmp);
			if(flag){
				recv_data_queue[i].resize(size_tmp);
				recvd.insert(i);
				MPI_Irecv(&recv_data_queue[status.MPI_SOURCE][0], size_tmp, \
						  dtype, status.MPI_SOURCE, TAG_DATA, comm, \
						  &recv_req_queue[status.MPI_SOURCE]);
			}
		}
		for(auto const& i: recvd) to_recv_from.erase(i);

	}

	int received_all(){
		return to_recv_from.empty();
	}

	void finalize_out(std::vector<T>& out){
		for (int i=0; i < recv_req_queue.size(); i++)\
				if (i != rank) MPI_Wait(&recv_req_queue[i], MPI_STATUS_IGNORE);
		
		long final_size = 0;
		for (auto const& part: recv_data_queue) final_size += part.size();
		out.reserve(final_size);

		for(auto && v: recv_data_queue)\
			out.insert(out.end(), v.begin(), v.end());

		out.insert(out.end(), data[rank].begin(), data[rank].end());
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

	// by now in rank_elements I have extracted elements distributed per rank

	// create a set of indices whom we should visit
	std::set<int> visited;
	std::set<int> to_visit;
	for(int i=0; i<size; i++){
		if (i != rank) to_visit.insert(i);
	}
	
	//say that I'm ready
	DataHandler<T> data_handler(rank, rank_elements, JUST_BYTES, comm);
	
	MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, win_status);
	*im_free = 1;
	MPI_Win_unlock(rank, win_status);

	int other_free;
	MPI_Request curr_req;

	int k = 0;
	std::vector<int> to_visit_vec;
	to_visit_vec.reserve(to_visit.size());
	// checking who else is free to communicate
	while (not to_visit.empty()) { //change it into a while loop
		visited.clear();

		// randomizing visit order
		to_visit_vec.clear();
		for(auto const& i : to_visit) to_visit_vec.push_back(i);
		std::shuffle(to_visit_vec.begin(), to_visit_vec.end(), rng);

		k++; 
		for(auto const& i : to_visit_vec){
			MPI_Win_lock(MPI_LOCK_SHARED, i, MPI_MODE_NOCHECK, win_status);
			MPI_Get(&other_free, 1, MPI_INT, \
					i, 0, 1, MPI_INT, win_status);
			// MPI_Wait(&curr_req, MPI_STATUS_IGNORE);
			MPI_Win_unlock(i, win_status);

			if (other_free){
				visited.insert(i);
				data_handler.send_to(i);
			}
		}
		//sending data
		for(auto const& i : visited) to_visit.erase(i);
		//receiving data
		data_handler.init_receive();
	}
	// finalizing setup
	while (not data_handler.received_all()){
		data_handler.init_receive();
	}
	data_handler.finalize_out(out);
	MPI_Type_free(&JUST_BYTES);

	MPI_Win_free(&win_status);
	MPI_Free_mem(im_free);
} // mpi_extract_if

#endif // A1_HPP 