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
	const MPI_Comm& comm;

	public:
	DataHandler(const std::vector<std::deque<T>>& data_deque, MPI_Comm comm) :data(data_deque), comm(comm){
		recv_data_queue.reserve(data_deque.size());
		recv_req_queue.reserve(data_deque.size());
	}

	void send_to(const std::set<int>& send_to, MPI_Datatype dtype){
		for (auto const& i: send_to){
			send_data_queue.emplace_back();
			for (auto const& value: data[i])\
				send_data_queue.back().push_back(value);

			send_req_queue.emplace_back();
			MPI_Isend(&send_data_queue.back(),
					  send_data_queue.back().size(), dtype,
					  i, TAG_DATA, comm, &send_req_queue.back());
		}
	}

	void clear_sent(){
		int flag=0;
		for(int i=0; i<send_req_queue.size(); i++){
			MPI_Test(&send_req_queue[i], &flag, MPI_STATUS_IGNORE);
			if (flag) send_data_queue[i].clear();
		}
	}

	void init_receive(const std::set<int>& recv_from, MPI_Datatype dtype){
		int size_tmp = -1;
		for (auto const& i: recv_from){
			MPI_Recv(&size_tmp, 1, MPI_INT, i, TAG_SIZE,\
					 comm, MPI_STATUS_IGNORE);
			recv_data_queue.emplace(recv_data_queue.begin()+i, size_tmp);
			// recv_req_queue.emplace(recv_data_queue.begin()+i);
			MPI_Irecv(&recv_data_queue[i], size_tmp, dtype,\
					  i, TAG_DATA, comm, &recv_req_queue[i]);
		}
	}

	void finalize_out(std::vector<T>& out){
		for (int i=0; i < recv_req_queue.size(); i++)\
			MPI_Wait(&recv_req_queue[i], MPI_STATUS_IGNORE);
		// for (auto const& request: recv_req_queue) MPI_Wait(&request, MPI_STATUS_IGNORE);
		// weird

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



	if (rank < 3) sleep(5);
	// by now in rank_elements I have extracted elements distributed per rank
	// also we can now check size of each message easily

	// create a set of indices whom we should visit
	std::set<int> visited;
	std::set<int> to_visit;
	for(int i=0; i<size; i++) if (i != rank) to_visit.insert(i);

	//check if somebody in the queue is ready to interact
	
	//say that I'm ready
	DataHandler<T> data_handler(rank_elements, comm);

	*im_free = 1;

	int other_free;
	MPI_Request curr_req;
	int size_tmp;

	// checking who else is free to communicate
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
				
				// transfer size
				size_tmp = rank_elements[i].size();
				MPI_Isend(&size_tmp, 1, MPI_INT, i, TAG_SIZE, comm, &curr_req);
				visited.insert(i);
			}
		}
		//sending data
		data_handler.send_to(visited, JUST_BYTES);

		MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, win_status);
		*im_free = 0;
		MPI_Win_unlock(rank, win_status);
		
		//receiving data
		data_handler.init_receive(visited, JUST_BYTES); //SEGFAULTS HERE

		MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, win_status);
		*im_free = 1;
		MPI_Win_unlock(rank, win_status);

		for (auto const& i: visited) to_visit.erase(i);
		
		std::cout << k << ":" << rank << " visited: [";
		for (auto const& i: visited) std::cout << i << ",";
		std::cout << "]" << std::endl;

		std::cout << k << ":" << rank << " has yet to visit: [";
		for (auto const& i: to_visit) std::cout << i << ",";
		std::cout << "]" << std::endl;
	}

	// finalizing setup
	// data_handler.finalize_out(out);
	MPI_Type_free(&JUST_BYTES);

	MPI_Win_free(&win_status);
	MPI_Free_mem(im_free);
} // mpi_extract_if

#endif // A1_HPP 