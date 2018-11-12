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
				send_data_queue.back().push_back(value);

			std::cout << "Sending: ";
			for (auto const& value: send_data_queue.back())\
				std::cout << value << ",";
			std::cout << std::endl;			

			send_req_queue.emplace_back();
			MPI_Isend(&send_data_queue.back(),
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

	void init_receive(const std::set<int>& recv_from, MPI_Datatype dtype, MPI_Comm comm){
		int size_tmp = -1;
		std::cout << "receiving from ";
		for (auto const& i: recv_from) std::cout << i << ",";
		std::cout << std::endl;

		for (auto const& i: recv_from){
			MPI_Status status;
			MPI_Probe(i, TAG_DATA, comm, &status);
			MPI_Get_count(&status, MPI_INT, &size_tmp);
			
			std::cout << "size: " << size_tmp << std::endl;
			recv_data_queue[i].resize(size_tmp);

			std::vector<int> v(size_tmp, 0);
			MPI_Recv(&v[0], size_tmp, dtype,\
					  i, TAG_DATA, comm, MPI_STATUS_IGNORE);
			std::cout << "Received: ";
			for (auto const& j: v) std::cout << j << ",";
			std::cout << std::endl;
		}
	}

	// void finalize_out(int rank, std::vector<T>& out){
		// std::cout << rank << " here with " << recv_req_queue.size() << " elems ";
		// MPI_Wait(recv_req_queue[1-rank], MPI_STATUS_IGNORE);
		// for (int i=0; i < recv_req_queue.size(); i++){
		// 	// if (i != rank) MPI_Wait(&recv_req_queue[i], MPI_STATUS_IGNORE);
		// 	std::cout << i << ",";
		// }
		// std::cout << std::endl;
		// }
		// for (auto const& request: recv_req_queue) MPI_Wait(&request, MPI_STATUS_IGNORE);
		// weird

		// long final_size = 0;
		// for (auto const& part: recv_data_queue) final_size += part.size();
		// std::cout << "reserved size: " << final_size << std::endl;
		// out.reserve(final_size);
		// for(auto && v: recv_data_queue)\
		//   out.insert(out.end(), v.begin(), v.end());
	// }

	~DataHandler(){
		send_data_queue.clear();
		send_req_queue.clear();
		recv_data_queue.clear();
		recv_data_queue.clear();
	}
};