#include "test.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>

#include <mpi.h>
#include <unistd.h>


int main(int argc, char* argv[]) {
    int size, rank;
    std::cout << "Start" << std::endl;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<std::deque<int>> data_queue;
    data_queue.resize(2);
    data_queue[0].push_back(0);
    data_queue[0].push_back(1);
    data_queue[0].push_back(2);

    data_queue[1].push_back(3);
    data_queue[1].push_back(4);
    data_queue[1].push_back(5);

    DataHandler<int> handler(data_queue);
    
    std::set<int> send_to = {1-rank};

    std::cout<< rank << " "; 
    handler.send_to(send_to, MPI_INT, MPI_COMM_WORLD);

    std::cout<< rank << " "; 
    handler.init_receive(send_to, MPI_INT, MPI_COMM_WORLD);
    handler.finalize_out(rank);
    handler.clear_sent();

   	// // // handler.print(1-rank);
   	std::cout << "Fin" << std::endl;
    return MPI_Finalize();
} // main
