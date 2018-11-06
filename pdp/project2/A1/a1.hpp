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

// IMPLEMENT ME!
template <typename T, typename Pred>
void mpi_extract_if(MPI_Comm comm, const std::vector<T>& in, std::vector<T>& out, Pred pred) {
	for(auto const& value: in) {
		if(pred(value)) out.push_back(value);
	}

	// communicating each nodes out size
	int rank;
	MPI_Comm_rank(comm, &rank);

	const int size = 4;
	int tab[size];

	if (rank == 0) {

		tab[0] = (int) out.size();
		MPI_Status stat;
		for (int i=1; i < size; i++)
			MPI_Recv(&tab[i], 1, MPI_INT, i, 111, comm, &stat);
	} else {
		tab[rank] = (int) out.size();
		MPI_Send(&tab[rank], 1, MPI_INT, 0, 111, comm);
	}
	if (rank == 0) {
		std::cout << "Out sizes: [";
		for(int i=0; i<size; i++) std::cout << tab[i] << ",";
		std::cout << "]" << std::endl;
	}

} // mpi_extract_if

#endif // A1_HPP
