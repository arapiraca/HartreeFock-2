#ifndef HFFUNCTIONS_H
#define HFFUNCTIONS_H

#include <iostream>
#include <math.h>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <ctime>
#include <bitset>
#include <iomanip>
#include <omp.h>
#include <cstdarg>
#include <cstdlib>

extern "C" void dsyev_( char* jobz, char* uplo, int* n, double* a, int* lda, double* w, double* work, int* lwork, int* info );
extern "C" void dgemm_( char* transa, char* transb, int* m, int* n, int* k, double* alpha, double* a, int* lda, double* b, int* ldb, double* beta, double* c, int* ldc );
#define RM_dgemm(a, b, c, d, e, f, g, h, i, j, k, l, m) dgemm_(b, a, d, c, e, f, i, d, g, j, k, l, d)

const std::string PATH = "/mnt/home/novarios/HartreeFock/files/";

struct Input_Parameters;
struct Shells;
struct States;
struct Model_Space;
struct Single_Particle_States;
class J_Matrix_Elements;

int number_level(int m, int n, int M);
int number_shell(int m, int n, int M);
int number_diff(int m, int n, int N);
int number_J(int m, int n, int M, int J);
Input_Parameters Get_Input_Parameters(const std::string &infile);
Model_Space Build_Model_Space(const Input_Parameters &Parameters);
std::vector<double> projection(const std::vector<double> &u, const std::vector<double> &v);
void Separate_Particles_Holes(Single_Particle_States &States, const int &p, const int &n, const Model_Space &Space);
void GramSchmidt(std::vector<std::vector<double> > &Vectors);
Single_Particle_States Build_Single_Particle_States(const Input_Parameters &Parameters, const Model_Space &Space);
void Read_J_Matrix_Elements(const std::string &MEfile, const Model_Space &Space, J_Matrix_Elements &ME);
J_Matrix_Elements Get_J_Matrix_Elements(const Input_Parameters &Parameters, const Model_Space &Space);
Single_Particle_States Hartree_Fock_States2(const Input_Parameters &Parameters, const Single_Particle_States &States, const Model_Space &Space, const J_Matrix_Elements &ME);
J_Matrix_Elements Convert_To_HF_Matrix_Elements(const std::string &MatrixElements, const Single_Particle_States &States, const Model_Space &Space, const J_Matrix_Elements &ME);

struct Input_Parameters{
  int P, N; //Number of protons and neutrons
  int COM; //flag for center-of-mass matrix elements
  std::string LevelScheme; //level scheme path
  std::string MatrixElements; //matrix elements path
  std::string COMMatrixElements; //com matrix elements path
};

struct Shells{
  std::vector<int> sp_ind;
  std::vector<int> pn_ind;
  std::vector<int> tz;
  std::vector<int> n;
  std::vector<int> l;
  std::vector<double> j;
  std::vector<double> energy;
  std::vector<std::vector<int> > sp_levels;
  std::vector<std::vector<int> > pn_levels;
  std::vector<std::string> name;
  void resize(size_t size);
};

struct States{
  std::vector<int> sp_ind;
  std::vector<int> pn_ind;
  std::vector<int> tz;
  std::vector<int> n;
  std::vector<int> l;
  std::vector<double> j;
  std::vector<double> m;
  std::vector<double> energy;
  std::vector<int> sp_shell;
  std::vector<int> pn_shell;
  void resize(size_t size);
};

struct Model_Space{
  int leveltot, pleveltot, nleveltot; //number of total single-particle levels
  int shelltot, pshelltot, nshelltot; //number of total shells
  int A; //nucleus mass
  double HOEnergy;
  double max2J;
  Shells pShells, nShells, spShells; //list of single particle state principal quantum numbers
  States pStates, nStates, spStates; //list of single particle state total angular momentum
  int Pocc, Nocc; //number of occupied proton and neutron shells
};

struct Single_Particle_States{
  std::vector<std::vector<double> > holes; //list of sp hole states given as vectors of coefficients
  std::vector<std::vector<double> > particles; //list of sp particle states given as vectors of coefficients
  std::vector<double> h_energies; //list of sp-hole energies
  std::vector<double> pt_energies; //list of sp-particle energies
  std::vector<std::vector<double> > protons; 
  std::vector<std::vector<double> > neutrons;
  std::vector<double> p_energies;
  std::vector<double> n_energies;
  std::vector<double> p_j;
  std::vector<double> n_j;
  std::vector<int> p_l;
  std::vector<int> n_l;
  std::vector<double> h_j;
  std::vector<double> pt_j;
  std::vector<double> h_l;
  std::vector<double> pt_l;
};

class J_Matrix_Elements{
private:
  int pshelltot, nshelltot;
  std::vector<double> pjvec, njvec;
  std::vector<double> POBME;
  std::vector<double> NOBME;
  std::vector<double> PPTBME;
  std::vector<double> NNTBME;
  std::vector<double> PNTBME;
public:
  J_Matrix_Elements(int, int, std::vector<double>, std::vector<double>); //constructor using shelltots and J2max
  void set_pJME(int, double);
  void set_nJME(int, double);
  double get_pJME(int) const;
  double get_nJME(int) const;
  void set_ppJME(int, int, int, int, double, double);
  void set_nnJME(int, int, int, int, double, double);
  void set_pnJME(int, int, int, int, double, double);
  double get_ppJME(int, int, int, int, double) const;
  double get_nnJME(int, int, int, int, double) const;
  double get_pnJME(int, int, int, int, double) const;
  int cut_ppJME();
  int cut_nnJME();
  int cut_pnJME();
};

#endif
