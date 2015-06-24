#include <iostream>
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
#include "CGC.h"

using namespace std;

extern "C" void dsyev_( char* jobz, char* uplo, int* n, double* a, int* lda, double* w, double* work, int* lwork, int* info );
extern "C" void dgemm_( char* transa, char* transb, int* m, int* n, int* k, double* alpha, double* a, int* lda, double* b, int* ldb, double* beta, double* c, int* ldc );

const string PATH = "/home/sam/Documents/HartreeFock/files/";

int min(int &a, int &b){ return (a <= b ? a : b); }
int max(int &a, int &b){ return (a >= b ? a : b); }
int sign(double &a){ return (a < 0 ? -1 : (a > 0)); }

//m<n in same set (size M)
int number_level(int m, int n, int M)
{
  if(m==n){ std::cerr << "number_level(m,n,M) m=" << m << " should not equal n=" << n << endl; exit(1); }
  return int(n - 1 - 0.5 * m * (3 + m - 2*M));
}
//m<=n in same set (size M)
int number_shell(int m, int n, int M)
{
  return int(n - 0.5 * m * (1 + m - 2*M));
}
//m,n in different sets (n set size N)
int number_diff(int m, int n, int N)
{
  return m * N + n;
}
//bra,ket in same set (size M) with coupled J
int number_J(int m, int n, int M, int J)
{
  return int((n - 0.5 * m * (1 + m - 2*M)) + J * 0.5 * (M*M + M));
}

struct Input_Parameters{
  int P, N; //Number of protons and neutrons
  int COM; //flag for center-of-mass matrix elements
  string LevelScheme; //level scheme path
  string MatrixElements; //matrix elements path
  string COMMatrixElements; //com matrix elements path
};

struct Shells{
  vector<int> sp_ind;
  vector<int> pn_ind;
  vector<int> tz;
  vector<int> n;
  vector<int> l;
  vector<double> j;
  vector<double> energy;
  vector<vector<int> > sp_levels;
  vector<vector<int> > pn_levels;
  vector<string> name;

  void resize(size_t size)
  {
    sp_ind.resize(size);
    pn_ind.resize(size);
    tz.resize(size);
    n.resize(size);
    l.resize(size);
    j.resize(size);
    energy.resize(size);
    sp_levels.resize(size);
    pn_levels.resize(size);
    name.resize(size);
  }

};

struct States{
  vector<int> sp_ind;
  vector<int> pn_ind;
  vector<int> tz;
  vector<int> n;
  vector<int> l;
  vector<double> j;
  vector<double> m;
  vector<double> energy;
  vector<int> sp_shell;
  vector<int> pn_shell;

  void resize(size_t size)
  {
    sp_ind.resize(size);
    pn_ind.resize(size);
    tz.resize(size);
    n.resize(size);
    l.resize(size);
    j.resize(size);
    m.resize(size);
    energy.resize(size);
    sp_shell.resize(size);
    pn_shell.resize(size);
  }

};

struct Model_Space{
  int leveltot, pleveltot, nleveltot; //number of total single-particle levels
  int shelltot, pshelltot, nshelltot; //number of total shells
  int A; //nucleus mass
  double HOEnergy;
  double max2J;
  Shells pShells, nShells, spShells; //list of single particle state principal quantum numbers
  States pStates, nStates, spStates; //list of single particle state total angular momentum
};

struct Single_Particle_States{
  vector<vector<double> > holes; //list of sp hole states given as vectors of coefficients
  vector<vector<double> > particles; //list of sp particle states given as vectors of coefficients
  vector<double> h_energies; //list of sp-hole energies
  vector<double> pt_energies; //list of sp-particle energies
  vector<vector<double> > protons; 
  vector<vector<double> > neutrons;
  vector<double> p_energies;
  vector<double> n_energies;
};

class J_Matrix_Elements{
private:
  int pshelltot, nshelltot;
  vector<double> pjvec, njvec;
  vector<double> POBME;
  vector<double> NOBME;
  vector<double> PPTBME;
  vector<double> NNTBME;
  vector<double> PNTBME;
public:
  J_Matrix_Elements(int, int, vector<double>, vector<double>); //constructor using shelltots and J2max
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

J_Matrix_Elements::J_Matrix_Elements(int pl, int nl, vector<double> pjs, vector<double> njs)
{
  double maxpj = -1.0;
  double maxnj = -1.0;
  for(int i = 0; i < int(pjs.size()); ++i){ if(pjs[i] > maxpj){ maxpj = pjs[i]; } }
  for(int i = 0; i < int(njs.size()); ++i){ if(njs[i] > maxnj){ maxnj = njs[i]; } }
  pshelltot = pl;
  nshelltot = nl;
  pjvec = pjs;
  njvec = njs;
  POBME.assign(pl, 0.0);
  NOBME.assign(nl, 0.0);
  PPTBME.assign(int(2.0 * (maxpj + 1.0) * 0.5 * (pow(0.5*(pow(pl, 2.0) + pl), 2.0) + 0.5*(pow(pl, 2.0) + pl))), 0.0);
  NNTBME.assign(int(2.0 * (maxnj + 1.0) * 0.5 * (pow(0.5*(pow(nl, 2.0) + nl), 2.0) + 0.5*(pow(nl, 2.0) + nl))), 0.0);
  PNTBME.assign(int((maxpj + maxnj + 1.0) * 0.5 * (pow(pl * nl, 2.0) + (pl * nl))), 0.0);
}

void J_Matrix_Elements::set_pJME(int i, double ME){ POBME[i] = ME; }
double J_Matrix_Elements::get_pJME(int i) const
{ return POBME[i]; }

void J_Matrix_Elements::set_nJME(int i, double ME){ NOBME[i] = ME; }
double J_Matrix_Elements::get_nJME(int i) const
{ return NOBME[i]; }

void J_Matrix_Elements::set_ppJME(int m, int n, int l, int k, double J, double ME)
{
  int branum, ketnum, vnum;
  double factor = 1.0;
  if(m > n){ factor *= pow(-1.0, int(pjvec[m] + pjvec[n] + J + 1.0)); swap(m, n); }
  if(l > k){ factor *= pow(-1.0, int(pjvec[l] + pjvec[k] + J + 1.0)); swap(l, k); }
  branum = number_shell(m, n, pshelltot);
  ketnum = number_shell(l, k, pshelltot);
  if(branum > ketnum){ swap(branum, ketnum); }
  vnum = number_J(branum, ketnum, int(0.5*(pow(pshelltot, 2.0) + pshelltot)), J);
  PPTBME[vnum] = factor * ME;
}

void J_Matrix_Elements::set_nnJME(int m, int n, int l, int k, double J, double ME)
{
  int branum, ketnum, vnum;
  double factor = 1.0;
  if(m > n){ factor *= pow(-1.0, int(njvec[m] + njvec[n] + J + 1.0)); swap(m, n); }
  if(l > k){ factor *= pow(-1.0, int(njvec[l] + njvec[k] + J + 1.0)); swap(l, k); }
  branum = number_shell(m, n, nshelltot);
  ketnum = number_shell(l, k, nshelltot);
  if(branum > ketnum){ swap(branum, ketnum); }
  vnum = number_J(branum, ketnum, int(0.5*(pow(nshelltot, 2.0) + nshelltot)), J);
  NNTBME[vnum] = factor * ME;
}

void J_Matrix_Elements::set_pnJME(int m, int n, int l, int k, double J, double ME)
{
  int branum, ketnum, vnum;
  branum = number_diff(m, n, nshelltot);
  ketnum = number_diff(l, k, nshelltot);
  if(branum > ketnum){ swap(branum, ketnum); }
  vnum = number_J(branum, ketnum, pshelltot * nshelltot, J);
  PNTBME[vnum] = ME;
}

double J_Matrix_Elements::get_ppJME(int m, int n, int l, int k, double J) const
{
  int branum, ketnum, vnum;
  double factor = 1.0;
  if(m > n){ factor *= pow(-1.0, int(pjvec[m] + pjvec[n] + J + 1.0)); swap(m, n); }
  if(l > k){ factor *= pow(-1.0, int(pjvec[l] + pjvec[k] + J + 1.0)); swap(l, k); }
  branum = number_shell(m, n, pshelltot);
  ketnum = number_shell(l, k, pshelltot);
  if(branum > ketnum){ swap(branum, ketnum); }
  vnum = number_J(branum, ketnum, int(0.5*(pow(pshelltot, 2.0) + pshelltot)), J);
  return factor * PPTBME[vnum];
}

double J_Matrix_Elements::get_nnJME(int m, int n, int l, int k, double J) const
{
  int branum, ketnum, vnum;
  double factor = 1.0;
  if(m > n){ factor *= pow(-1.0, int(njvec[m] + njvec[n] + J + 1.0)); swap(m, n); }
  if(l > k){ factor *= pow(-1.0, int(njvec[l] + njvec[k] + J + 1.0)); swap(l, k); }
  branum = number_shell(m, n, nshelltot);
  ketnum = number_shell(l, k, nshelltot);
  if(branum > ketnum){ swap(branum, ketnum); }
  vnum = number_J(branum, ketnum, int(0.5*(pow(nshelltot, 2.0) + nshelltot)), J);
  return factor * NNTBME[vnum];
}

double J_Matrix_Elements::get_pnJME(int m, int n, int l, int k, double J) const
{
  int branum, ketnum, vnum;
  branum = number_diff(m, n, nshelltot);
  ketnum = number_diff(l, k, nshelltot);
  if(branum > ketnum){ swap(branum, ketnum); }
  vnum = number_J(branum, ketnum, pshelltot * nshelltot, J);
  return PNTBME[vnum];
}

int J_Matrix_Elements::cut_ppJME()
{
  int indpp, ind = 0;
  for(int i = 0; i < int(PPTBME.size()); ++i)
    {
      if(PPTBME[i] != 0){ indpp = i+1; ++ind; }
    }
  PPTBME.resize(indpp);
  return ind;
}

int J_Matrix_Elements::cut_nnJME()
{
  int indnn, ind = 0;
  for(int i = 0; i < int(NNTBME.size()); ++i)
    {
      if(NNTBME[i] != 0){ indnn = i+1; ++ind; }
    }
  NNTBME.resize(indnn);
  return ind;
}

int J_Matrix_Elements::cut_pnJME()
{
  int indpn, ind = 0;
  for(int i = 0; i < int(PNTBME.size()); ++i)
    {
      if(PNTBME[i] != 0){ indpn = i+1; ++ind; }
    }
  PNTBME.resize(indpn);
  return ind;
}


class M_Matrix_Elements{
private:
  int pleveltot, nleveltot;
  vector<double> POBME;
  vector<double> NOBME;
  vector<double> PPTBME;
  vector<double> NNTBME;
  vector<double> PNTBME;
public:
  M_Matrix_Elements(int, int); //constructor using leveltots
  void set_pMME(int, double);
  void set_nMME(int, double);
  double get_pMME(int) const;
  double get_nMME(int) const;
  void set_ppMME(int, int, int, int, double);
  void set_nnMME(int, int, int, int, double);
  void set_pnMME(int, int, int, int, double);
  double get_ppMME(int, int, int, int) const;
  double get_nnMME(int, int, int, int) const;
  double get_pnMME(int, int, int, int) const;
  int cut_ppMME();
  int cut_nnMME();
  int cut_pnMME();
};

M_Matrix_Elements::M_Matrix_Elements(int pl, int nl)
{
  pleveltot = pl;
  nleveltot = nl;
  POBME.resize(pl);
  NOBME.resize(nl);
  PPTBME.resize(int(0.5 * (pow(0.5*(pow(pl, 2.0) - pl), 2.0) + 0.5*(pow(pl, 2.0) - pl))));
  NNTBME.resize(int(0.5 * (pow(0.5*(pow(nl, 2.0) - nl), 2.0) + 0.5*(pow(nl, 2.0) - nl))));
  PNTBME.resize(int(0.5 * (pow(pl * nl, 2.0) + (pl * nl))));
}

void M_Matrix_Elements::set_pMME(int i, double ME){ POBME[i] = ME; }
double M_Matrix_Elements::get_pMME(int i) const
{ return POBME[i]; }

void M_Matrix_Elements::set_nMME(int i, double ME){ NOBME[i] = ME; }
double M_Matrix_Elements::get_nMME(int i) const
{ return NOBME[i]; }

void M_Matrix_Elements::set_ppMME(int m, int n, int l, int k, double ME)
{
  int branum, ketnum, vnum;
  double factor = 1.0;
  if(m > n){ factor *= -1.0; swap(m, n); }
  if(l > k){ factor *= -1.0; swap(l, k); }
  branum = number_level(m, n, pleveltot);
  ketnum = number_level(l, k, pleveltot);
  if(branum > ketnum){ swap(branum, ketnum); }
  vnum = number_shell(branum, ketnum, int(0.5*(pow(pleveltot, 2.0) - pleveltot)));
  PPTBME[vnum] = factor * ME;
}

void M_Matrix_Elements::set_nnMME(int m, int n, int l, int k, double ME)
{
  int branum, ketnum, vnum;
  double factor = 1.0;
  if(m > n){ factor *= -1.0; swap(m, n); }
  if(l > k){ factor *= -1.0; swap(l, k); }
  branum = number_level(m, n, nleveltot);
  ketnum = number_level(l, k, nleveltot);
  if(branum > ketnum){ swap(branum, ketnum); }
  vnum = number_shell(branum, ketnum, int(0.5*(pow(nleveltot, 2.0) - nleveltot)));
  NNTBME[vnum] = factor * ME;
}

void M_Matrix_Elements::set_pnMME(int m, int n, int l, int k, double ME)
{
  int branum, ketnum, vnum;
  branum = number_diff(m, n, nleveltot);
  ketnum = number_diff(l, k, nleveltot);
  if(branum > ketnum){ swap(branum, ketnum); }
  vnum = number_shell(branum, ketnum, pleveltot * nleveltot);
  PNTBME[vnum] = ME;
}

double M_Matrix_Elements::get_ppMME(int m, int n, int l, int k) const
{
  int branum, ketnum, vnum;
  double factor = 1.0;
  if(m > n){ factor *= -1.0; swap(m, n); }
  if(l > k){ factor *= -1.0; swap(l, k); }
  branum = number_level(m, n, pleveltot);
  ketnum = number_level(l, k, pleveltot);
  if(branum > ketnum){ swap(branum, ketnum); }
  vnum = number_shell(branum, ketnum, int(0.5*(pow(pleveltot, 2.0) - pleveltot)));
  return factor * PPTBME[vnum];
}

double M_Matrix_Elements::get_nnMME(int m, int n, int l, int k) const
{
  int branum, ketnum, vnum;
  double factor = 1.0;
  if(m > n){ factor *= -1.0; swap(m, n); }
  if(l > k){ factor *= -1.0; swap(l, k); }
  branum = number_level(m, n, nleveltot);
  ketnum = number_level(l, k, nleveltot);
  if(branum > ketnum){ swap(branum, ketnum); }
  vnum = number_shell(branum, ketnum, int(0.5*(pow(nleveltot, 2.0) - nleveltot)));
  return factor * NNTBME[vnum];
}

double M_Matrix_Elements::get_pnMME(int m, int n, int l, int k) const
{
  int branum, ketnum, vnum;
  branum = number_diff(m, n, nleveltot);
  ketnum = number_diff(l, k, nleveltot);
  if(branum > ketnum){ swap(branum, ketnum); }
  vnum = number_shell(branum, ketnum, pleveltot * nleveltot);
  return PNTBME[vnum];
}

int M_Matrix_Elements::cut_ppMME()
{
  int indpp, ind = 0;
  for(int i = 0; i < int(PPTBME.size()); ++i)
    {
      if(PPTBME[i] != 0){ indpp = i+1; ++ind; }
    }
  PPTBME.resize(indpp);
  return ind;
}

int M_Matrix_Elements::cut_nnMME()
{
  int indnn, ind = 0;
  for(int i = 0; i < int(NNTBME.size()); ++i)
    {
      if(NNTBME[i] != 0){ indnn = i+1; ++ind; }
    }
  NNTBME.resize(indnn);
  return ind;
}

int M_Matrix_Elements::cut_pnMME()
{
  int indpn, ind = 0;
  for(int i = 0; i < int(PNTBME.size()); ++i)
    {
      if(PNTBME[i] != 0){ indpn = i+1; ++ind; }
    }
  PNTBME.resize(indpn);
  return ind;
}

struct CCD{
  vector<double> PP_e;
  vector<double> NN_e;
  vector<double> PN_e;
  vector<double> PP_t;
  vector<double> NN_t;
  vector<double> PN_t;
  double CCDE;
};


//Initialize program from input file
Input_Parameters Get_Input_Parameters(const string &infile)
{ 
  std::cout << "Reading Input File" << endl;
  std::cout << "------------------" << endl;

  Input_Parameters Input; // Input information
  string path; // path for input file
  string line; // string for each file line
  ifstream filestream;	// input file
  int index; // count to keep track of input line
  size_t colon; // string element equal to ':'
  string substr; // substring of relevant information

  path = PATH + infile;
  filestream.open(path.c_str());
  if (!filestream.is_open()){ cerr << "Input file, " << path << ", does not exist" << endl; exit(1); }

  //find lines that start with '\*'
  index = 0;
  while (getline(filestream, line))
    { if(line[0] == '\\' && line[1] == '*')
	{  
	  ++index;
	  colon = line.find(':');
	  if( colon == line.size() - 1 ){ continue; };
	  substr = line.substr(colon + 2, line.size());
	  switch(index)
	    {
	    case 1:
	      Input.P = atoi(substr.c_str());
	      break;
	    case 2:
	      Input.N = atoi(substr.c_str());
	      break;
	    case 3:
	      Input.LevelScheme = substr;
	      break;
	    case 4:
	      Input.MatrixElements = substr;
	      if(Input.MatrixElements[Input.MatrixElements.size() - 2] == '_' && 
		 Input.MatrixElements[Input.MatrixElements.size() - 1] == 'M')
		{
		  Input.MatrixElements.erase(Input.MatrixElements.end() - 2, Input.MatrixElements.end());
		}
	      break;
	    } 
	}
      else{ continue; };
    }
  
  std::cout << "Level Scheme = " << Input.LevelScheme << endl;
  std::cout << "Matrix Elements = " << Input.MatrixElements << " " << Input.COMMatrixElements << endl << endl;
  
  return Input;
}


Model_Space Build_Model_Space(const Input_Parameters &Parameters)
{
  std::cout << "Reading Model Space File" << endl;
  std::cout << "------------------------" << endl;

  Model_Space Space; // Model Space information
  string fullpath; // Model Space file path
  string phline, number; // phline - string for each file line, number - string for first word of each line
  ifstream splevels; // Model space file
  istringstream phstream; // Stream of file line string
  size_t index1, index2; // Indicies for finding parameters among file lines
  int TotOrbs; // # total orbits
  int indp2, indn2, ind2; // indices for filling proton and neutron vectors
  int ind, n, l, j2, l2n, t; // initialize level index, n, l, and 2j from file. m is added later
  vector<int> indvec, nvec, lvec, j2vec, tvec, l2nvec; // initialize corresponding vectors
  double energy; // level energy
  stringstream a, b; // streams for naming shells
  string a1, b1, c1; // strings for naming shells
  string shell, tz; // strings for naming shells
  int shelllength; // number of single particle states for each shell
  
  fullpath = PATH + Parameters.LevelScheme + ".sp";
  splevels.open(fullpath.c_str());
  if (!splevels.is_open()){ cerr << "Level Scheme file does not exist" << endl; exit(1); }

  //Read Model Space file, get Mass, Oscillator Energy, Total number of shells
  getline(splevels, phline);
  phstream.str(phline);
  phstream >> number;
  while (number != "Mass"){ getline(splevels, phline); phstream.str(phline); phstream >> number; };
  index1 = phline.find_last_of(" \t");
  index2 = phline.find_last_of("0123456789");
  Space.A = std::atoi( phline.substr(index1 + 1, index2 - index1).c_str() );
  while (number != "Oscillator"){ getline(splevels, phline); phstream.str(phline); phstream >> number; };
  index1 = phline.find_last_of(" \t");
  index2 = phline.find_last_of("0123456789");
  Space.HOEnergy = std::atof( phline.substr(index1 + 1, index2 - index1).c_str() );
  while (number != "Total"){ getline(splevels, phline); phstream.str(phline); phstream >> number; };
  index1 = phline.find_last_of(" \t");
  index2 = phline.find_last_of("0123456789");
  TotOrbs = std::atoi( phline.substr(index1 + 1, index2 - index1).c_str() );
  std::cout << "A = " << Space.A << ", HOEnergy = " << Space.HOEnergy << ", TotOrbs = " << TotOrbs << endl;
  getline(splevels, phline);
  phstream.str(phline);
  phstream >> number;
  while (number != "Number:"){ getline(splevels, phline); phstream.str(phline); phstream >> number; };

  //read rest of level scheme parameters
  Space.spShells.resize(TotOrbs);
  Space.pShells.resize(TotOrbs);
  Space.nShells.resize(TotOrbs);

  Space.shelltot = 0;
  Space.pshelltot = 0;
  Space.nshelltot = 0;
  Space.leveltot = 0; //keep running counts of single-particle states
  Space.pleveltot = 0; //number of proton single particle states
  Space.nleveltot = 0; //number of neutron single particle states
  Space.max2J = -1;

  for(int i = 0; i < TotOrbs; ++i)
    {
      phstream.str(phline);
      phstream >> number >> ind >> n >> l >> j2 >> t >> l2n >> energy;

      //energy *= (1.0 - 1.0/Space.A); //subtract COM part

      //Name shell by orbital angular momentum
      switch(l)
	{
	case 0:
	  shell = "s";
	  break;
	case 1:
	  shell = "p";
	  break;
	case 2:
	  shell = "d";
	  break;
	case 3:
	  shell = "f";
	  break;
	case 4:
	  shell = "g";
	  break;
	case 5:
	  shell = "h";
	  break;
	default:
	  shell = "M";
	  break;
	}
      a << n;
      b << j2;
      a >> a1;
      b >> b1;

      if(t == -1)
	{
	  Space.spShells.pn_ind[Space.shelltot] = Space.pshelltot;
	  Space.pShells.sp_ind[Space.pshelltot] = i;
	  Space.pShells.pn_ind[Space.pshelltot] = Space.pshelltot;
	  Space.pShells.tz[Space.pshelltot] = t;
	  Space.pShells.n[Space.pshelltot] = n;
	  Space.pShells.l[Space.pshelltot] = l;
	  Space.pShells.j[Space.pshelltot] = 0.5 * j2;
	  Space.pShells.energy[Space.pshelltot] = energy;
	  tz = "P";
	  c1 = tz + a1 + shell + b1 + "/2";
	  Space.pShells.name[Space.pshelltot] = c1;
	  ++Space.pshelltot;
	  Space.pleveltot += j2 + 1;
	}
      else if(t == 1)
	{
	  Space.spShells.pn_ind[Space.shelltot] = Space.nshelltot;
	  Space.nShells.sp_ind[Space.nshelltot] = i;
	  Space.nShells.pn_ind[Space.nshelltot] = Space.nshelltot;
	  Space.nShells.tz[Space.nshelltot] = t;
	  Space.nShells.n[Space.nshelltot] = n;
	  Space.nShells.l[Space.nshelltot] = l;
	  Space.nShells.j[Space.nshelltot] = 0.5 * j2;
	  Space.nShells.energy[Space.nshelltot] = energy;
	  tz = "N";
	  c1 = tz + a1 + shell + b1 + "/2";
	  Space.pShells.name[Space.nshelltot] = c1;
	  ++Space.nshelltot;
	  Space.nleveltot += j2 + 1;
	}
      Space.spShells.sp_ind[Space.shelltot] = i;
      Space.spShells.tz[Space.shelltot] = t;
      Space.spShells.n[Space.shelltot] = n;
      Space.spShells.l[Space.shelltot] = l;
      Space.spShells.j[Space.shelltot] = 0.5 * j2;
      Space.spShells.energy[Space.shelltot] = energy;
      Space.spShells.name[Space.shelltot] = c1;
      ++Space.shelltot;
      Space.leveltot += j2 + 1;

      if(j2 > Space.max2J){ Space.max2J = j2; } 
      if( i < TotOrbs - 1 ){ getline(splevels, phline); };
    }
  splevels.close();
  
  //Resize M-Scheme single particle vectors to indtot
  Space.pShells.resize(Space.pshelltot);
  Space.nShells.resize(Space.nshelltot);
  Space.spStates.resize(Space.leveltot);
  Space.pStates.resize(Space.pleveltot);
  Space.nStates.resize(Space.nleveltot);

  //Fill M-Scheme single particle vectors
  ind2 = 0;
  indp2 = 0;
  indn2 = 0;
  for(int i = 0; i < Space.pshelltot; ++i)
    {
      shelllength = 2 * Space.pShells.j[i] + 1;
      Space.spShells.sp_levels[i].resize(shelllength);
      Space.spShells.pn_levels[i].resize(shelllength);
      Space.pShells.sp_levels[i].resize(shelllength);
      Space.pShells.pn_levels[i].resize(shelllength);
      for(int j = 0; j < shelllength; ++j)
	{
	  Space.pStates.sp_ind[indp2] = ind2;
	  Space.pStates.pn_ind[indp2] = indp2;
	  Space.pStates.tz[indp2] = Space.pShells.tz[i];
	  Space.pStates.n[indp2] = Space.pShells.n[i];
	  Space.pStates.l[indp2] = Space.pShells.l[i];
	  Space.pStates.j[indp2] = Space.pShells.j[i];
	  Space.pStates.m[indp2] = -1.0 * Space.pShells.j[i] + j;
	  Space.pStates.energy[indp2] = Space.pShells.energy[i];
	  Space.pStates.sp_shell[indp2] = Space.pShells.sp_ind[i];
	  Space.pStates.pn_shell[indp2] = Space.pShells.pn_ind[i];
	  Space.pShells.sp_levels[i][j] = ind2;
	  Space.pShells.pn_levels[i][j] = indp2;

	  Space.spStates.sp_ind[ind2] = ind2;
	  Space.spStates.pn_ind[ind2] = indp2;
	  Space.spStates.tz[ind2] = Space.pShells.tz[i];
	  Space.spStates.n[ind2] = Space.pShells.n[i];
	  Space.spStates.l[ind2] = Space.pShells.l[i];
	  Space.spStates.j[ind2] = Space.pShells.j[i];
	  Space.spStates.m[ind2] = -1.0 * Space.pShells.j[i] + j;
	  Space.spStates.energy[ind2] = Space.pShells.energy[i];
	  Space.spStates.sp_shell[ind2] = Space.pShells.sp_ind[i];
	  Space.spStates.pn_shell[ind2] = Space.pShells.pn_ind[i];
	  Space.spShells.sp_levels[i][j] = ind2;
	  Space.spShells.pn_levels[i][j] = indp2;
	  ++indp2;
	  ++ind2;
	}
    }
  for(int i = 0; i < Space.nshelltot; ++i)
    {
      shelllength = 2 * Space.nShells.j[i] + 1;
      Space.spShells.sp_levels[i + Space.pshelltot].resize(shelllength);
      Space.spShells.pn_levels[i + Space.pshelltot].resize(shelllength);
      Space.nShells.sp_levels[i].resize(shelllength);
      Space.nShells.pn_levels[i].resize(shelllength);
      for(int j = 0; j < shelllength; ++j)
	{
	  Space.nStates.sp_ind[indn2] = ind2;
	  Space.nStates.pn_ind[indn2] = indn2;
	  Space.nStates.tz[indn2] = Space.nShells.tz[i];
	  Space.nStates.n[indn2] = Space.nShells.n[i];
	  Space.nStates.l[indn2] = Space.nShells.l[i];
	  Space.nStates.j[indn2] = Space.nShells.j[i];
	  Space.nStates.m[indn2] = -1.0 * Space.nShells.j[i] + j;
	  Space.nStates.energy[indn2] = Space.nShells.energy[i];
	  Space.nStates.sp_shell[indn2] = Space.nShells.sp_ind[i];
	  Space.nStates.pn_shell[indn2] = Space.nShells.pn_ind[i];
	  Space.nShells.sp_levels[i][j] = ind2;
	  Space.nShells.pn_levels[i][j] = indn2;

	  Space.spStates.sp_ind[ind2] = ind2;	  
	  Space.spStates.pn_ind[ind2] = indn2;
	  Space.spStates.tz[ind2] = Space.nShells.tz[i];
	  Space.spStates.n[ind2] = Space.nShells.n[i];
	  Space.spStates.l[ind2] = Space.nShells.l[i];
	  Space.spStates.j[ind2] = Space.nShells.j[i];
	  Space.spStates.m[ind2] = -1.0 * Space.nShells.j[i] + j;
	  Space.spStates.energy[ind2] = Space.nShells.energy[i];
	  Space.spStates.sp_shell[ind2] = Space.nShells.sp_ind[i];
	  Space.spStates.pn_shell[ind2] = Space.nShells.pn_ind[i];
	  Space.spShells.sp_levels[i + Space.pshelltot][j] = ind2;
	  Space.spShells.pn_levels[i + Space.pshelltot][j] = indn2;
	  ++indn2;
	  ++ind2;
	}
    }

  return Space;
}


//Gives negative projection of v onto u
vector<double> projection(const vector<double> &u, const vector<double> &v)
{
  double innerprod;
  double norm;
  vector<double> proj;
  innerprod = 0.0;
  norm = 0.0;
  proj.resize(u.size());
  for(int i = 0; i < int(u.size()); ++i)
    { innerprod += v[i]*u[i]; norm += u[i]*u[i]; }
  for(int i = 0; i < int(u.size()); ++i)
    { proj[i] = -1.0*(innerprod/norm)*u[i]; }

  return proj;
}
      
//ignores existing holes and particles and fills them from protons and neutrons
void Separate_Particles_Holes(Single_Particle_States &States, const int &p, const int &n, const Model_Space &Space)
{
  int hcount, pcount, NN;
  NN = Space.leveltot;
  States.holes.resize(p + n);
  States.particles.resize(NN - (p + n));
  States.h_energies.resize(p + n);
  States.pt_energies.resize(NN - (p + n));
  for(int i = 0; i < p + n; ++i){ States.holes[i].assign(NN, 0); }
  for(int i = 0; i < NN - (p + n); ++i){ States.particles[i].assign(NN, 0); }

  //find lowest eigenvalues (p and n)
  hcount = 0;
  pcount = 0;
  for(int i = 0; i < Space.pleveltot; ++i)
    {
      if(i < p)
	{
	  for(int j = 0; j < NN; ++j)
	    {
	      if(j < Space.pleveltot){ States.holes[hcount][j] = States.protons[i][j]; }
	      else{ States.holes[hcount][j] = 0.0; }
	    }
	  States.h_energies[hcount] = States.p_energies[i];
	  ++hcount;
	}
      else
	{
	  for(int j = 0; j < NN; ++j)
	    {
	      if(j < Space.pleveltot){ States.particles[pcount][j] = States.protons[i][j]; }
	      else{ States.particles[pcount][j] = 0; }
	    }
	  States.pt_energies[pcount] = States.p_energies[i];
	  ++pcount;
	}
    }
  for(int i = 0; i < Space.nleveltot; ++i)
    {
      if(i < n)
	{
	  for(int j = 0; j < NN; ++j)
	    {
	      if(j >= Space.pleveltot){ States.holes[hcount][j] = States.neutrons[i][j - Space.pleveltot]; }
	      else{ States.holes[hcount][j] = 0; }
	    }
	  States.h_energies[hcount] = States.n_energies[i];
	  ++hcount;
	}
      else
	{
	  for(int j = 0; j < NN; ++j)
	    {
	      if(j >= Space.pleveltot){ States.particles[pcount][j] = States.neutrons[i][j - Space.pleveltot]; }
	      else{ States.particles[pcount][j] = 0; }
	    }
	  States.pt_energies[pcount] = States.n_energies[i];
	  ++pcount;
	}
    }
}



void GramSchmidt(vector<vector<double> > &Vectors)
{
  if(Vectors.size() == 0){ return; }
  vector<double> tempvec;
  double norm;
  int N = int(Vectors.size());
  int NN = int(Vectors[0].size());
  for(int i = 0; i < N; ++i)
    {
      norm = 0;
      for(int j = 0; j < NN; ++j){ norm += pow(Vectors[i][j], 2); }
      for(int j = 0; j < NN; ++j){ Vectors[i][j] /= sqrt(norm); }
      for(int j = i + 1; j < N; ++j)
	{
	  tempvec = projection(Vectors[i], Vectors[j]);
	  for(int k = 0; k < NN; ++k){ Vectors[j][k] += tempvec[k]; }
	}
    }
  for(int i = 0; i < N; ++i)
    {
      norm = 0.0;
      for(int j = 0; j < NN; ++j){ norm += pow(Vectors[i][j], 2); }
      for(int j = 0; j < NN; ++j)
	{ 
	  if(abs(Vectors[i][j]/sqrt(norm)) < 0.00001)
	    { norm -= pow(Vectors[i][j], 2); Vectors[i][j] = 0.0; }
	}
      for(int j = 0; j < NN; ++j){ Vectors[i][j] /= sqrt(norm); }
    }
}


Single_Particle_States Build_Single_Particle_States(const Input_Parameters &Parameters, const Model_Space &Space)
{
  std::cout << "Building Single-Particle States" << endl;
  std::cout << "-------------------------------" << endl;

  Single_Particle_States States; // Initial states
  int ind; // count for filling states
  double tempen; // temp energy for ordering states

  States.protons.resize(Space.pleveltot);
  States.neutrons.resize(Space.nleveltot);
  States.p_energies.resize(Space.pleveltot);
  States.n_energies.resize(Space.nleveltot);
  for(int i = 0; i < Space.pleveltot; ++i)
    { 
      States.protons[i].resize(Space.pleveltot);
      States.p_energies[i] = 0;
    }
  for(int i = 0; i < Space.nleveltot; ++i)
    { 
      States.neutrons[i].resize(Space.nleveltot);
      States.n_energies[i] = 0;
    }

  //SP states as initial vectors
  for(int i = 0; i < Space.pleveltot; ++i)
    {
      for(int j = 0; j < Space.pleveltot; ++j)
	{
	  if(i == j){ States.protons[i][j] = 1.0; }
	  else{ States.protons[i][j] = 0.0; }
	}
    }
  for(int i = 0; i < Space.nleveltot; ++i)
    {
      for(int j = 0; j < Space.nleveltot; ++j)
	{
	  if(i == j){ States.neutrons[i][j] = 1.0; }
	  else{ States.neutrons[i][j] = 0.0; }
	}
    }

  //Gram Schmidt 
  GramSchmidt(States.protons);
  GramSchmidt(States.neutrons);

  //Get Energies
  for(int i = 0; i < Space.pleveltot; ++i)
    {
      for(int j = 0; j < Space.pleveltot; ++j)
	{
	  States.p_energies[i] += pow(States.protons[i][j], 2.0) * Space.pStates.energy[j];	  
	}
    }
  for(int i = 0; i < Space.nleveltot; ++i)
    {
      for(int j = 0; j < Space.nleveltot; ++j)
	{
	  States.n_energies[i] += pow(States.neutrons[i][j], 2.0) * Space.nStates.energy[j];	  
	}
    }

  for(int i = 0; i < Space.pleveltot - 1; ++i)
    {
      ind = i;
      tempen = States.p_energies[i];
      for(int j = i + 1; j < Space.pleveltot; ++j)
	{
	  if(States.p_energies[j] < tempen){ tempen = States.p_energies[j]; ind = j; }
	}
      swap(States.p_energies[i], States.p_energies[ind]);
      swap(States.protons[i], States.protons[ind]);
    }

  for(int i = 0; i < Space.nleveltot - 1; ++i)
    {
      ind = i;
      tempen = States.n_energies[i];
      for(int j = i + 1; j < Space.nleveltot; ++j)
	{
	  if(States.n_energies[j] < tempen){ tempen = States.n_energies[j]; ind = j; }
	}
      swap(States.n_energies[i], States.n_energies[ind]);
      swap(States.neutrons[i], States.neutrons[ind]);
    }

  //Separate States
  Separate_Particles_Holes(States, Parameters.P, Parameters.N, Space);
  
  return States;

}



void Convert_To_M_Matrix_Elements(const string &MatrixElements, const Model_Space &Space, const J_Matrix_Elements &J_ME, M_Matrix_Elements &M_ME)
{
  std::cout << "Converting Matrix Elements from J-Scheme to M-Scheme" << endl;
  std::cout << "----------------------------------------------------" << endl;

  int ind; // count total number of M-Scheme matrix elements = 0;
  double p, q, r, s; // J-Scheme indices
  double CGC1, CGC2; // Clebsch-Gordon Coefficients for bra/ket-J coupling
  double factor1; // matrix element from specific coupling and combinatorial factors
  ofstream mschemefile; // file to print M-Scheme matrix elements
  double JME, tempME; // for accumulating matrix element
  double M, minJ, maxJ;
  
  for(int i = 0; i < Space.pleveltot; ++i){ M_ME.set_pMME(i, Space.pShells.energy[Space.pStates.pn_shell[i]]); }
  for(int i = 0; i < Space.nleveltot; ++i){ M_ME.set_nMME(i, Space.nShells.energy[Space.nStates.pn_shell[i]]); }

  ind = 0;
  //PP
  for(int m = 0; m < Space.pleveltot - 1; ++m)
    {
      for(int n = m + 1; n < Space.pleveltot; ++n)
	{
	  for(int l = m; l < Space.pleveltot - 1; ++l)
	    {
	      for(int k = l + 1; k < Space.pleveltot; ++k)
		{
		  tempME = 0.0;
		  p = Space.pStates.pn_shell[m];
		  q = Space.pStates.pn_shell[n];
		  r = Space.pStates.pn_shell[l];
		  s = Space.pStates.pn_shell[k];
		  M = Space.pStates.m[m] + Space.pStates.m[n];
		  if((m == l && n > k) || (M != Space.pStates.m[l] + Space.pStates.m[k])){ continue; }
		  maxJ = min(Space.pStates.j[m] + Space.pStates.j[n], Space.pStates.j[l] + Space.pStates.j[k]);
		  minJ = max(abs(Space.pStates.j[m] - Space.pStates.j[n]), abs(Space.pStates.j[l] - Space.pStates.j[k]));
		  minJ = max(minJ, abs(M));
		  for(int J = minJ; J <= maxJ; ++J)
		    { 
		      JME = J_ME.get_ppJME(p,q,r,s,J);
		      if(JME == 0){ continue; }
		      CGC1 = CGC(Space.pStates.j[m],Space.pStates.m[m],Space.pStates.j[n],Space.pStates.m[n],J,M);
		      CGC2 = CGC(Space.pStates.j[l],Space.pStates.m[l],Space.pStates.j[k],Space.pStates.m[k],J,M);
		      factor1 = 1.0;
		      if(Space.pStates.n[m] == Space.pStates.n[n] && Space.pStates.l[m] == Space.pStates.l[n]){ factor1 *= 2.0; }
		      if(Space.pStates.n[l] == Space.pStates.n[k] && Space.pStates.l[l] == Space.pStates.l[k]){ factor1 *= 2.0; }
		      factor1 = sqrt(factor1);
		      tempME += factor1 * CGC1 * CGC2 * JME;
		    }
		  M_ME.set_ppMME(m,n,l,k,tempME);
		}
	    }
	}
    }

  ind += M_ME.cut_ppMME(); //resize ppMME

  //NN
  for(int m = 0; m < Space.nleveltot - 1; ++m)
    {
      for(int n = m + 1; n < Space.nleveltot; ++n)
	{
	  for(int l = m; l < Space.nleveltot - 1; ++l)
	    {
	      for(int k = l + 1; k < Space.nleveltot; ++k)
		{
		  tempME = 0.0;
		  p = Space.nStates.pn_shell[m];
		  q = Space.nStates.pn_shell[n];
		  r = Space.nStates.pn_shell[l];
		  s = Space.nStates.pn_shell[k];
		  M = Space.nStates.m[m] + Space.nStates.m[n];
		  if((m == l && n > k) || (M != Space.nStates.m[l] + Space.nStates.m[k])){ continue; }
		  maxJ = min(Space.nStates.j[m] + Space.nStates.j[n], Space.nStates.j[l] + Space.nStates.j[k]);
		  minJ = max(abs(Space.nStates.j[m] - Space.nStates.j[n]), abs(Space.nStates.j[l] - Space.nStates.j[k]));
		  minJ = max(minJ, abs(M));
		  for(int J = minJ; J <= maxJ; ++J)
		    { 
		      JME = J_ME.get_nnJME(p,q,r,s,J);
		      if(JME == 0){ continue; }
		      CGC1 = CGC(Space.nStates.j[m],Space.nStates.m[m],Space.nStates.j[n],Space.nStates.m[n],J,M);
		      CGC2 = CGC(Space.nStates.j[l],Space.nStates.m[l],Space.nStates.j[k],Space.nStates.m[k],J,M);
		      factor1 = 1.0;
		      if(Space.nStates.n[m] == Space.nStates.n[n] && Space.nStates.l[m] == Space.nStates.l[n]){ factor1 *= 2.0; }
		      if(Space.nStates.n[l] == Space.nStates.n[k] && Space.nStates.l[l] == Space.nStates.l[k]){ factor1 *= 2.0; }
		      factor1 = sqrt(factor1);
		      tempME += factor1 * CGC1 * CGC2 * JME;
		    }
		  M_ME.set_nnMME(m,n,l,k,tempME);
		}
	    }
	}
    }

  ind += M_ME.cut_nnMME(); //resize nnMME

  //PN
  for(int m = 0; m < Space.pleveltot; ++m)
    {
      for(int n = 0; n < Space.nleveltot; ++n)
	{
	  for(int l = m; l < Space.pleveltot; ++l)
	    {
	      for(int k = 0; k < Space.nleveltot; ++k)
		{
		  tempME = 0.0;
		  p = Space.pStates.pn_shell[m];
		  q = Space.nStates.pn_shell[n];
		  r = Space.pStates.pn_shell[l];
		  s = Space.nStates.pn_shell[k];
		  M = Space.pStates.m[m] + Space.nStates.m[n];
		  if((m == l && n > k) || (M != Space.pStates.m[l] + Space.nStates.m[k])){ continue; }
		  maxJ = min(Space.pStates.j[m] + Space.nStates.j[n], Space.pStates.j[l] + Space.nStates.j[k]);
		  minJ = max(abs(Space.pStates.j[m] - Space.pStates.j[n]), abs(Space.pStates.j[l] - Space.nStates.j[k]));
		  minJ = max(minJ, abs(M));
		  for(int J = minJ; J <= maxJ; ++J)
		    { 
		      JME = J_ME.get_pnJME(p,q,r,s,J);
		      if(JME == 0){ continue; }
		      CGC1 = CGC(Space.pStates.j[m],Space.pStates.m[m],Space.nStates.j[n],Space.nStates.m[n],J,M);
		      CGC2 = CGC(Space.pStates.j[l],Space.pStates.m[l],Space.nStates.j[k],Space.nStates.m[k],J,M);
		      tempME += CGC1 * CGC2 * JME;
		    }
		  M_ME.set_pnMME(m,n,l,k,tempME);
		}
	    }
	}
    }

  ind += M_ME.cut_pnMME(); //resize pnMME

  // print M_ME to file
  mschemefile.open((PATH + MatrixElements + "_M.int").c_str());
  mschemefile << "Total number of twobody matx elements:" << "\t" << ind << "\n";
  for(int m = 0; m < Space.pleveltot - 1; ++m)
    {
      for(int n = m + 1; n < Space.pleveltot; ++n)
	{
	  for(int l = m; l < Space.pleveltot - 1; ++l)
	    {
	      for(int k = l + 1; k < Space.pleveltot; ++k)
		{
		  M = Space.pStates.m[m] + Space.pStates.m[n];
		  if((m == l && n > k) || (M != Space.pStates.m[l] + Space.pStates.m[k])){ continue; }
		  tempME = M_ME.get_ppMME(m,n,l,k);
		  if(tempME == 0){ continue; }
		  mschemefile << Space.pStates.sp_ind[m] + 1 << "\t" << Space.pStates.sp_ind[n] + 1 << "\t" 
			      << Space.pStates.sp_ind[l] + 1 << "\t" << Space.pStates.sp_ind[k] + 1 << "\t" 
			      << tempME << "\n";
		}
	    }
	}
    }
  for(int m = 0; m < Space.pleveltot; ++m)
    {
      for(int n = 0; n < Space.nleveltot; ++n)
	{
	  for(int l = m; l < Space.pleveltot; ++l)
	    {
	      for(int k = 0; k < Space.nleveltot; ++k)
		{
		  M = Space.pStates.m[m] + Space.nStates.m[n];
		  if((m == l && n > k) || (M != Space.pStates.m[l] + Space.nStates.m[k])){ continue; }
		  tempME = M_ME.get_pnMME(m,n,l,k);
		  if(tempME == 0){ continue; }
		  mschemefile << Space.pStates.sp_ind[m] + 1 << "\t" << Space.nStates.sp_ind[n] + 1 << "\t" 
			      << Space.pStates.sp_ind[l] + 1 << "\t" << Space.nStates.sp_ind[k] + 1 << "\t" 
			      << tempME << "\n";
		}
	    }
	}
    }
  for(int m = 0; m < Space.nleveltot - 1; ++m)
    {
      for(int n = m + 1; n < Space.nleveltot; ++n)
	{
	  for(int l = m; l < Space.nleveltot - 1; ++l)
	    {
	      for(int k = l + 1; k < Space.nleveltot; ++k)
		{
		  M = Space.nStates.m[m] + Space.nStates.m[n];
		  if((m == l && n > k) || (M != Space.nStates.m[l] + Space.nStates.m[k])){ continue; }
		  tempME = M_ME.get_nnMME(m,n,l,k);
		  if(tempME == 0){ continue; }
		  mschemefile << Space.nStates.sp_ind[m] + 1 << "\t" << Space.nStates.sp_ind[n] + 1 << "\t" 
			      << Space.nStates.sp_ind[l] + 1 << "\t" << Space.nStates.sp_ind[k] + 1 << "\t" 
			      << tempME << "\n";
		}
	    }
	}
    }

  mschemefile.close();
  
  std::cout << "Number of M-Scheme Two-Body Matrix Elements = " << ind << endl << endl;
  
}


void Read_J_Matrix_Elements(const string &MEfile, const Model_Space &Space, J_Matrix_Elements &ME)
{
  string fullpath1, fullpath2; // file path string and string with "_M" added (for J,M-Scheme)
  int NumElements; // number of ME read in from file
  string number; // string for first word of each line
  ifstream interaction;	// interaction file
  string interactionline; // interaction file line
  istringstream interactionstream; // stream of file line string
  size_t index1, index2; // indicies for finding parameters among file lines
  double TBME; // interaction two-body interaction ME and two-body COM ME
  int shell1, shell2, shell3, shell4, coupJ, coupT, par; // interaction file contents

  interaction.open(MEfile.c_str());
  if (!interaction.is_open()){ cerr << "Matrix Element file, " << MEfile << ", does not exist" << endl; exit(1); }
  getline(interaction, interactionline);
  interactionstream.str(interactionline);
  interactionstream >> number;
  while (number != "Total")
    { 
      getline(interaction, interactionline);
      interactionstream.str(interactionline);
      interactionstream >> number;
    }
  index1 = interactionline.find_first_of("0123456789");
  index2 = interactionline.find_last_of("0123456789");
  NumElements = std::atoi( interactionline.substr(index1, index2 - index1 + 1).c_str() );

  for(int i = 0; i < Space.pshelltot; ++i){ ME.set_pJME(i, Space.pShells.energy[i]); }
  for(int i = 0; i < Space.nshelltot; ++i){ ME.set_nJME(i, Space.nShells.energy[i]); }

  getline(interaction, interactionline);
  interactionstream.str(interactionline);
  interactionstream >> number;
  while (number != "Tz")
    {
      getline(interaction, interactionline);
      interactionstream.str(interactionline);
      interactionstream >> number;
    }
  for(int i = 0; i < NumElements; ++i)
    {
      getline(interaction, interactionline);
      istringstream(interactionline) >> coupT >> par >> coupJ >> shell1 >> shell2 >> shell3 >> shell4 >> TBME;
      coupJ *= 0.5;
      if(Space.spShells.tz[shell1-1] == 1 && Space.spShells.tz[shell2-1] == -1)
	{ swap(shell1, shell2); TBME *= pow(-1.0, Space.spShells.j[shell1-1] + Space.spShells.j[shell2-1] - coupJ); }
      if(Space.spShells.tz[shell3-1] == 1 && Space.spShells.tz[shell3-1] == -1)
	{ swap(shell3, shell4); TBME *= pow(-1.0, Space.spShells.j[shell3-1] + Space.spShells.j[shell4-1] - coupJ); }
      shell1 = Space.spShells.pn_ind[shell1 - 1];
      shell2 = Space.spShells.pn_ind[shell2 - 1];
      shell3 = Space.spShells.pn_ind[shell3 - 1];
      shell4 = Space.spShells.pn_ind[shell4 - 1];
      if(coupT == -1) //PP
	{
	  if(ME.get_ppJME(shell1, shell2, shell3, shell4, coupJ) == 0){ ME.set_ppJME(shell1, shell2, shell3, shell4, coupJ, TBME); }
	}
      else if(coupT == 1) //NN
	{
	  if(ME.get_nnJME(shell1, shell2, shell3, shell4, coupJ) == 0){ ME.set_nnJME(shell1, shell2, shell3, shell4, coupJ, TBME); }
	}
      else if(coupT == 0) //PN
	{
	  if(ME.get_pnJME(shell1, shell2, shell3, shell4, coupJ) == 0){ ME.set_pnJME(shell1, shell2, shell3, shell4, coupJ, TBME); }
	}	  
    }
  ME.cut_ppJME();
  ME.cut_nnJME();
  ME.cut_pnJME();

  interaction.close();
  
}

void Read_M_Matrix_Elements(const string &MEfile, const Model_Space &Space, M_Matrix_Elements &ME)
{
  int NumElements; // number of ME read in from file
  string number; // string for first word of each line
  ifstream interaction;	// interaction file
  string interactionline; // interaction file line
  istringstream interactionstream; // stream of file line string
  size_t index1, index2; // indicies for finding parameters among file lines
  double TBME; // interaction two-body interaction ME and two-body COM ME
  int shell1, shell2, shell3, shell4, coupT; // interaction file contents

  interaction.open(MEfile.c_str());
  if (!interaction.is_open()){ cerr << "Matrix Element file, " << MEfile << ", does not exist" << endl; exit(1); }
  getline(interaction, interactionline);
  interactionstream.str(interactionline);
  interactionstream >> number;
  while (number != "Total")
    { 
      getline(interaction, interactionline);
      interactionstream.str(interactionline);
      interactionstream >> number;
    }
  index1 = interactionline.find_first_of("0123456789");
  index2 = interactionline.find_last_of("0123456789");
  NumElements = std::atoi( interactionline.substr(index1, index2 - index1 + 1).c_str() );
  
  for(int i = 0; i < Space.pleveltot; ++i){ ME.set_pMME(i, Space.pStates.energy[i]); }
  for(int i = 0; i < Space.nleveltot; ++i){ ME.set_nMME(i, Space.nStates.energy[i]); }
  
  for(int i = 0; i < NumElements; ++i)
    {
      getline(interaction, interactionline);
      istringstream(interactionline) >> shell1 >> shell2 >> shell3 >> shell4 >> TBME;
      if(Space.spStates.tz[shell1-1] == 1 && Space.spStates.tz[shell2-1] == -1){ swap(shell1, shell2); TBME *= -1.0; }
      if(Space.spStates.tz[shell3-1] == 1 && Space.spStates.tz[shell4-1] == -1){ swap(shell3, shell4); TBME *= -1.0; }
      coupT = int(0.5*(Space.spStates.tz[shell1-1]+Space.spStates.tz[shell2-1]));
      shell1 = Space.spStates.pn_ind[shell1 - 1];
      shell2 = Space.spStates.pn_ind[shell2 - 1];
      shell3 = Space.spStates.pn_ind[shell3 - 1];
      shell4 = Space.spStates.pn_ind[shell4 - 1];
      if(coupT == -1) //PP
	{
	  if(ME.get_ppMME(shell1, shell2, shell3, shell4) == 0){ ME.set_ppMME(shell1, shell2, shell3, shell4, TBME); }
	}
      else if(coupT == 1) //NN
	{
	  if(ME.get_nnMME(shell1, shell2, shell3, shell4) == 0){ ME.set_nnMME(shell1, shell2, shell3, shell4, TBME); }
	}
      else if(coupT == 0) //PN
	{
	  if(ME.get_pnMME(shell1, shell2, shell3, shell4) == 0){ ME.set_pnMME(shell1, shell2, shell3, shell4, TBME); }
	}	  
    }
  ME.cut_ppMME();
  ME.cut_nnMME();
  ME.cut_pnMME();

  interaction.close();
  
}


M_Matrix_Elements Get_Matrix_Elements(const Input_Parameters &Parameters, const Model_Space &Space)
{
  std::cout << "Reading Matrix Elements" << endl;
  std::cout << "-----------------------" << endl;

  M_Matrix_Elements M_ME(Space.pleveltot, Space.nleveltot); // ME to be filled with interaction file
  string fullpath1, fullpath2; // file path string and string with "_M" added (for J,M-Scheme)
  size_t intsize; // size of MEfile string
  ifstream interaction;	// interaction file

  //open interaction file
  fullpath1 = PATH + Parameters.MatrixElements + ".int";
  fullpath2 = PATH + Parameters.MatrixElements + "_M.int";
  intsize = Parameters.MatrixElements.size();
  interaction.open(fullpath2.c_str()); // try m-scheme first
  if(interaction.is_open())
    {   
      interaction.close();
      Read_M_Matrix_Elements(fullpath2, Space, M_ME);
    }
  else if(Parameters.MatrixElements[intsize-2] == '_' && Parameters.MatrixElements[intsize-1] == 'M')
    {
      Read_M_Matrix_Elements(fullpath1, Space, M_ME);
    }
  else
    {
      J_Matrix_Elements J_ME(Space.pshelltot, Space.nshelltot, Space.pShells.j, Space.nShells.j);
      Read_J_Matrix_Elements(fullpath1, Space, J_ME);
      Convert_To_M_Matrix_Elements(Parameters.MatrixElements, Space, J_ME, M_ME);
    }

  //std::cout << "Number of One-Body Matrix Elements = " << M_ME.POBME.size()+M_ME.NOBME.size() << endl;
  //std::cout << "Number of J-Scheme Two-Body Matrix Elements = " << M_ME.PPTBME.size()+M_ME.NNTBME.size()+M_ME.PNTBME.size() << endl << endl;
  return M_ME;

};



Single_Particle_States Hartree_Fock_States2(const Input_Parameters &Parameters, const Single_Particle_States &States, const Model_Space &Space, const M_Matrix_Elements &ME)
{
  Single_Particle_States HF; // P + N States below Fermi level (initialize to States)
  int P, N; // Number of States and Number of State Components
  double error; // Energy error between iterations
  double Bshift; // level shift parameter
  int ind; // Index to keep track of iteration number
  double dens; // Factor for permuting matrix element indicies and density matrix element
  vector<double> pdensmat, ndensmat, pfock, nfock; // Fock Matrix
  vector<int> brakvec; // Braket for Matrix Element Permutation
  char jobz, uplo; // Parameters for Diagonalization, Multiplication
  int plda, nlda; // Parameter for Diagonalization
  vector<double> pw, pwork, nw, nwork; // EigenEnergy Vector and work vector
  int plwork, nlwork, pinfo, ninfo; // Parameters for Diagonaliztion
  vector<vector<double> > protons2, neutrons2, protons3, neutrons3; // lists for 2nd derivative of coefficients

  HF = States;
  P = Space.pleveltot;
  N = Space.nleveltot;
  brakvec.resize(4);
  jobz = 'V';
  uplo = 'U';
  plda = P;
  nlda = N;
  plwork = (3+2)*P;
  nlwork = (3+2)*N;
  Bshift = 100.0;

  protons3 = HF.protons;
  neutrons3 = HF.neutrons;

  ind = 0;
  error = 1000;
  ofstream resultsfile;
  resultsfile.open("HFresults.int");
  while((error > 0.000001 && ind < 100) || ind < 10)
    {
      std::cout << "!! " << ind << " !!" << endl;
      for(int i = 0; i < int(HF.holes.size()); ++i)
	{
	  //for(int j = 0; j < int(HF.holes[i].size()); ++j){ std::cout << HF.holes[i][j] << " "; }
	  std::cout << HF.h_energies[i] << " ";
	  resultsfile << HF.h_energies[i] << " ";
	}
      resultsfile << "\n";
      std::cout << endl;

      ++ind;
      error = 0;
      //Make Fock Matrix
      pfock.assign(P * P, 0);
      nfock.assign(N * N, 0);
      pdensmat.assign(P * P, 0);
      ndensmat.assign(N * N, 0);
      
      for(int i = 0; i < int(HF.holes.size()); ++i)
	{
	  double tempm = 0;
	  for(int j = 0; j < int(HF.holes[i].size()); ++j){ tempm += pow(HF.holes[i][j], 2.0) * Space.spStates.m[j]; }
	  std::cout << tempm << " ";
	}
      std::cout << endl;
      for(int i = 0; i < int(HF.holes.size()/2); ++i)
	{
	  for(int j = 0; j < int(HF.holes[i].size()/2); ++j){ std::cout << HF.holes[i][j] << " "; }
	  std::cout << endl;
	}

      //Build PP dens matrix
      for(int i = 0; i < P; ++i)
	{
	  for(int j = i; j < P; ++j)
	    {
	      dens = 0.0;
	      for(int beta = 0; beta < Parameters.P; ++beta){ dens += HF.protons[beta][i]*HF.protons[beta][j]; }
	      pdensmat[P * i + j] = dens;
	      if(i != j){ pdensmat[P * j + i] = dens; }
	    }
	}

      //Build NN dens matrix
      for(int i = 0; i < N; ++i)
	{
	  for(int j = i; j < N; ++j)
	    {
	      dens = 0.0;
	      for(int beta = 0; beta < Parameters.N; ++beta){ dens += HF.neutrons[beta][i]*HF.neutrons[beta][j]; }
	      ndensmat[N * i + j] = dens;
	      if(i != j){ ndensmat[N * j + i] = dens; }
	    }
	}

      //Build P-fock matrix with PP and PN matrix elements
      for(int m = 0; m < P; ++m)
	{
	  pfock[P * m + m] += ME.get_pMME(m); // Add diagonal elements to fock matrices
	  for(int l = m; l < P; ++l)
	    {
	      pfock[P * m + l] -= Bshift * pdensmat[P * m + l]; // Subtract level shift
	      if(m != l){ pfock[P * l + m] -= Bshift * pdensmat[P * m + l]; }
	      for(int n = 0; n < P; ++n)
		{
		  if(n == m){ continue; }
		  for(int k = 0; k < P; ++k)
		    {
		      if(k == l){ continue; }	      
		      pfock[P * m + l] += pdensmat[P * k + n] * ME.get_ppMME(m,n,l,k);
		      if(m != l){ pfock[P * l + m] += pdensmat[P * k + n] * ME.get_ppMME(m,n,l,k); }
		    }
		}
	      for(int n = 0; n < N; ++n)
		{
		  for(int k = 0; k < N; ++k)
		    {
		      pfock[P * m + l] += ndensmat[N * k + n] * ME.get_pnMME(m,n,l,k);
		      if(m != l){ pfock[P * l + m] += ndensmat[N * k + n] * ME.get_pnMME(m,n,l,k); }
		    }
		}
	    }
	}

      //Build N-fock matrix with NN and PN matrix elements
      for(int m = 0; m < N; ++m)
	{
	  nfock[N * m + m] += ME.get_nMME(m); // Add diagonal elements to fock matrices
	  for(int l = m; l < N; ++l)
	    {
	      nfock[N * m + l] -= Bshift * ndensmat[N * m + l];
	      if(m != l){ nfock[N * l + m] -= Bshift * ndensmat[N * m + l]; }
	      for(int n = 0; n < N; ++n)
		{
		  if(n == m){ continue; }
		  for(int k = 0; k < N; ++k)
		    {
		      if(k == l){ continue; }		      
		      nfock[N * m + l] += ndensmat[N * k + n] * ME.get_nnMME(m,n,l,k);
		      if(m != l){ nfock[N * l + m] += ndensmat[N * k + n] * ME.get_nnMME(m,n,l,k); }
		    }
		}
	      for(int n = 0; n < P; ++n)
		{
		  for(int k = 0; k < P; ++k)
		    {
		      //switch m,n and l,k for np order
		      nfock[N * m + l] += pdensmat[P * k + n] * ME.get_pnMME(n,m,k,l);
		      if(m != l){ nfock[N * l + m] += pdensmat[P * k + n] * ME.get_pnMME(n,m,k,l); }
		    }
		}
	    }
	}

      for(int i = 0; i < P*P; ++i){ if(abs(pfock[i]) < 1.0e-8){ pfock[i] = 0; } }
      for(int i = 0; i < N*N; ++i){ if(abs(nfock[i]) < 1.0e-8){ nfock[i] = 0; } }
      
      pw.assign(P,0);
      nw.assign(N,0);
      pwork.assign(plwork,0);
      nwork.assign(nlwork,0);

      if(P != 0){ dsyev_(&jobz, &uplo, &P, & *pfock.begin(), &plda, & *pw.begin(), & *pwork.begin(), &plwork, &pinfo); }
      if(N != 0){ dsyev_(&jobz, &uplo, &N, & *nfock.begin(), &nlda, & *nw.begin(), & *nwork.begin(), &nlwork, &ninfo); }
      
      //Add back Level-shift parameter
      for(int i = 0; i < Parameters.P; ++i){ pw[i] += Bshift; }
      for(int i = 0; i < Parameters.N; ++i){ nw[i] += Bshift; }

      //Get proton states, neutron states, and error
      for(int i = 0; i < P; ++i)
	{ 
	  error += abs(HF.p_energies[i] - pw[i]); 
	  HF.p_energies[i] = pw[i];
	  for(int j = 0; j < P; ++j){ HF.protons[i][j] = pfock[P * i + j]; }
	}
      for(int i = 0; i < N; ++i)
	{ 
	  error += abs(HF.n_energies[i] - nw[i]); 
	  HF.n_energies[i] = nw[i];
	  for(int j = 0; j < N; ++j){ HF.neutrons[i][j] = nfock[N * i + j]; }
	}

      GramSchmidt(HF.protons);
      GramSchmidt(HF.neutrons);

      Separate_Particles_Holes(HF, Parameters.P, Parameters.N, Space);
	
      std::cout << "error = " << error << endl << endl;
    }
  
  resultsfile.close();
  
  return HF;
  
}



M_Matrix_Elements Convert_To_HF_Matrix_Elements(const string &MatrixElements, const Single_Particle_States &States, const Model_Space &Space, const M_Matrix_Elements &ME)
{
  std::cout << "Converting Matrix Elements from M-Scheme to HF M-Scheme" << endl;
  std::cout << "----------------------------------------------------" << endl;

  M_Matrix_Elements HF_ME(Space.pleveltot, Space.nleveltot); // HF M-Scheme matrix elements
  int plength, pmatlength, nlength, nmatlength, pnmatlength; // max length of M-Scheme indicies, length of J_ME
  int num1, num2;
  double tempel;
  vector<double> PP_M, NN_M, PN_M, PP_V, NN_V, PN_V, PP_C, NN_C, PN_C; // Matrices of coefficients
  char transa, transb;
  double alpha, beta;
  int ind;
  ofstream mschemefile; // file to print M-Scheme matrix elements

  plength = Space.pleveltot;
  nlength = Space.nleveltot;

  // Fill OBME with HF single-particle energies
  for(int i = 0; i < plength; ++i){ HF_ME.set_pMME(i, States.p_energies[i]); }
  for(int i = 0; i < plength; ++i){ HF_ME.set_nMME(i, States.n_energies[i]); }

  std::cout << "!! 1" << endl;

  pmatlength = int(0.5 * (pow(plength, 2.0) - plength));
  PP_M.assign(pow(pmatlength, 2.0), 0);
  PP_V.assign(pow(pmatlength, 2.0), 0);
  PP_C.resize(pow(pmatlength, 2.0));
  for(int p = 0; p < plength-1; ++p)
    {
      for(int q = p + 1; q < plength; ++q)
	{
	  for(int alpha = 0; alpha < plength-1; ++alpha)
	    {
	      for(int gamma = alpha + 1; gamma < plength; ++gamma)
		{
		  num1 = number_level(p, q, plength);
		  num2 = number_level(alpha, gamma, plength);
		  tempel = States.protons[p][alpha] * States.protons[q][gamma];
		  PP_M[pmatlength * num1 + num2] = tempel;
		  if(num1 != num2){ PP_M[pmatlength * num2 + num1] = tempel; }
		}
	    }
	}
    }
  for(int alpha = 0; alpha < plength-1; ++alpha)
    {
      for(int gamma = alpha + 1; gamma < plength; ++gamma)
	{
	  for(int beta = alpha; beta < plength-1; ++beta)
	    {
	      for(int delta = beta + 1; delta < plength; ++delta)
		{
		  num1 = number_level(alpha, gamma, plength);
		  num2 = number_level(beta, delta, plength);
		  if(num1 > num2){ swap(num1, num2); }
		  tempel = ME.get_ppMME(alpha, gamma, beta, delta);
		  PP_V[pmatlength * num1 + num2] = tempel;
		  if(num1 != num2){ PP_V[pmatlength * num2 + num1] = tempel; }
		}
	    }
	}
    }

  std::cout << "!! 2" << endl;

  nmatlength = int(0.5 * (pow(nlength, 2.0) - nlength));
  NN_M.assign(pow(nmatlength, 2.0), 0);
  NN_V.assign(pow(nmatlength, 2.0), 0);
  NN_C.resize(pow(nmatlength, 2.0));
  for(int p = 0; p < nlength-1; ++p)
    {
      for(int q = p + 1; q < nlength; ++q)
	{
	  for(int alpha = 0; alpha < nlength-1; ++alpha)
	    {
	      for(int gamma = alpha + 1; gamma < nlength; ++gamma)
		{
		  num1 = number_level(p, q, nlength);
		  num2 = number_level(alpha, gamma, nlength);
		  tempel = States.neutrons[p][alpha] * States.neutrons[q][gamma];
		  NN_M[nmatlength * num1 + num2] = tempel;
		  if(num1 != num2){ NN_M[nmatlength * num2 + num1] = tempel; }
		}
	    }
	}
    }
  for(int alpha = 0; alpha < nlength-1; ++alpha)
    {
      for(int gamma = alpha + 1; gamma < nlength; ++gamma)
	{
	  for(int beta = alpha; beta < nlength-1; ++beta)
	    {
	      for(int delta = beta + 1; delta < nlength; ++delta)
		{
		  num1 = number_level(alpha, gamma, nlength);
		  num2 = number_level(beta, delta, nlength);
		  if(num1 > num2){ swap(num1, num2); }
		  tempel = ME.get_nnMME(alpha, gamma, beta, delta);
		  NN_V[pmatlength * num1 + num2] = tempel;
		  if(num1 != num2){ NN_V[pmatlength * num2 + num1] = tempel; }
		}
	    }
	}
    }

  std::cout << "!! 3" << endl;

  pnmatlength = plength * nlength;
  PN_M.assign(pow(pnmatlength, 2.0), 0);
  PN_V.assign(pow(pnmatlength, 2.0), 0);
  PN_C.resize(pow(pnmatlength, 2.0));
  for(int p = 0; p < plength; ++p)
    {
      for(int q = 0; q < nlength; ++q)
	{
	  for(int alpha = 0; alpha < plength; ++alpha)
	    {
	      for(int gamma = 0; gamma < nlength; ++gamma)
		{
		  num1 = number_diff(p, q, nlength);
		  num2 = number_diff(alpha, gamma, nlength);
		  tempel = States.protons[p][alpha] * States.neutrons[q][gamma];
		  PN_M[pnmatlength * num1 + num2] = tempel;
		  if(num1 != num2){ PN_M[pnmatlength * num2 + num1] = tempel; }
		}
	    }
	}
    }
  for(int alpha = 0; alpha < plength; ++alpha)
    {
      for(int gamma = 0; gamma < nlength; ++gamma)
	{
	  for(int beta = alpha; beta < plength; ++beta)
	    {
	      for(int delta = 0; delta < nlength; ++delta)
		{
		  num1 = number_diff(alpha, gamma, nlength);
		  num2 = number_diff(beta, delta, nlength);
		  if(num1 > num2){ swap(num1, num2); }
		  tempel = ME.get_pnMME(alpha, gamma, beta, delta);
		  PN_V[pnmatlength * num1 + num2] = tempel;
		  if(num1 != num2){ PN_V[pnmatlength * num2 + num1] = tempel; }
		}
	    }
	}
    }

  std::cout << "!! 4" << endl;

  transa = 'N';
  transb = 'N';
  alpha = 1.0;
  beta = 0.0;

  if(pmatlength != 0)
    {
      dgemm_(&transa, &transb, &pmatlength, &pmatlength, &pmatlength, &alpha, & *PP_V.begin(), &pmatlength, & *PP_M.begin(), &pmatlength, &beta, & *PP_C.begin(), &pmatlength );
      PP_V = PP_C;
      dgemm_(&transa, &transb, &pmatlength, &pmatlength, &pmatlength, &alpha, & *PP_M.begin(), &pmatlength, & *PP_V.begin(), &pmatlength, &beta, & *PP_C.begin(), &pmatlength );
      PP_V = PP_C;
    }

  std::cout << "!! 5" << endl;

  if(nmatlength != 0)
    {
      dgemm_(&transa, &transb, &nmatlength, &nmatlength, &nmatlength, &alpha, & *NN_V.begin(), &nmatlength, & *NN_M.begin(), &nmatlength, &beta, & *NN_C.begin(), &nmatlength );
      NN_V = NN_C;
      dgemm_(&transa, &transb, &nmatlength, &nmatlength, &nmatlength, &alpha, & *NN_M.begin(), &nmatlength, & *NN_V.begin(), &nmatlength, &beta, & *NN_C.begin(), &nmatlength );
      NN_V = NN_C;
    }

  std::cout << "!! 6" << endl;

  if(pnmatlength != 0)
    {
      dgemm_(&transa, &transb, &pnmatlength, &pnmatlength, &pnmatlength, &alpha, & *PN_V.begin(), &pnmatlength, & *PN_M.begin(), &pnmatlength, &beta, & *PN_C.begin(), &pnmatlength );
      PN_V = PN_C;
      dgemm_(&transa, &transb, &pnmatlength, &pnmatlength, &pnmatlength, &alpha, & *PN_M.begin(), &pnmatlength, & *PN_V.begin(), &pnmatlength, &beta, & *PN_C.begin(), &pnmatlength );
      PN_V = PN_C;
    }

  std::cout << "!! 7" << endl;

  ind = 0;
  for(int p = 0; p < plength-1; ++p)
    {
      for(int q = p + 1; q < plength; ++q)
	{
	  for(int r = p; r < plength-1; ++r)
	    {
	      for(int s = r + 1; s < plength; ++s)
		{
		  num1 = number_level(p, q, plength);
		  num2 = number_level(r, s, plength);
		  if(num1 > num2){ swap(num1, num2); }
		  if(PP_V[pmatlength * num1 + num2] != 0)
		    { 
		      HF_ME.set_ppMME(p, q, r, s, PP_V[pmatlength * num1 + num2]);
		      ++ind;
		    }
		}
	    }
	}
    }
  HF_ME.cut_ppMME();

  std::cout << "!! 8" << endl;

  for(int p = 0; p < nlength-1; ++p)
    {
      for(int q = p + 1; q < nlength; ++q)
	{
	  for(int r = p; r < nlength-1; ++r)
	    {
	      for(int s = r + 1; s < nlength; ++s)
		{
		  num1 = number_level(p, q, nlength);
		  num2 = number_level(r, s, nlength);
		  if(num1 > num2){ swap(num1, num2); }
		  if(NN_V[nmatlength * num1 + num2] != 0)
		    { 
		      HF_ME.set_nnMME(p, q, r, s, NN_V[nmatlength * num1 + num2]);
		      ++ind;
		    }
		}
	    }
	}
    }
  HF_ME.cut_nnMME();

  std::cout << "!! 9" << endl;

  for(int p = 0; p < plength; ++p)
    {
      for(int q = 0; q < nlength; ++q)
	{
	  for(int r = p; r < plength; ++r)
	    {
	      for(int s = 0; s < nlength; ++s)
		{
		  num1 = number_diff(p, q, nlength);
		  num2 = number_diff(r, s, nlength);
		  if(num1 > num2){ swap(num1, num2); }
		  if(PN_V[pnmatlength * num1 + num2] != 0)
		    { 
		      HF_ME.set_pnMME(p, q, r, s, PN_V[pnmatlength * num1 + num2]);
		      ++ind;
		    }
		}
	    }
	}
    }
  HF_ME.cut_pnMME();

  std::cout << "!! 10" << endl;

  // print M_ME to file
  mschemefile.open((PATH + MatrixElements + "_HF.int").c_str());
  mschemefile << "Total number of twobody matx elements:" << "\t" << ind << "\n";
  for(int m = 0; m < plength - 1; ++m)
    {
      for(int n = m + 1; n < plength; ++n)
	{
	  for(int l = m; l < plength - 1; ++l)
	    {
	      for(int k = l + 1; k < plength; ++k)
		{
		  if(HF_ME.get_ppMME(m, n, l, k) == 0){ continue; }
		  mschemefile << Space.pStates.sp_ind[m] + 1 << "\t" << Space.pStates.sp_ind[n] + 1 << "\t" 
			      << Space.pStates.sp_ind[l] + 1 << "\t" << Space.pStates.sp_ind[k] + 1 << "\t" 
			      << HF_ME.get_ppMME(m, n, l, k) << "\n";
		}
	    }
	}
    }
  for(int m = 0; m < plength; ++m)
    {
      for(int n = 0; n < nlength; ++n)
	{
	  for(int l = m; l < plength; ++l)
	    {
	      for(int k = 0; k < nlength; ++k)
		{
		  if(HF_ME.get_pnMME(m, n, l, k) == 0){ continue; }
		  mschemefile << Space.pStates.sp_ind[m] + 1 << "\t" << Space.nStates.sp_ind[n] + 1 << "\t" 
			      << Space.pStates.sp_ind[l] + 1 << "\t" << Space.nStates.sp_ind[k] + 1 << "\t" 
			      << HF_ME.get_pnMME(m, n, l, k) << "\n";
		}
	    }
	}
    }
  for(int m = 0; m < nlength - 1; ++m)
    {
      for(int n = m + 1; n < nlength; ++n)
	{
	  for(int l = m; l < nlength - 1; ++l)
	    {
	      for(int k = l + 1; k < nlength; ++k)
		{
		  if(HF_ME.get_nnMME(m, n, l, k) == 0){ continue; }
		  mschemefile << Space.nStates.sp_ind[m] + 1 << "\t" << Space.nStates.sp_ind[n] + 1 << "\t" 
			      << Space.nStates.sp_ind[l] + 1 << "\t" << Space.nStates.sp_ind[k] + 1 << "\t" 
			      << HF_ME.get_nnMME(m, n, l, k) << "\n";
		}
	    }
	}
    }

  mschemefile.close();

  return HF_ME;

  }


int main(int argc, char* argv[])
{ 
  //clock_t t1, t2;    	//initialize program clock
  //t1 = clock();

  string inputfile = "input.dat";
  Input_Parameters Parameters = Get_Input_Parameters(inputfile);
  Model_Space Space = Build_Model_Space(Parameters);
  M_Matrix_Elements ME = Get_Matrix_Elements(Parameters, Space);
  Single_Particle_States States = Build_Single_Particle_States(Parameters, Space);
  Single_Particle_States HFStates = Hartree_Fock_States2(Parameters, States, Space, ME);
  M_Matrix_Elements HF_ME = Convert_To_HF_Matrix_Elements(Parameters.MatrixElements, HFStates, Space, ME);
  
  /*double hfenergy = 0.0;
  int num, braketnum;
  for(int i = 0; i < int(HFStates.holes.size()); ++i)
    {
      hfenergy += HFStates.h_energies[i];
    }
  for(int i = 0; i < Parameters.P - 1; ++i)
    {
      for(int j = i + 1; j < Parameters.P; ++j)
	{
	  num = number_level_same(i, j, Space.pleveltot);
	  braketnum = number_shell_same(num, num, int(0.5 * (pow(Space.pleveltot, 2.0) - Space.pleveltot)));
	  hfenergy -= HF_ME.PPTBME[braketnum];
	}
    }
  for(int i = 0; i < Parameters.P; ++i)
    {
      for(int j = 0; j < Parameters.N; ++j)
	{
	  num = number_diff(i, j, Space.nleveltot);
	  braketnum = number_shell_same(num, num, Space.pleveltot * Space.nleveltot);
	  hfenergy -= HF_ME.PNTBME[braketnum];
	}
    }
  for(int i = 0; i < Parameters.N - 1; ++i)
    {
      for(int j = i + 1; j < Parameters.N; ++j)
	{
	  num = number_level_same(i, j, Space.pleveltot);
	  braketnum = number_shell_same(num, num, int(0.5 * (pow(Space.nleveltot, 2.0) - Space.nleveltot)));
	  hfenergy -= HF_ME.NNTBME[braketnum];
	}
    }

    std::cout << endl << endl << "HF Energy = " << hfenergy + CC.CCDE << endl;*/
		  	  
  //t2 = clock();
  //float diff((float)t2 - (float)t1);
  //std::cout << diff / CLOCKS_PER_SEC << "sec" << endl;

  int a;
  std::cin >> a;

  return 0;

}
