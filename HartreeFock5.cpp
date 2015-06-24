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
#define RM_dgemm(a, b, c, d, e, f, g, h, i, j, k, l, m) dgemm_(b, a, d, c, e, f, i, d, g, j, k, l, d)

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
  int Pocc, Nocc; //number of occupied proton and neutron shells
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
  vector<double> p_j;
  vector<double> n_j;
  vector<int> p_l;
  vector<int> n_l;
  vector<double> h_j;
  vector<double> pt_j;
  vector<double> h_l;
  vector<double> pt_l;
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

  Space.Pocc = 0;
  Space.Nocc = 0;

  for(int i = 0; i < TotOrbs; ++i)
    {
      phstream.str(phline);
      phstream >> number >> ind >> n >> l >> j2 >> t >> l2n >> energy;

      //energy *= (1.0 - 1.0/Space.A); //subtract COM part
      energy *= 0.5;

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
	  if(Space.pleveltot == Parameters.P){ Space.Pocc = Space.pshelltot; }
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
	  if(Space.nleveltot == Parameters.N){ Space.Nocc = Space.nshelltot; }
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
  
  if(Space.Pocc == 0 || Space.Nocc == 0){ cerr << "Not Closed-Shell" << endl; exit(1); }

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
  NN = Space.shelltot;
  States.holes.resize(p + n);
  States.particles.resize(NN - (p + n));
  States.h_energies.resize(p + n);
  States.pt_energies.resize(NN - (p + n));
  States.h_j.resize(p + n);
  States.pt_j.resize(NN - (p + n));
  States.h_l.resize(p + n);
  States.pt_l.resize(NN - (p + n));
  for(int i = 0; i < p + n; ++i){ States.holes[i].assign(NN, 0); }
  for(int i = 0; i < NN - (p + n); ++i){ States.particles[i].assign(NN, 0); }

  //find lowest eigenvalues (p and n)
  hcount = 0;
  pcount = 0;
  for(int i = 0; i < Space.pshelltot; ++i)
    {
      if(i < p)
	{
	  for(int j = 0; j < NN; ++j)
	    {
	      if(j < Space.pshelltot){ //std::cout << endl << j << " " << Space.pshelltot << endl; 
		States.holes[hcount][j] = States.protons[i][j]; }
	      else{ States.holes[hcount][j] = 0.0; }
	    }
	  States.h_energies[hcount] = States.p_energies[i];
	  States.h_j[hcount] = States.p_j[i];
	  States.h_l[hcount] = States.p_l[i];
	  ++hcount;
	}
      else
	{
	  for(int j = 0; j < NN; ++j)
	    {
	      if(j < Space.pshelltot){ States.particles[pcount][j] = States.protons[i][j]; }
	      else{ States.particles[pcount][j] = 0; }
	    }
	  States.pt_energies[pcount] = States.p_energies[i];
	  States.pt_j[pcount] = States.p_j[i];
	  States.pt_l[pcount] = States.p_l[i];
	  ++pcount;
	}
    }
  for(int i = 0; i < Space.nshelltot; ++i)
    {
      if(i < n)
	{
	  for(int j = 0; j < NN; ++j)
	    {
	      if(j >= Space.pshelltot){ States.holes[hcount][j] = States.neutrons[i][j - Space.pshelltot]; }
	      else{ States.holes[hcount][j] = 0; }
	    }
	  States.h_energies[hcount] = States.n_energies[i];
	  States.h_j[hcount] = States.n_j[i];
	  States.h_l[hcount] = States.n_l[i];
	  ++hcount;
	}
      else
	{
	  for(int j = 0; j < NN; ++j)
	    {
	      if(j >= Space.pshelltot){ States.particles[pcount][j] = States.neutrons[i][j - Space.pshelltot]; }
	      else{ States.particles[pcount][j] = 0; }
	    }
	  States.pt_energies[pcount] = States.n_energies[i];
	  States.pt_j[pcount] = States.n_j[i];
	  States.pt_l[pcount] = States.n_l[i];
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
  //int ind; // count for filling states
  //double tempen; // temp energy for ordering states

  States.protons.resize(Space.pshelltot);
  States.neutrons.resize(Space.nshelltot);
  States.p_energies.resize(Space.pshelltot);
  States.n_energies.resize(Space.nshelltot);
  States.p_j.resize(Space.pshelltot);
  States.n_j.resize(Space.nshelltot);
  States.p_l.resize(Space.pshelltot);
  States.n_l.resize(Space.nshelltot);
  for(int i = 0; i < Space.pshelltot; ++i)
    { 
      States.protons[i].resize(Space.pshelltot);
      States.p_energies[i] = 0;
    }
  for(int i = 0; i < Space.nshelltot; ++i)
    { 
      States.neutrons[i].resize(Space.nshelltot);
      States.n_energies[i] = 0;
    }

  //SP states as initial vectors
  for(int i = 0; i < Space.pshelltot; ++i)
    {
      for(int j = 0; j < Space.pshelltot; ++j)
	{
	  if(i == j)
	    {
	      States.protons[i][j] = 1.0;
	      States.p_j[i] = Space.pShells.j[j];
	      States.p_l[i] = Space.pShells.l[j];
	    }
	  else{ States.protons[i][j] = 0.0; }
	}
    }
  for(int i = 0; i < Space.nshelltot; ++i)
    {
      for(int j = 0; j < Space.nshelltot; ++j)
	{
	  if(i == j)
	    {
	      States.neutrons[i][j] = 1.0;
	      States.n_j[i] = Space.nShells.j[j];
	      States.n_l[i] = Space.nShells.l[j];
	    }
	  else{ States.neutrons[i][j] = 0.0; }
	}
    }

  //Get Energies
  for(int i = 0; i < Space.pshelltot; ++i)
    {
      States.p_energies[i] = Space.pShells.energy[i];
    }
  for(int i = 0; i < Space.nshelltot; ++i)
    {
      States.n_energies[i] = Space.nShells.energy[i];
    }

  // Order states by energy
  /*for(int i = 0; i < Space.pshelltot - 1; ++i)
    {
      ind = i;
      tempen = States.p_energies[i];
      for(int j = i + 1; j < Space.pshelltot; ++j)
	{
	  if(States.p_energies[j] < tempen){ tempen = States.p_energies[j]; ind = j; }
	}
      swap(States.p_energies[i], States.p_energies[ind]);
      swap(States.protons[i], States.protons[ind]);
      swap(States.p_j[i], States.p_j[ind]);
      swap(States.p_l[i], States.p_l[ind]);
    }

  for(int i = 0; i < Space.nshelltot - 1; ++i)
    {
      ind = i;
      tempen = States.n_energies[i];
      for(int j = i + 1; j < Space.nshelltot; ++j)
	{
	  if(States.n_energies[j] < tempen){ tempen = States.n_energies[j]; ind = j; }
	}
      swap(States.n_energies[i], States.n_energies[ind]);
      swap(States.neutrons[i], States.neutrons[ind]);
      swap(States.n_j[i], States.n_j[ind]);
      swap(States.n_l[i], States.n_l[ind]);
      }*/

  //Separate States
  Separate_Particles_Holes(States, Space.Pocc, Space.Nocc, Space);
  
  return States;

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
  double TBME, hom, r2, p2; // interaction two-body interaction ME and two-body COM ME
  int shell1, shell2, shell3, shell4, coupJ, coupT, par; // interaction file contents
  int int1 = 0;

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
      istringstream(interactionline) >> coupT >> par >> coupJ >> shell1 >> shell2 >> shell3 >> shell4 >> TBME >> hom >> r2 >> p2;
      coupJ *= 0.5;
      //TBME = TBME - (Space.HOEnergy/Space.A) * (hom + r2 + p2);
      //renormalize to remove state-dependent term for HF
      if((shell1 == shell2 || shell3 == shell4) && coupJ%2 != 0){ continue; }
      if(shell1 == shell2){ TBME *= sqrt(2.0); }
      if(shell3 == shell4){ TBME *= sqrt(2.0); }
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
	  if(ME.get_ppJME(shell1, shell2, shell3, shell4, coupJ) == 0){ ++int1; ME.set_ppJME(shell1, shell2, shell3, shell4, coupJ, TBME); }
	}
      else if(coupT == 1) //NN
	{
	  if(ME.get_nnJME(shell1, shell2, shell3, shell4, coupJ) == 0){ ++int1; ME.set_nnJME(shell1, shell2, shell3, shell4, coupJ, TBME); }
	}
      else if(coupT == 0) //PN
	{
	  if(ME.get_pnJME(shell1, shell2, shell3, shell4, coupJ) == 0){ ++int1; ME.set_pnJME(shell1, shell2, shell3, shell4, coupJ, TBME); }
	}	  
    }
  ME.cut_ppJME();
  ME.cut_nnJME();
  ME.cut_pnJME();

  std::cout << endl << endl << "!!!!" << int1 << "!!!!" << endl << endl;

  interaction.close();
  
}


J_Matrix_Elements Get_J_Matrix_Elements(const Input_Parameters &Parameters, const Model_Space &Space)
{
  std::cout << "Reading J-Matrix Elements" << endl;
  std::cout << "-----------------------" << endl;

  J_Matrix_Elements J_ME(Space.pshelltot, Space.nshelltot, Space.pShells.j, Space.nShells.j); // ME to be filled with interaction file
  string fullpath1, fullpath2; // file path string and string with "_M" added (for J,M-Scheme)
  size_t intsize; // size of MEfile string
  ifstream interaction;	// interaction file

  //open interaction file
  fullpath1 = Parameters.MatrixElements;
  fullpath1 = PATH + Parameters.MatrixElements + ".int";
  fullpath2 = PATH + Parameters.MatrixElements + "_M.int";
  intsize = Parameters.MatrixElements.size();
  if(Parameters.MatrixElements[intsize-2] == '_' && Parameters.MatrixElements[intsize-1] == 'M')
    {
      fullpath1.resize(intsize-2);
      fullpath2 = PATH + fullpath1 + ".int";
      Read_J_Matrix_Elements(fullpath2, Space, J_ME);
    }
  else
    {
      fullpath2 = PATH + fullpath1 + ".int";
      Read_J_Matrix_Elements(fullpath1, Space, J_ME);
    }

  //std::cout << "Number of One-Body Matrix Elements = " << M_ME.POBME.size()+M_ME.NOBME.size() << endl;
  //std::cout << "Number of J-Scheme Two-Body Matrix Elements = " << M_ME.PPTBME.size()+M_ME.NNTBME.size()+M_ME.PNTBME.size() << endl << endl;
  return J_ME;

};


Single_Particle_States Hartree_Fock_States2(const Input_Parameters &Parameters, const Single_Particle_States &States, const Model_Space &Space, const J_Matrix_Elements &ME)
{
  Single_Particle_States HF; // P + N States below Fermi level (initialize to States)
  int P, N; // Number of States and Number of State Components
  double error; // Energy error between iterations
  double Bshift; // level shift parameter
  int ind; // Index to keep track of iteration number
  //double dens; // Factor for permuting matrix element indicies and density matrix element
  vector<double> pdensmat, ndensmat, pfock, nfock; // Fock Matrix
  vector<int> brakvec; // Braket for Matrix Element Permutation
  char jobz, uplo; // Parameters for Diagonalization, Multiplication
  int plda, nlda; // Parameter for Diagonalization
  vector<double> pw, pwork, nw, nwork; // EigenEnergy Vector and work vector
  int plwork, nlwork, pinfo, ninfo; // Parameters for Diagonaliztion
  double term, term2;
  int minj, maxj;
  int tempint = 0;

  HF = States;
  P = Space.pshelltot;
  N = Space.nshelltot;
  brakvec.resize(4);
  jobz = 'V';
  uplo = 'U';
  plda = P;
  nlda = N;
  plwork = (3+2)*P;
  nlwork = (3+2)*N;
  Bshift = 0.0;

  ind = 0;
  error = 1000;
  while((error > 1e-8 && ind < 10) || ind < 10)
    {
      /*std::cout << "!! " << ind << " !!" << endl;
      for(int i = 0; i < int(HF.holes.size()); ++i)
	{
	  std::cout << HF.h_energies[i] << " ";
	}
      std::cout << endl;
      for(int i = 0; i < int(HF.holes.size()); ++i)
	{
	  std::cout << HF.h_j[i] << " ";
	}
      std::cout << endl;
      for(int i = 0; i < int(HF.holes.size()); ++i)
	{
	  std::cout << HF.h_l[i] << " ";
	}
	std::cout << endl;*/


      ++ind;
      error = 0;
      //Make Fock Matrix
      pfock.assign(P * P, 0);
      nfock.assign(N * N, 0);
      pdensmat.assign(P * P, 0);
      ndensmat.assign(N * N, 0);
      
      /*for(int i = 0; i < int(HF.holes.size()); ++i)
	{
	  for(int j = 0; j < int(HF.holes[i].size()); ++j){ std::cout << HF.holes[i][j] << " "; }
	  std::cout << endl;
	}

      for(int i = 0; i < int(HF.protons.size()); ++i)
	{
	  for(int j = 0; j < int(HF.protons[i].size()); ++j){ std::cout << HF.protons[i][j] << " "; }
	  std::cout << endl;
	}
      std::cout << endl;
      for(int i = 0; i < int(HF.neutrons.size()); ++i)
	{
	  for(int j = 0; j < int(HF.neutrons[i].size()); ++j){ std::cout << HF.neutrons[i][j] << " "; }
	  std::cout << endl;
	  }*/

      //Build PP dens matrix
      /*for(int i = 0; i < P; ++i)
	{
	  for(int j = i; j < P; ++j)
	    {
	      if(Space.pShells.j[i] != Space.pShells.j[j] || Space.pShells.l[i] != Space.pShells.l[j]){ continue; }
	      dens = 0.0;
	      for(int beta = 0; beta < Space.Pocc; ++beta)
		{
		  if(Space.pShells.j[i] != HF.p_j[beta] || Space.pShells.l[i] != HF.p_l[beta]){ continue; }
		  dens += HF.protons[beta][i] * HF.protons[beta][j];
		}
	      pdensmat[P * i + j] = dens;
	      if(i != j){ pdensmat[P * j + i] = dens; }
	    }
	}
      
      //Build NN dens matrix
      for(int i = 0; i < N; ++i)
	{
	  for(int j = i; j < N; ++j)
	    {
	      if(Space.nShells.j[i] != Space.nShells.j[j] || Space.nShells.l[i] != Space.nShells.l[j]){ continue; }
	      dens = 0.0;
	      for(int beta = 0; beta < Space.Nocc; ++beta)
		{
		  if(Space.nShells.j[i] != HF.n_j[beta] || Space.nShells.l[i] != HF.n_l[beta]){ continue; }
		  dens += HF.neutrons[beta][i] * HF.neutrons[beta][j];
		}
	      ndensmat[N * i + j] = dens;
	      if(i != j){ ndensmat[N * j + i] = dens; }
	    }
	    }*/

      //Build P-fock matrix with PP and PN matrix elements
      for(int m = 0; m < P; ++m)
	{
	  pfock[P * m + m] += Space.pShells.energy[m]; // Add diagonal elements to fock matrices
	  //pfock[P * m + m] += (2*Space.pShells.j[m] + 1) * Space.pShells.energy[m]; // Add diagonal elements to fock matrices
	  for(int l = 0; l < P; ++l)
	    {
	      term2 = 0;
	      if(Space.pShells.j[m] != Space.pShells.j[l] || Space.pShells.l[m] != Space.pShells.l[l]){ continue; }
	      //pfock[P * m + l] -= Bshift * pdensmat[P * m + l]; // Subtract level shift
	      for(int beta = 0; beta < Space.Pocc; ++beta) // Sum over occupied shells
		{
		  minj = abs(Space.pShells.j[m] - HF.p_j[beta]);
		  maxj = Space.pShells.j[m] + HF.p_j[beta];
		  for(int J = minj; J <= maxj; ++J)
		    {
		      for(int n = 0; n < P; ++n)
			{
			  if(Space.pShells.j[n] != HF.p_j[beta] || Space.pShells.l[n] != HF.p_l[beta]){ continue; }
			  for(int k = 0; k < P; ++k)
			    {
			      if(Space.pShells.j[k] != HF.p_j[beta] || Space.pShells.l[k] != HF.p_l[beta]){ continue; }
			      //term = pdensmat[P * k + n] * (2*J + 1) * ME.get_ppJME(m,n,l,k,J) / (2*Space.pShells.j[m] + 1);
			      term = HF.protons[beta][n] * HF.protons[beta][k] * ME.get_ppJME(m,n,l,k,J);
			      term *= (2*J + 1)/(2*Space.pShells.j[m] + 1);
			      pfock[P * m + l] += term;
			      term2 += term;
			      if(ind == 1 && m == 0 && l == 5)
				{
				  ++tempint;
				  //std::cout << Space.pShells.sp_ind[m]+1 <<" "<< Space.pShells.sp_ind[n]+1 <<" "<< Space.pShells.sp_ind[l]+1 <<" "<< Space.pShells.sp_ind[k]+1 <<"   "<< J <<"   "<< HF.protons[beta][n] <<" "<< HF.protons[beta][k] <<" "<< (2*J + 1)/(2*Space.pShells.j[m]+1) <<" "<< ME.get_ppJME(m,n,l,k,J) <<"   "<< term << endl;
				}
			    }
			}
		    }
		  if(Space.pShells.j[m] != HF.p_j[beta] || Space.pShells.l[m] != HF.p_l[beta]){ continue; }
		  pfock[P * m + l] -= HF.protons[beta][m] * HF.protons[beta][l] * Bshift;
		}
	      for(int beta = 0; beta < Space.Nocc; ++beta) // Sum over occupied shells
		{
		  minj = abs(Space.nShells.j[m] - HF.n_j[beta]);
		  maxj = Space.nShells.j[m] + HF.n_j[beta];
		  for(int J = minj; J <= maxj; ++J)
		    {
		      for(int n = 0; n < N; ++n)
			{
			  if(Space.nShells.j[n] != HF.n_j[beta] || Space.nShells.l[n] != HF.n_l[beta]){ continue; }
			  for(int k = 0; k < N; ++k)
			    {
			      if(Space.nShells.j[k] != HF.n_j[beta] || Space.nShells.l[k] != HF.n_l[beta]){ continue; }
			      //term = ndensmat[N * k + n] * (2*J + 1) * ME.get_pnJME(m,n,l,k,J) / (2*Space.pShells.j[m] + 1);
			      term = HF.neutrons[beta][n] * HF.neutrons[beta][k] * ME.get_pnJME(m,n,l,k,J);
			      term *= (2*J + 1)/(2*Space.pShells.j[m] + 1);
			      pfock[P * m + l] += term;
			      term2 += term;
			      if(ind == 1 && m == 0 && l == 5)
				{
				  ++tempint;
				  //std::cout << Space.pShells.sp_ind[m]+1 <<" "<< Space.nShells.sp_ind[n]+1 <<" "<< Space.pShells.sp_ind[l]+1 <<" "<< Space.nShells.sp_ind[k]+1 <<"   "<< J <<"   "<< HF.neutrons[beta][n] <<" "<< HF.neutrons[beta][k] <<" "<< (2*J + 1)/(2*Space.pShells.j[m]+1) <<" "<< ME.get_pnJME(m,n,l,k,J) <<"   "<< term << endl;
				}
			    }
			}
		    }
		}
	      //if(m == l){ std::cout << m << " " << l << " " << term2 << endl; }
	    }
	}
      
      /*for(int i = 0; i < P; ++i)
	{
	  for(int j = 0; j < P; ++j)
	    {
	      if(i == j){ std::cout << i+1 << " " << Space.pShells.energy[i] << " " << pfock[P * i + j] - Space.pShells.energy[i] << endl; }
	    }
	    }*/

      //Build N-fock matrix with NN and PN matrix elements
      for(int m = 0; m < N; ++m)
	{
	  nfock[N * m + m] += Space.nShells.energy[m]; // Add diagonal elements to fock matrices
	  //nfock[N * m + m] += (2*Space.nShells.j[m] + 1) * Space.nShells.energy[m]; // Add diagonal elements to fock matrices
	  for(int l = 0; l < N; ++l)
	    {
	      if(Space.nShells.j[m] != Space.nShells.j[l] || Space.nShells.l[m] != Space.nShells.l[l]){ continue; }
	      //nfock[N * m + l] -= Bshift * ndensmat[N * m + l];
	      for(int beta = 0; beta < Space.Nocc; ++beta) // Sum over occupied shells
		{
		  minj = abs(Space.nShells.j[m] - HF.n_j[beta]);
		  maxj = Space.nShells.j[m] + HF.n_j[beta];
		  for(int J = minj; J <= maxj; ++J)
		    {
		      for(int n = 0; n < N; ++n)
			{
			  if(Space.nShells.j[n] != HF.n_j[beta] || Space.nShells.l[n] != HF.n_l[beta]){ continue; }
			  for(int k = 0; k < N; ++k)
			    {
			      if(Space.nShells.j[k] != HF.n_j[beta] || Space.nShells.l[k] != HF.n_l[beta]){ continue; }
			      //term = ndensmat[N * k + n] * (2*J + 1) * ME.get_nnJME(m,n,l,k,J) / (2*Space.nShells.j[m] + 1);
			      term = HF.neutrons[beta][n] * HF.neutrons[beta][k] * ME.get_nnJME(m,n,l,k,J);
			      term *= (2*J + 1)/(2*Space.nShells.j[m] + 1);
			      nfock[N * m + l] += term;
			    }
			}
		    }
		  if(Space.nShells.j[m] != HF.n_j[beta] || Space.nShells.l[m] != HF.n_l[beta]){ continue; }
		  nfock[N * m + l] -= HF.neutrons[beta][m] * HF.neutrons[beta][l] * Bshift;
		}
	      for(int beta = 0; beta < Space.Pocc; ++beta) // Sum over occupied shells
		{
		  minj = abs(Space.nShells.j[m] - HF.p_j[beta]);
		  maxj = Space.nShells.j[m] + HF.p_j[beta];
		  for(int J = minj; J <= maxj; ++J)
		    {
		      for(int n = 0; n < P; ++n)
			{
			  if(Space.pShells.j[n] != HF.p_j[beta] || Space.pShells.l[n] != HF.p_l[beta]){ continue; }
			  for(int k = 0; k < P; ++k)
			    {
			      if(Space.pShells.j[k] != HF.p_j[beta] || Space.pShells.l[k] != HF.p_l[beta]){ continue; }
			      //term = pdensmat[P * k + n] * (2*J + 1) * ME.get_pnJME(n,m,k,l,J) / (2*Space.nShells.j[m] + 1);
			      term = HF.protons[beta][n] * HF.protons[beta][k] * ME.get_pnJME(n,m,k,l,J);
			      term *= (2*J + 1)/(2*Space.nShells.j[m] + 1);
			      term *= pow(-1.0, Space.pShells.j[n] + Space.nShells.j[m] + Space.pShells.j[k] + Space.nShells.j[l]);
			      nfock[N * m + l] += term;
			    }
			}
		    }
		}
	    }
	}      

      for(int i = 0; i < int(pfock.size()); ++i){ if(abs(pfock[i]) < 0.00000001){ pfock[i] = 0; }}
      for(int i = 0; i < int(nfock.size()); ++i){ if(abs(nfock[i]) < 0.00000001){ nfock[i] = 0; }}
      
      pw.assign(P,0);
      nw.assign(N,0);
      pwork.assign(plwork,0);
      nwork.assign(nlwork,0);

      std::cout << endl;
      for(int i = 0; i < P; ++i)
	{
	  for(int j = 0; j < P; ++j)
	    {
	      std::cout << pfock[P * i + j] << " ";
	    }
	  std::cout << endl;
	}
      std::cout << endl;

      if(P != 0){ dsyev_(&jobz, &uplo, &P, & *pfock.begin(), &plda, & *pw.begin(), & *pwork.begin(), &plwork, &pinfo); }
      if(N != 0){ dsyev_(&jobz, &uplo, &N, & *nfock.begin(), &nlda, & *nw.begin(), & *nwork.begin(), &nlwork, &ninfo); }
      
      for(int i = 0; i < int(pfock.size()); ++i){ if(abs(pfock[i]) < 0.00000001){ pfock[i] = 0; }}
      for(int i = 0; i < int(nfock.size()); ++i){ if(abs(nfock[i]) < 0.00000001){ nfock[i] = 0; }}

      /*for(int i = 0; i < P; ++i)
	{
	  for(int j = 0; j < P; ++j)
	    {
	      std::cout << pfock[P * i + j] << " ";
	    }
	  std::cout << endl;
	}
	std::cout << endl;*/

      //Add back Level-shift parameter
      for(int i = 0; i < Space.Pocc; ++i){ pw[i] += Bshift; }
      for(int i = 0; i < Space.Nocc; ++i){ nw[i] += Bshift; }

      //Get proton states, neutron states, and error
      /*for(int i = 0; i < P; ++i)
	{ 
	  double tempamp = 0;
	  int tempind = 0;
	  for(int j = 0; j < P; ++j){ if(abs(pfock[P * i + j]) > tempamp){ tempamp = abs(pfock[P * i + j]); tempind = j; }; }
	  for(int j = 0; j < P; ++j)
	    {
	      if(Space.pShells.j[j] != Space.pShells.j[tempind] || Space.pShells.l[j] != Space.pShells.l[tempind])
		{ HF.protons[i][j] = 0; }
	      else{ HF.protons[i][j] = pfock[P * i + j]; }
	    }
	  HF.p_j[i] = Space.pShells.j[tempind];
	  HF.p_l[i] = Space.pShells.l[tempind];
	  error += abs(HF.p_energies[i] - pw[i]); 
	  HF.p_energies[i] = pw[i];
	  //error += abs(HF.p_energies[i] - pw[i]/(2*HF.p_j[i] + 1)); 
	  //HF.p_energies[i] = pw[i]/(2*HF.p_j[i] + 1);
	}
      for(int i = 0; i < N; ++i)
	{ 
	  double tempamp = 0;
	  int tempind = 0;
	  for(int j = 0; j < N; ++j){ if(abs(nfock[N * i + j]) > tempamp){ tempamp = abs(nfock[N * i + j]); tempind = j; }; }
	  for(int j = 0; j < N; ++j)
	    {
	      if(Space.nShells.j[j] != Space.nShells.j[tempind] || Space.nShells.l[j] != Space.nShells.l[tempind])
		{ HF.neutrons[i][j] = 0; }
	      else{ HF.neutrons[i][j] = nfock[N * i + j]; }
	    }
	  HF.n_j[i] = Space.nShells.j[tempind];
	  HF.n_l[i] = Space.nShells.l[tempind];
	  error += abs(HF.n_energies[i] - nw[i]);
	  HF.n_energies[i] = nw[i];	  
	  //error += abs(HF.n_energies[i] - nw[i]/(2*HF.n_j[i] + 1)); 
	  //HF.n_energies[i] = nw[i]/(2*HF.n_j[i] + 1);
	}*/

      for(int i = 0; i < P; ++i)
	{ 
	  for(int j = 0; j < P; ++j)
	    { 
	      if(abs(pfock[P * i + j]) > 0)
		{
		  //pw[i] /= (2.0 * Space.pShells.j[j] + 1);
		  HF.p_j[i] = Space.pShells.j[j];
		  HF.p_l[i] = Space.pShells.l[j];
		  break;
		}
	    }
	  for(int j = 0; j < P; ++j){ HF.protons[i][j] = pfock[P * i + j]; }
	}
      for(int i = 0; i < N; ++i)
	{ 
	  for(int j = 0; j < N; ++j)
	    { 
	      if(abs(nfock[N * i + j]) > 0)
		{
		  //nw[i] /= (2.0 * Space.nShells.j[j] + 1);
		  HF.n_j[i] = Space.nShells.j[j];
		  HF.n_l[i] = Space.nShells.l[j];
		  break;
		}
	    }
	  for(int j = 0; j < N; ++j){ HF.neutrons[i][j] = nfock[N * i + j]; }
	}

      int ind2;
      double tempen2;
      // Order states by energy
      for(int i = 0; i < P - 1; ++i)
	{
	  ind2 = i;
	  tempen2 = pw[i];
	  for(int j = i + 1; j < P; ++j)
	    {
	      if(pw[j] < tempen2){ tempen2 = pw[j]; ind2 = j; }
	    }
	  std::swap(pw[i], pw[ind2]);
	  std::swap(HF.protons[i], HF.protons[ind2]);
	  std::swap(HF.p_j[i], HF.p_j[ind2]);
	  std::swap(HF.p_l[i], HF.p_l[ind2]);
	}
      
      for(int i = 0; i < N - 1; ++i)
	{
	  ind2 = i;
	  tempen2 = nw[i];
	  for(int j = i + 1; j < N; ++j)
	    {
	      if(nw[j] < tempen2){ tempen2 = nw[j]; ind2 = j; }
	    }
	  std::swap(nw[i], nw[ind2]);
	  std::swap(HF.neutrons[i], HF.neutrons[ind2]);
	  std::swap(HF.n_j[i], HF.n_j[ind2]);
	  std::swap(HF.n_l[i], HF.n_l[ind2]);
	}
      
      std::cout << endl;
      for(int i = 0; i < P; ++i){ std::cout << pw[i] << " "; }
      std::cout << endl;

      for(int i = 0; i < P; ++i)
	{
	  error += abs(HF.p_energies[i] - pw[i]); 
	  HF.p_energies[i] = pw[i];
	}
      for(int i = 0; i < N; ++i)
	{
	  error += abs(HF.n_energies[i] - nw[i]); 
	  HF.n_energies[i] = nw[i];
	}
      error /= P + N;
      
      GramSchmidt(HF.protons);
      GramSchmidt(HF.neutrons);

      Separate_Particles_Holes(HF, Space.Pocc, Space.Nocc, Space);
	
      std::cout << "error = " << error << endl << endl;
    }

  vector<int> p_n(HF.protons.size());
  vector<int> n_n(HF.neutrons.size());
  
  for(int i = 0; i < int(HF.protons.size()); ++i)
    {
      int tempn = -1;
      int templ = HF.p_l[i];
      double tempj = HF.p_j[i];
      for(int j = 0; j <= i; ++j)
	{
	  if(HF.p_l[j] == templ && HF.p_j[j] == tempj){ ++tempn; }
	}
      p_n[i] = tempn;
    }
  for(int i = 0; i < int(HF.neutrons.size()); ++i)
    {
      int tempn = -1;
      int templ = HF.n_l[i];
      double tempj = HF.n_j[i];
      for(int j = 0; j <= i; ++j)
	{
	  if(HF.n_l[j] == templ && HF.n_j[j] == tempj){ ++tempn; }
	}
      n_n[i] = tempn;
    }

  ofstream HFlevelfile;
  string filename = PATH + Parameters.LevelScheme + "_HF.sp";
  HFlevelfile.open(filename.c_str());
  HFlevelfile << "Mass number A of chosen nucleus (important for CoM corrections): \t" << Space.A << "\n";
  HFlevelfile << "Oscillator energy: \t" << Space.HOEnergy << "\n";
  HFlevelfile << "Total number of single-particle orbits: \t" << Space.shelltot << "\n";
  HFlevelfile << "Legend:   \tn \tl \t2j \ttz \t2n+l \tHO-energy \tevalence \tparticle/hole \tinside/outside \n";
  for(int i = 0; i < int(HF.protons.size()); ++i)
    {
      HFlevelfile << "Number:   " << i+1 << "\t" << p_n[i] << "\t" << HF.p_l[i] << "\t" << int(2*HF.p_j[i]) << "\t";
      HFlevelfile << "-1" << "\t" << 2*p_n[i]+HF.p_l[i] << "\t" << setprecision(8) << HF.p_energies[i] << "\t" << "0.000000" << "\t";
      if(i < Space.Pocc){ HFlevelfile << "hole    " << "\t" << "inside" << "\n"; }
      else{ HFlevelfile << "particle" << "\t" << "inside" << "\n"; }
    }
  for(int i = 0; i < int(HF.neutrons.size()); ++i)
    {
      HFlevelfile << "Number:   " << i+int(HF.protons.size())+1 << "\t" << n_n[i] << "\t" << HF.n_l[i] << "\t" << int(2*HF.n_j[i]) << "\t";
      HFlevelfile << "1" << "\t" << 2*n_n[i]+HF.n_l[i] << "\t" << setprecision(8) << HF.n_energies[i] << "\t" << "0.000000" << "\t";
      if(i < Space.Nocc){ HFlevelfile << "hole    " << "\t" << "inside" << "\n"; }
      else{ HFlevelfile << "particle" << "\t" << "inside" << "\n"; }
    }

  HFlevelfile.close();

  return HF;
  
}



J_Matrix_Elements Convert_To_HF_Matrix_Elements(const string &MatrixElements, const Single_Particle_States &States, const Model_Space &Space, const J_Matrix_Elements &ME)
{
  std::cout << "Converting Matrix Elements from J-Scheme to HF J-Scheme" << endl;
  std::cout << "----------------------------------------------------" << endl;

  J_Matrix_Elements HF_ME(Space.pshelltot, Space.nshelltot, Space.pShells.j, Space.nShells.j); // HF M-Scheme matrix elements
  int plength, pmatlength, nlength, nmatlength, pnmatlength; // max length of M-Scheme indicies, length of J_ME
  int num1, num2;
  double tempel;
  vector<double> PP_M1, PP_M2, NN_M1, NN_M2, PN_M1, PN_M2; // Matrices of coefficients
  vector<double> PP_V, NN_V, PN_V;
  vector<double> PP_C, NN_C, PN_C;
  char transa, transb;
  double alpha1, beta1;
  int ind;
  ofstream jschemefile; // file to print M-Scheme matrix elements

  plength = Space.pshelltot;
  nlength = Space.nshelltot;

  // Fill OBME with HF single-particle energies
  for(int i = 0; i < plength; ++i){ HF_ME.set_pJME(i, States.p_energies[i]); }
  for(int i = 0; i < nlength; ++i){ HF_ME.set_nJME(i, States.n_energies[i]); }

  pmatlength = pow(plength, 2.0);
  PP_M1.assign(pow(pmatlength, 2.0), 0);
  PP_M2.assign(pow(pmatlength, 2.0), 0);
  for(int p = 0; p < plength; ++p)
    {
      for(int q = 0; q < plength; ++q)
	{
	  for(int alpha = 0; alpha < plength; ++alpha)
	    {
	      for(int gamma = 0; gamma < plength; ++gamma)
		{
		  num1 = number_diff(p, q, plength);
		  num2 = number_diff(alpha, gamma, plength);
		  tempel = States.protons[p][alpha] * States.protons[q][gamma];
		  PP_M1[pmatlength * num1 + num2] = tempel;
		  PP_M2[pmatlength * num2 + num1] = tempel;
		}
	    }
	}
    }

  nmatlength = pow(nlength, 2.0);
  NN_M1.assign(pow(nmatlength, 2.0), 0);
  NN_M2.assign(pow(nmatlength, 2.0), 0);
  for(int p = 0; p < nlength; ++p)
    {
      for(int q = 0; q < nlength; ++q)
	{
	  for(int alpha = 0; alpha < nlength; ++alpha)
	    {
	      for(int gamma = 0; gamma < nlength; ++gamma)
		{
		  num1 = number_diff(p, q, nlength);
		  num2 = number_diff(alpha, gamma, nlength);
		  tempel = States.neutrons[p][alpha] * States.neutrons[q][gamma];
		  NN_M1[nmatlength * num1 + num2] = tempel;
		  NN_M2[nmatlength * num2 + num1] = tempel;
		}
	    }
	}
    }

  pnmatlength = plength * nlength;
  PN_M1.assign(pow(pnmatlength, 2.0), 0);
  PN_M2.assign(pow(pnmatlength, 2.0), 0);
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
		  PN_M1[pnmatlength * num1 + num2] = tempel;
		  PN_M2[pnmatlength * num2 + num1] = tempel;
		}
	    }
	}
    }

  ind = 0;
  for(int J = 0; J <= Space.max2J; ++J)
    {
      PP_V.assign(pow(pmatlength, 2.0), 0);
      for(int alpha = 0; alpha < plength; ++alpha)
	{
	  for(int gamma = 0; gamma < plength; ++gamma)
	    {
	      for(int beta = 0; beta < plength; ++beta)
		{
		  for(int delta = 0; delta < plength; ++delta)
		    {
		      num1 = number_diff(alpha, gamma, plength);
		      num2 = number_diff(beta, delta, plength);
		      tempel = ME.get_ppJME(alpha, gamma, beta, delta, J);
		      PP_V[pmatlength * num1 + num2] = tempel;
		    }
		}
	    }
	}

      NN_V.assign(pow(nmatlength, 2.0), 0);
      for(int alpha = 0; alpha < nlength; ++alpha)
	{
	  for(int gamma = 0; gamma < nlength; ++gamma)
	    {
	      for(int beta = 0; beta < nlength; ++beta)
		{
		  for(int delta = 0; delta < nlength; ++delta)
		    {
		      num1 = number_diff(alpha, gamma, nlength);
		      num2 = number_diff(beta, delta, nlength);
		      tempel = ME.get_nnJME(alpha, gamma, beta, delta, J);
		      if(alpha == gamma){ tempel /= 2.0 / sqrt(1.0 + pow(-1.0, J)); }
		      if(beta == delta){ tempel /= 2.0 / sqrt(1.0 + pow(-1.0, J)); }
		      NN_V[pmatlength * num1 + num2] = tempel;
		    }
		}
	    }
	}
      
      PN_V.assign(pow(pnmatlength, 2.0), 0);
      for(int alpha = 0; alpha < plength; ++alpha)
	{
	  for(int gamma = 0; gamma < nlength; ++gamma)
	    {
	      for(int beta = 0; beta < plength; ++beta)
		{
		  for(int delta = 0; delta < nlength; ++delta)
		    {
		      num1 = number_diff(alpha, gamma, nlength);
		      num2 = number_diff(beta, delta, nlength);
		      tempel = ME.get_pnJME(alpha, gamma, beta, delta, J);
		      PN_V[pnmatlength * num1 + num2] = tempel;
		    }
		}
	    }
	}

      transa = 'N';
      transb = 'N';
      alpha1 = 1.0;
      beta1 = 0.0;
      
      PP_C.assign(pow(pmatlength, 2.0), 0);
      if(pmatlength != 0)
	{
	  RM_dgemm(&transa, &transb, &pmatlength, &pmatlength, &pmatlength, &alpha1, & *PP_M1.begin(), &pmatlength, & *PP_V.begin(), &pmatlength, &beta1, & *PP_C.begin(), &pmatlength );
	  RM_dgemm(&transa, &transb, &pmatlength, &pmatlength, &pmatlength, &alpha1, & *PP_C.begin(), &pmatlength, & *PP_M2.begin(), &pmatlength, &beta1, & *PP_V.begin(), &pmatlength );
	}
        
      NN_C.assign(pow(nmatlength, 2.0), 0);
      if(nmatlength != 0)
	{
	  RM_dgemm(&transa, &transb, &nmatlength, &nmatlength, &nmatlength, &alpha1, & *NN_M1.begin(), &nmatlength, & *NN_V.begin(), &nmatlength, &beta1, & *NN_C.begin(), &nmatlength );
	  RM_dgemm(&transa, &transb, &nmatlength, &nmatlength, &nmatlength, &alpha1, & *NN_C.begin(), &nmatlength, & *NN_M2.begin(), &nmatlength, &beta1, & *NN_V.begin(), &nmatlength );
	}
      
      PN_C.assign(pow(pnmatlength, 2.0), 0);
      if(pnmatlength != 0)
	{
	  RM_dgemm(&transa, &transb, &pnmatlength, &pnmatlength, &pnmatlength, &alpha1, & *PN_M1.begin(), &pnmatlength, & *PN_V.begin(), &pnmatlength, &beta1, & *PN_C.begin(), &pnmatlength );
	  RM_dgemm(&transa, &transb, &pnmatlength, &pnmatlength, &pnmatlength, &alpha1, & *PN_C.begin(), &pnmatlength, & *PN_M2.begin(), &pnmatlength, &beta1, & *PN_V.begin(), &pnmatlength );
	}
       
      for(int p = 0; p < plength; ++p)
	{
	  for(int q = p; q < plength; ++q)
	    {
	      for(int r = p; r < plength; ++r)
		{
		  for(int s = r; s < plength; ++s)
		    {
		      num1 = number_diff(p, q, plength);
		      num2 = number_diff(r, s, plength);
		      if(num1 > num2){ swap(num1, num2); }
		      if(abs(PP_V[pmatlength * num1 + num2]) > 1.0e-8)
			{ 
			  HF_ME.set_ppJME(p, q, r, s, J, PP_V[pmatlength * num1 + num2]);
			  ++ind;
			}
		    }
		}
	    }
	}
          
      for(int p = 0; p < nlength; ++p)
	{
	  for(int q = p; q < nlength; ++q)
	    {
	      for(int r = p; r < nlength; ++r)
		{
		  for(int s = r; s < nlength; ++s)
		    {
		      num1 = number_diff(p, q, nlength);
		      num2 = number_diff(r, s, nlength);
		      if(num1 > num2){ swap(num1, num2); }
		      if(abs(NN_V[nmatlength * num1 + num2]) > 1.0e-8)
			{ 
			  HF_ME.set_nnJME(p, q, r, s, J, NN_V[nmatlength * num1 + num2]);
			  ++ind;
			}
		    }
		}
	    }
	}
           
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
		      if(abs(PN_V[pnmatlength * num1 + num2]) > 1.0e-8)
			{ 
			  HF_ME.set_pnJME(p, q, r, s, J, PN_V[pnmatlength * num1 + num2]);
			  ++ind;
			}
		    }
		}
	    }
	}
    }

  HF_ME.cut_ppJME();
  HF_ME.cut_nnJME();
  HF_ME.cut_pnJME();

  // print J_ME to file
  jschemefile.open((PATH + MatrixElements + "_HF.int").c_str());
  jschemefile << "Total number of twobody matx elements:" << "\t" << ind << "\n";
  jschemefile << "----> Interaction part\n";   
  jschemefile << "Nucleon-Nucleon interaction model:n3lo\n";            
  jschemefile << "Type of calculation:vlowk\n";              
  jschemefile << "Number and value of starting energies:   1	0.000000E+00\n";
  jschemefile << "Total number of twobody matx elements:\t" << ind << "\n";
  jschemefile << "Matrix elements with the following legend:  Tz,  Par,  2J,  a,  b,  c,  d,  <ab|V|cd>\n";
  for(int J = 0; J <= Space.max2J; ++J)
    {
      for(int m = 0; m < plength; ++m)
	{
	  for(int n = m; n < plength; ++n)
	    {
	      for(int l = m; l < plength; ++l)
		{
		  for(int k = l; k < plength; ++k)
		    {
		      if(HF_ME.get_ppJME(m, n, l, k, J) == 0){ continue; }
		      if(m == n){ tempel /= 2.0 / sqrt(1.0 + pow(-1.0, J)); }
		      if(l == k){ tempel /= 2.0 / sqrt(1.0 + pow(-1.0, J)); }
		      jschemefile << "\t -1\t" << pow(-1.0,States.p_l[m]+States.p_l[n]) << "\t" << 2*J << "\t"
				  << m + 1 << "\t" << n + 1 << "\t" << l + 1 << "\t" << k + 1
				  << "\t" << setprecision(8) << HF_ME.get_ppJME(m, n, l, k, J) << "\n";
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
		      if(HF_ME.get_pnJME(m, n, l, k, J) == 0){ continue; }
		      jschemefile << "\t  0\t" << pow(-1.0,States.p_l[m]+States.n_l[n]) << "\t" << 2*J << "\t"
				  << m + 1 << "\t" << n + plength + 1 << "\t" << l + 1 << "\t" << k + plength + 1 
				  << "\t" << setprecision(8) << HF_ME.get_pnJME(m, n, l, k, J) << "\n";
		    }
		}
	    }
	}
      for(int m = 0; m < nlength; ++m)
	{
	  for(int n = m; n < nlength; ++n)
	    {
	      for(int l = m; l < nlength; ++l)
		{
		  for(int k = l; k < nlength; ++k)
		    {
		      if(HF_ME.get_nnJME(m, n, l, k, J) == 0){ continue; }
		      if(m == n){ tempel /= 2.0 / sqrt(1.0 + pow(-1.0, J)); }
		      if(l == k){ tempel /= 2.0 / sqrt(1.0 + pow(-1.0, J)); }
		      jschemefile << "\t  1\t" << pow(-1.0,States.n_l[m]+States.n_l[n]) << "\t" << 2*J << "\t"
				  << m + plength + 1 << "\t" << n + plength + 1 << "\t" << l + plength + 1 << "\t"
				  << k + plength + 1 << "\t" << setprecision(8) << HF_ME.get_nnJME(m, n, l, k, J) << "\n";
		    }
		}
	    }
	}
    }

  jschemefile.close();
  
  return HF_ME;

  }



int main(int argc, char* argv[])
{ 
  //clock_t t1, t2;    	//initialize program clock
  //t1 = clock();

  string inputfile = "input.dat";
  Input_Parameters Parameters = Get_Input_Parameters(inputfile);
  Model_Space Space = Build_Model_Space(Parameters);
  J_Matrix_Elements JME = Get_J_Matrix_Elements(Parameters, Space);
  Single_Particle_States States = Build_Single_Particle_States(Parameters, Space);
  Single_Particle_States HFStates = Hartree_Fock_States2(Parameters, States, Space, JME);
  J_Matrix_Elements HF_ME = Convert_To_HF_Matrix_Elements(Parameters.MatrixElements, HFStates, Space, JME);

  //double hf_energy = 0;
  //double factor;
  /*for(int i = 0; i < int(HFStates.holes.size()); ++i)
    {
      hf_energy += (2 * HFStates.h_j[i] + 1) * HFStates.h_energies[i];
    }
  std::cout << hf_energy << endl;
  for(int J = 0; J <= Space.max2J; ++J)
    {
      for(int i = 0; i < Space.Pocc; ++i)
	{
	  for(int j = 0; j < Space.Pocc; ++j)
	    {
	      hf_energy -= 0.5 * (2 * J + 1) * HF_ME.get_ppJME(i, j, i, j, J) / (2*Space.pShells.j[i] + 1);
	    }
	}
      std::cout << hf_energy << endl;
      for(int i = 0; i < Space.Nocc; ++i)
	{
	  for(int j = 0; j < Space.Nocc; ++j)
	    {
	      hf_energy -= 0.5 * (2 * J + 1) * HF_ME.get_nnJME(i, j, i, j, J) / (2*Space.nShells.j[i] + 1);
	    }
	}
      std::cout << hf_energy << endl;
      for(int i = 0; i < Space.Pocc; ++i)
	{
	  for(int j = 0; j < Space.Nocc; ++j)
	    {
	      hf_energy -= 0.5 * (2 * J + 1) * HF_ME.get_pnJME(i, j, i, j, J) / (2*Space.pShells.j[i] + 1);
	    }
	}
      std::cout << hf_energy << endl;
      for(int i = 0; i < Space.Nocc; ++i)
	{
	  for(int j = 0; j < Space.Pocc; ++j)
	    {
	      hf_energy -= 0.5 * (2 * J + 1) * HF_ME.get_pnJME(j, i, j, i, J) / (2*Space.pShells.j[i] + 1);
	    }
	}
    }

    std::cout << "HF Energy = " << hf_energy << endl;*/
		  	  
  //t2 = clock();
  //float diff((float)t2 - (float)t1);
  //std::cout << diff / CLOCKS_PER_SEC << "sec" << endl;

  std::cout << "END!!!" << endl;

  int a;
  std::cin >> a;

  return 0;

}
