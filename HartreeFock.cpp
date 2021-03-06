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

const string PATH = "/home/sam/Documents/HartreeFock/files/";

struct Input_Parameters{
  int COM; //flag for center-of-mass matrix elements
  string LevelScheme; //level scheme path
  string MatrixElements; //matrix elements path
  string COMMatrixElements; //com matrix elements path
};

struct Model_Space{
  string type; //pn or t
  int indp; //number of proton major orbits
  int indn; //number of neutron major orbits
  int indtot; //number of total orbits
  vector<int> levelsind; //list of single particle state indicies (1,2...)
  vector<int> levelsn; //list of single particle state principal quantum numbers
  vector<int> levelsl; //list of single particle state orbital angular momentum
  vector<double> levelsj; //list of single particle state total angular momentum
  vector<double> levelsm; //list of single particle state total angular momentum projection
  vector<double> levelst; //list of single particle state isospins
  vector<int> levelschemeind; //list of major shell indicies (1,2...)
  vector<double> shellsj; //list of major shell angular momentums
  vector<double> shellst; //list of major shell isospin projection
  vector<vector<int> > shellsm; //list of single particle state indicies for each major shell
  vector<string> shellsname;
};

struct Single_Particle_States{
  vector<vector<double> > configs; //list of sp-states given as vectors of coefficients
  vector<double> energies; //list of sp-energies
};

struct Matrix_Elements{
  char type; //j or m scheme
  vector<vector<int> > Braket; //list of 4 J-scheme states that make up TBME
  vector<int> J; //list of coupled angular momentum (2x)
  vector<int> T; //list of coupled isospin (2x)
  vector<double> OBME; //list of one-body matrix element values
  vector<double> TBME; //list of two-body matrix element values
};




//Initialize program from input file
Input_Parameters Get_Input_Parameters(string infile)
{ 
  Input_Parameters Input;
  
  std::cout << "Reading Input File" << endl;
  std::cout << "------------------" << endl;

  /***Read Input File***/
  string line;    //initialize placeholder string
  ifstream filestream;
  string path = PATH + infile;
  filestream.open(path.c_str());
  if (!filestream.is_open())
    {
      cerr << "Input file, " << path << ", does not exist" << endl; exit(1);
    }

  //find lines that start with '\*'
  int index = 0; //keep track of input line
  while (getline(filestream, line))
    { if(line[0] == '\\' && line[1] == '*')
	{  
	  ++index;
	  size_t colon = line.find(':');
	  if( colon == line.size() - 1 ){ continue; };
	  string substr = line.substr(colon + 2, line.size());
	  switch(index)
	    {
	    case 1:
	      Input.LevelScheme = substr;
	      break;
	    case 2:
	      Input.MatrixElements = substr;
	      if (Input.MatrixElements[Input.MatrixElements.size() - 2] == '_' && Input.MatrixElements[Input.MatrixElements.size() - 1] == 'M')
		{
		  Input.MatrixElements.erase(Input.MatrixElements.end() - 2, Input.MatrixElements.end());
		}
	      break;
	    case 3:
	      Input.COM = atoi(substr.c_str());
	      break;
	    case 4:
	      if(Input.COM == 1)
		{ 
		  Input.COMMatrixElements = substr;
		  if (Input.COMMatrixElements[Input.COMMatrixElements.size() - 2] == '_' && Input.COMMatrixElements[Input.COMMatrixElements.size() - 1] == 'M')
		    {
		      Input.COMMatrixElements.erase(Input.COMMatrixElements.end() - 2, Input.COMMatrixElements.end());
		    }
		  break;
		}
	      else{ break; }
	    } 
	}
      else{ continue; };
    }

  std::cout << "Level Scheme = " << Input.LevelScheme << endl;
  std::cout << "Matrix Elements = " << Input.MatrixElements << " " << Input.COMMatrixElements << endl << endl;

  return Input;
}





Model_Space Build_Model_Space(Input_Parameters Parameters)
{
  Model_Space Space;

  std::cout << "Reading Model Space File" << endl;
  std::cout << "------------------------" << endl;

  //Unpacking the Level Scheme
  string fileformat;    //initialize placeholder string and level format
  int coreA, coreZ, TotOrbs, Shells, POrbs, NOrbs;    //initialize core A, core Z, # total orbits, # major shells, # p orbits, # n orbits
  int ind, n, l, j2;    //initialize level index, n, l, and 2j from file. m is added later.
                        //also initialize corresponding vectors. j from file is 2j
  Space.indtot = 0; //keep running counts of single-particle states
  Space.indp = 0; //number of proton single particle states
  Space.indn = 0; //number of neutron single particle states
  
  //open level scheme file named splevels

  ifstream splevels;
  string fullpath = PATH + Parameters.LevelScheme + ".sp";
  splevels.open(fullpath.c_str());
  if (!splevels.is_open())
    {
      cerr << "Level Scheme file does not exist" << endl; exit(1);
    };

  //skip lines that start with '!'
  string phline;
  getline(splevels, phline);
  while (phline[0] == '!'){ getline(splevels, phline); }

  //read rest of level scheme parameters
  Space.type = phline;
  for(int i = 0; i < int(Space.type.size()); ++i){ std::cout << Space.type[i]  << " "; }; std::cout << endl;
  getline(splevels, phline);
  istringstream(phline) >> coreA >> coreZ;
  getline(splevels, phline);
  istringstream(phline) >> TotOrbs;
  getline(splevels, phline);
  if(Space.type == "pn"){ istringstream(phline) >> Shells >> POrbs >> NOrbs; }
  else if(Space.type == "t"){ istringstream(phline) >> Shells >> POrbs; NOrbs = 0; }
  else{ cerr << "Level Scheme is not formatted as 'pn' or 't'" << endl; splevels.close(); exit(1); };

  vector<int> indvec(TotOrbs);
  vector<int> nvec(TotOrbs);
  vector<int> lvec(TotOrbs);
  vector<int> j2vec(TotOrbs);

  Space.shellsj.resize(TotOrbs);
  Space.shellst.resize(TotOrbs);
  Space.shellsm.resize(TotOrbs);
  Space.shellsname.resize(TotOrbs);

  std::cout << "Core: Z = " << coreZ << ", A = " << coreA << endl;
  if(Space.type == "pn"){ std::cout << "Proton Valence Shells:" << endl; };
  if(Space.type == "t"){ std::cout << "Valence Shells:" << endl; };
  int sp_count = 0; //count single-particle states
  char shell;
  for(int i = 0; i < TotOrbs; ++i)
    {
      if(Space.type == "pn" && i == POrbs){ std::cout << "Neutron Valence Shells:" << endl; };
      getline(splevels, phline);
      istringstream(phline) >> ind >> n >> l >> j2;
      //std::cout << ind << " " << n << " " << l << " " << j2 << endl;
      indvec[i] = ind;
      nvec[i] = n;
      lvec[i] = l;
      j2vec[i] = j2;
      Space.shellsj[i] = 0.5*j2;
      if(i < POrbs){ Space.shellst[i] = 0.5; }
      else{ Space.shellst[i] = -0.5; };

      switch(l)
	{
	case 0:
	  shell = 's';
	  break;
	case 1:
	  shell = 'p';
	  break;
	case 2:
	  shell = 'd';
	  break;
	case 3:
	  shell = 'f';
	  break;
	case 4:
	  shell = 'g';
	  break;
	case 5:
	  shell = 'h';
	  break;
	default:
	  shell = 'M';
	  break;
	};

      stringstream a, b, c;
      a << n;
      b << shell;
      c << j2;
      string a1 = a.str(), b1 = b.str(), c1 = c.str();
      Space.shellsname[i] = a1 + b1 + c1 + "/2";
      std::cout << Space.shellsname[i] << endl;

      sp_count += j2 + 1;
      Space.shellsm[i].resize(j2 + 1);
    };
  std::cout << endl;

  if(Space.type == "t"){ sp_count = 2 * sp_count; };

  Space.levelst.resize(sp_count);
  Space.levelsind.resize(sp_count);
  Space.levelsn.resize(sp_count);
  Space.levelsl.resize(sp_count);
  Space.levelsj.resize(sp_count);
  Space.levelsm.resize(sp_count);
  Space.levelschemeind.resize(sp_count);

  if(Space.type == "pn")
    {
      for(int i = 0; i < TotOrbs; ++i)
	{
	  int shelllength = j2vec[i] + 1;
	  for(int j = 0; j < shelllength; ++j)
	    {
	      Space.indtot++;
	      if(indvec[i] <= POrbs){ Space.indp++; Space.levelst[Space.indtot - 1] = 0.5; }
	      else{ Space.indn++; Space.levelst[Space.indtot - 1] = -0.5; };	//protons T_z = 1/2, neutrons T_z = -1/2
	      Space.levelsind[Space.indtot - 1] = Space.indtot;
	      Space.levelsn[Space.indtot - 1] = nvec[i];
	      Space.levelsl[Space.indtot - 1] = lvec[i];
	      Space.levelsj[Space.indtot - 1] = 0.5*j2vec[i];
	      Space.levelsm[Space.indtot - 1] = -0.5*j2vec[i] + j;
	      Space.levelschemeind[Space.indtot - 1] = indvec[i];
	      Space.shellsm[i][j] = Space.indtot;
	    };
	};
    }
  else if(Space.type == "t")
    {
      for(int n = 0; n < 2; ++n)
	{
	  for(int i = 0; i < TotOrbs; ++i)
	    {
	      int shelllength = j2vec[i] + 1;
	      for(int j = 0; j < shelllength; ++j)
		{
		  Space.indtot++;
		  if(n*TotOrbs + indvec[i] <= TotOrbs){ Space.indp++; Space.levelst[Space.indtot - 1] = 0.5; }
		  else{ Space.indn++; Space.levelst[Space.indtot - 1] = -0.5; };	//protons T_z = 1/2, neutrons T_z = -1/2
		  Space.levelsind[Space.indtot - 1] = Space.indtot;
		  Space.levelsn[Space.indtot - 1] = nvec[i];
		  Space.levelsl[Space.indtot - 1] = lvec[i];
		  Space.levelsj[Space.indtot - 1] = 0.5*j2vec[i];
		  Space.levelsm[Space.indtot - 1] = -0.5*j2vec[i] + j;
		  Space.levelschemeind[Space.indtot - 1] = indvec[i];
		  Space.shellsm[i][j] = Space.indtot;
		};
	    };
	};
    }

  //Print Single Particle State Details
  /*for (int i = 0; i < Space.indtot; ++i)
    {
      std::cout << Space.levelst[i] << " " << Space.levelsind[i] << " " << Space.levelsn[i] << " " << Space.levelsl[i] << " " << Space.levelsj[i] << " " << Space.levelsm[i] << " " << Space.levelschemeind[i] << endl;
      }*/

  splevels.close();

  return Space;
}





Single_Particle_States Build_Single_Particle_States(Input_Parameters Parameters, Model_Space Space, Matrix_Elements ME)
{
  int N = Space.indtot;
  Single_Particle_States States;
  States.configs.resize(N);
  States.energies.resize(N);
  for(int i = 0; i < N; ++i){ States.configs[i].resize(N/2); }

  std::cout << "Building Single-Particle States" << endl;
  std::cout << "-------------------------------" << endl;

  /*std::srand(std::time(0));
  for (int i = 0; i < N; ++i)
    {
      double norm = 0.0;
      for (int j = 0; j < N/2; ++j)
	{
	  double entry = std::rand();
	  std::cout << entry << endl;
	  States.configs[i][j] = entry;
	  norm += entry*entry;
	};
      for (int j = 0; j < N/2; ++j)
	{
	  States.configs[i][j] = States.configs[i][j]/sqrt(norm);
	};
	};*/

  for(int i = 0; i < N; ++i)
    {
      for(int j = 0; j < N/2; ++j)
	{
	  if(i == j || (i - N/2) == j){ States.configs[i][j] = 1.0; }
	  else{ States.configs[i][j] = 0.0; };
	  //std::cout << States.configs[i][j] << " ";
	}
      //std::cout << endl;
    }
  
  for(int i = 0; i < N; ++i)
    {
      States.energies[i] = ME.OBME[Space.levelschemeind[i] - 1];
    };
  
  //Print State Details
  /*std::cout << endl;
  for(int i = 0; i < N; ++i)
    {
      double norm2 = 0.0;
      for(int j = 0; j < N/2; ++j)
	{
	  norm2 += States.configs[i][j]*States.configs[i][j];
	  std::cout << States.configs[i][j] << " ";
	};
      std::cout << "vector " << i << ": " << norm2 << endl;
    };
    std::cout << endl;*/

  return States;

}
  


vector<double> Possible_Couplings(double j1, double j2, double m1)
{
  double tempmom = abs(j1 + j2);
  int ind = 0;
  vector<double> angmom(int(tempmom - abs(j1 - j2) + 1.1)); 
  while (tempmom >= abs(j1 - j2) && tempmom >= abs(m1))
    { 
      angmom[ind] = tempmom; ++ind;
      tempmom = tempmom - 1.0;
    };
  angmom.resize(ind);
  
  return angmom;
}



Matrix_Elements Convert_To_M_Matrix_Elements(string MatrixElements, Model_Space Space, Matrix_Elements J_ME)
{
  Matrix_Elements M_ME;

  std::cout << "Converting Matrix Elements from J-Scheme to M-Scheme" << endl;
  std::cout << "----------------------------------------------------" << endl;

  M_ME.OBME = J_ME.OBME;

  //Change J-Scheme matrix elements to M-Scheme
  int length1 = int(Space.levelsind.size());
  int length2 = int(J_ME.TBME.size());
  int l2, k2, m2, n2;

  int size = int(length1*length1*(length1 - 1)*(length1 - 1)/8);
  int ind = 0;
  vector<int> braket(4);
  M_ME.Braket.resize(size);
  for(int i = 0; i < size; ++i){ M_ME.Braket[i].resize(4); };
  M_ME.TBME.resize(size);

  double tbodyphase;

  int testint1 = 0;
  
  for (int m1 = 0; m1 < length1 - 1; ++m1)
    {
      for (int n1 = m1 + 1; n1 < length1; ++n1)
	{
	  for (int l1 = m1; l1 < length1 - 1; ++l1)
	    {
	      for (int k1 = l1 + 1; k1 < length1; ++k1)
		{
		  if(m1 == l1 && n1 > k1){ continue; };
		  double angproj1 = Space.levelsm[m1] + Space.levelsm[n1];
		  double angproj2 = Space.levelsm[l1] + Space.levelsm[k1];
		  double isoproj1 = Space.levelst[m1] + Space.levelst[n1];
		  double isoproj2 = Space.levelst[l1] + Space.levelst[k1];
		  if (angproj1 == angproj2 && isoproj1 == isoproj2)
		    {
		      ++testint1;
		      
		      vector<double> angmom1 = Possible_Couplings(Space.levelsj[m1], Space.levelsj[n1], angproj1);
		      vector<double> angmom2 = Possible_Couplings(Space.levelsj[l1], Space.levelsj[k1], angproj2); 

		      int size3 = (angmom1.size() >= angmom2.size() ? angmom1.size() : angmom2.size());
		      int ind3 = 0;
		      vector<double> angmom3(size3);
		      for (int i = 0; i < int(angmom1.size()); ++i)
			{
			  for (int j = 0; j < int(angmom2.size()); ++j)
			    {
			      if (angmom1[i] == angmom2[j]){ angmom3[ind3] = angmom1[i]; ++ind3; break; };
			    };
			};
		      angmom3.resize(ind3);

		      vector<double> iso3 = Possible_Couplings(0.5, 0.5, isoproj1);

		      // get interaction file indices that correspond to the level scheme indices
		      m2 = Space.levelschemeind[m1];
		      n2 = Space.levelschemeind[n1];
		      l2 = Space.levelschemeind[l1];
		      k2 = Space.levelschemeind[k1];
		      double tempmatel = 0.0;

		      for (int b = 0; b < int(angmom3.size()); ++b)	//	loop over relevant angular momentum
			{
			  for (int c = 0; c < int(iso3.size()); ++c)	//	loop over relevant isospin
			    {
			      for (int a = 0; a < length2; ++a)	//	loop over interaction file lines
				{
				  tbodyphase = 1.0;
				  double tbody1, tbody2, tbody3, tbody4;
				  tbody1 = J_ME.Braket[a][0]; tbody2 = J_ME.Braket[a][1]; tbody3 = J_ME.Braket[a][2]; tbody4 = J_ME.Braket[a][3]; 
				  if (((tbody1 == m2 && tbody2 == n2 && tbody3 == l2 && tbody4 == k2) ||
				       (tbody1 == l2 && tbody2 == k2 && tbody3 == m2 && tbody4 == n2)) && J_ME.J[a] == angmom3[b] && J_ME.T[a] == iso3[c])
				    {
				      double CGC1 = CGC(Space.levelsj[m1], Space.levelsm[m1], Space.levelsj[n1], Space.levelsm[n1], angmom3[b], angproj1);
				      double CGC2 = CGC(Space.levelsj[l1], Space.levelsm[l1], Space.levelsj[k1], Space.levelsm[k1], angmom3[b], angproj2);
				      double CGC3 = CGC(0.5, Space.levelst[m1], 0.5, Space.levelst[n1], iso3[c], isoproj1);
				      double CGC4 = CGC(0.5, Space.levelst[l1], 0.5, Space.levelst[k1], iso3[c], isoproj2);
				      tempmatel = tempmatel + tbodyphase*CGC1*CGC2*CGC3*CGC4*J_ME.TBME[a];
				    };
				};
			    };
			};

		      double factor1 = 1.0, factor2 = 1.0, factor3;
		      if (Space.levelsn[m1] == Space.levelsn[n1] && Space.levelsj[m1] == Space.levelsj[n1] && Space.levelsl[m1] == Space.levelsl[n1])
			{ factor1 = 2.0; };
		      if (Space.levelsn[l1] == Space.levelsn[k1] && Space.levelsj[l1] == Space.levelsj[k1] && Space.levelsl[l1] == Space.levelsl[k1])
			{ factor2 = 2.0; };
		      factor3 = sqrt(factor1*factor2);
		      
		      if(abs(factor3*tempmatel) > 0.00001)
			{
			  braket[0] = m1 + 1; 
			  braket[1] = n1 + 1;
			  braket[2] = l1 + 1;
			  braket[3] = k1 + 1;
			  M_ME.Braket[ind] = braket;
			  M_ME.TBME[ind] = factor3*tempmatel;
			  ++ind;
			};
		    };
		};
	    };
	};
    };
  
  std::cout << endl;

  M_ME.Braket.resize(ind);
  M_ME.TBME.resize(ind);


  ofstream mschemefile;
  mschemefile.open((PATH + MatrixElements + "_M.int").c_str());
  for (int i = 0; i < int(Space.shellsname.size()); ++i)
    {
      if(i < int(Space.shellsname.size())/2)
	{ mschemefile << "! P - " << Space.shellsname[i] << " = "; }
      else{ mschemefile << "! N - " << Space.shellsname[i] << " = "; }
      for (int j = 0; j < int(Space.shellsm[i].size()); ++j)
	{
	  if(j != int(Space.shellsm[i].size()) - 1)
	    { mschemefile << Space.shellsm[i][j] << ", "; }
	  else{ mschemefile << Space.shellsm[i][j] << "\n"; }
	}
    };
  mschemefile << ind << "\t";
  for (int i = 0; i < int(M_ME.OBME.size()); ++i)
    {
      mschemefile << M_ME.OBME[i] << "\t";
    }
  mschemefile << "\n";
  for (int i = 0; i < int(M_ME.TBME.size()); ++i)
    {
      mschemefile << M_ME.Braket[i][0] << "\t" << M_ME.Braket[i][1] << "\t" << M_ME.Braket[i][2] << "\t" << M_ME.Braket[i][3] << "\t";
      mschemefile << M_ME.TBME[i] << "\n";
    }
  mschemefile.close();

  
  std::cout << "Number of M-Scheme Two-Body Matrix Elements = " << ind << endl << endl;
  
  return M_ME;
}



Matrix_Elements Read_Matrix_Elements(string MEfile, Model_Space Space)
{
  Matrix_Elements ME;

  int NumElements;
  double OBME, TBME;
  int shell1, shell2, shell3, shell4, coupJ, coupT; // J-scheme interaction file contents
  ifstream interaction;	// interaction file
  string interactionline; // interaction file line
  
  string fullpath2 = PATH + MEfile + ".int";
  string fullpath3 = PATH + MEfile + "_M.int";
  interaction.open(fullpath3.c_str()); // try m-scheme first
  size_t intsize = MEfile.size();

  //open interaction file
  if (interaction.is_open())
    { ME.type = 'M'; }
  else if (MEfile[intsize-2] == '_' && MEfile[intsize-1] == 'M')
    { ME.type = 'M'; interaction.open(fullpath2.c_str()); }
  else
    { ME.type = 'J'; interaction.open(fullpath2.c_str()); }

  if (!interaction.is_open())
    {
      cerr << "Matrix Element file, " << MEfile << ", does not exist" << endl; exit(1);
    }
  
  //skip lines that start with '!'
  getline(interaction, interactionline);
  while (interactionline[0] == '!'){ getline(interaction, interactionline); }

  //read matrix element parameters and one-body matrix elements
  istringstream filestring(interactionline);
  filestring >> NumElements;

  //get one-body matrix elements that correspond to the ordered proton/neutron shells
  while (filestring >> OBME)
    {
      ME.OBME.push_back(OBME);
    }

  if(ME.OBME.size() != Space.shellsname.size())
    { cerr << "Space/Interaction Mismatch with " << MEfile << endl; exit(1); }

  vector<int> braket(4);
  ME.Braket.resize(NumElements);
  for(int i = 0; i < NumElements; ++i){ ME.Braket[i].resize(4); };
  ME.J.resize(NumElements);
  ME.T.resize(NumElements);
  ME.TBME.resize(NumElements);
  
  //read two-body parameters and two-body matrix elements
  double tempS1 = 0, tempS2 = 0, tempS3 = 0, tempS4 = 0, tempJ = -1, tempT = -1;
  for(int i = 0; i < NumElements; ++i)
    {
      getline(interaction, interactionline);
      if(ME.type == 'J')
	{ istringstream(interactionline) >> shell1 >> shell2 >> shell3 >> shell4 >> coupJ >> coupT >> TBME;
	  if (shell1 == tempS3 && shell2 == tempS4 && shell3 == tempS1 && shell4 == tempS2 && coupJ == tempJ && coupT == tempT)
	    { --NumElements; --i; continue; };
	  tempS1 = shell1; tempS2 = shell2; tempS3 = shell3; tempS4 = shell4; tempJ = coupJ; tempT = coupT;
	  if(shell2 < shell1)
	    {
	      swap(shell1, shell2);
	      TBME = TBME * pow(-1.0, int(Space.shellsj[shell1 - 1] + Space.shellsj[shell2 - 1] - coupJ - coupT));
	    }
	  if(shell4 < shell3)
	    {
	      swap(shell3, shell4);
	      TBME = TBME * pow(-1.0, int(Space.shellsj[shell3 - 1] + Space.shellsj[shell4 - 1] - coupJ - coupT));
	    }
	  if((shell3 < shell1) || (shell3 == shell1 && shell4 < shell2))
	    {
	      swap(shell1, shell3);
	      swap(shell2, shell4);
	    }
	  braket[0] = shell1;
	  braket[1] = shell2;
	  braket[2] = shell3;
	  braket[3] = shell4;
	  ME.Braket[i] = braket;
	  ME.J[i] = coupJ;
	  ME.T[i] = coupT;
	  ME.TBME[i] = TBME;
	}
      else if(ME.type == 'M')
	{ istringstream(interactionline) >> shell1 >> shell2 >> shell3 >> shell4 >> TBME;
	  if (shell1 == tempS3 && shell2 == tempS4 && shell3 == tempS1 && shell4 == tempS2)
	    { --NumElements; --i; continue; };
	  tempS1 = shell1; tempS2 = shell2; tempS3 = shell3; tempS4 = shell4;
	  if(shell2 < shell1)
	    {
	      swap(shell1, shell2);
	      TBME = TBME * -1.0;
	    }
	  if(shell4 < shell3)
	    {
	      swap(shell3, shell4);
	      TBME = TBME * -1.0;
	    }
	  if((shell3 < shell1) || (shell3 == shell1 && shell4 < shell2))
	    {
	      swap(shell1, shell3);
	      swap(shell2, shell4);
	    }
	  braket[0] = shell1;
	  braket[1] = shell2;
	  braket[2] = shell3;
	  braket[3] = shell4;
	  ME.Braket[i] = braket;
	  ME.TBME[i] = TBME;
	}
    }
  interaction.close();

  ME.Braket.resize(NumElements);
  ME.J.resize(NumElements);
  ME.T.resize(NumElements);
  ME.TBME.resize(NumElements);

  return ME;

}



Matrix_Elements Get_Matrix_Elements(Input_Parameters Parameters, Model_Space Space)
{
  Matrix_Elements ME;
  Matrix_Elements COM_ME;

  std::cout << "Reading Matrix Elements" << endl;
  std::cout << "-----------------------" << endl;

  //Get Matrix Elements
  ME = Read_Matrix_Elements(Parameters.MatrixElements, Space);

  //Get COM matrix elements
  if(Parameters.COM == 1)
    {
      COM_ME = Read_Matrix_Elements(Parameters.COMMatrixElements, Space);
    }

  std::cout << "Number of One-Body Matrix Elements = " << ME.OBME.size() << endl;
  std::cout << "Number of " << ME.type << "-Scheme Two-Body Matrix Elements = " << ME.TBME.size() << endl << endl;

  if(Parameters.COM == 1)
    {
      std::cout << "Number of COM One-Body Matrix Elements = " << COM_ME.OBME.size() << endl;
      std::cout << "Number of COM " << COM_ME.type << "-Scheme Two-Body Matrix Elements = " << COM_ME.TBME.size() << endl << endl;
    }
					  
  if(ME.type == 'J')
    { 
      ME = Convert_To_M_Matrix_Elements(Parameters.MatrixElements, Space, ME);
    }
  if(Parameters.COM == 1)
    {
      if(COM_ME.type == 'J')
	{ 
	  COM_ME = Convert_To_M_Matrix_Elements(Parameters.COMMatrixElements, Space, COM_ME);
	}
      //add COM
      int m1, n1, l1, k1, m2, n2, l2, k2;
      for(int i = 0; i < int(ME.OBME.size()); ++i)
	{ ME.OBME[i] += 50.0 * COM_ME.OBME[i]; }
      for(int i = 0; i < int(ME.TBME.size()); ++i)
	{
	  m1 = ME.Braket[i][0], n1 = ME.Braket[i][1], l1 = ME.Braket[i][2], k1 = ME.Braket[i][3];
	  for(int j = 0; j < int(COM_ME.TBME.size()); ++j)
	    {
	      m2 = COM_ME.Braket[j][0], n2 = COM_ME.Braket[j][1], l2 = COM_ME.Braket[j][2], k2 = COM_ME.Braket[j][3];
	      if(m1 == m2 && n1 == n2 && l1 == l2 && k1 == k2)
		{ ME.TBME[i] += 50.0 * COM_ME.TBME[j]; break; }
	    }
	} 
    }
  
  return ME;

};



double Hartree_Fock_Line(int vector, int index, Model_Space Space, Single_Particle_States States, Matrix_Elements ME)
{
  //vector and index start at 0
  int N = int(States.configs.size());
  double line = States.configs[vector][index] * ME.OBME[Space.levelschemeind[index] - 1];
  //if(vector == 1 && index == 1){ std::cout << line << " "; };
  for(int i = 0; i < int(ME.TBME.size()); ++i)
    {
      int m = ME.Braket[i][0];
      int n = ME.Braket[i][1];
      int l = ME.Braket[i][2];
      int k = ME.Braket[i][3];
      double factor = 1.0;
      for(int lambda = 0; lambda < N/2; ++lambda)
	{
	  if(index == lambda){ continue; }
	  else if(index < lambda && (m != index || n != lambda)){ continue; }
	  else if(lambda < index && (m != lambda || n != index)){ continue; }; //because m < n
	  if(lambda < index){ factor *= -1.0; };
	  for(int nu = 0; nu < N/2; ++nu)
	    {
	      for(int sigma = 0; sigma < N/2; ++sigma)
		{
		  if(nu == sigma){ continue; }
		  else if(nu < sigma && (l != nu || k != sigma)){ continue; }
		  else if(sigma < nu && (l != sigma || k != nu)){ continue; }; //because l < k
		  if(sigma < nu){ factor *= -1.0; };
		  for(int beta = 0; beta < N; ++beta)
		    {
		      line += factor*States.configs[beta][lambda]*States.configs[vector][nu]*States.configs[beta][sigma]*ME.TBME[i];
		      //if(vector == 0){std::cout << factor << " " << States.configs[beta][lambda] << " " << States.configs[vector][nu] << " " << States.configs[beta][sigma] << " " << ME.TBME[i] << endl;};
		    }
		}
	    }
	}
    }
  
  //if(vector == 1 && index == 1){std::cout << endl;};
  return line;

};



Single_Particle_States Hartree_Fock_States(Single_Particle_States States, Model_Space Space, Matrix_Elements ME)
{
  int N = int(States.configs.size());
  Single_Particle_States HF = States;
  double error1 = 1000;
  int ind = 0;
  while(error1 > 0.0000001)
    {
      ++ind;
      //std::cout << ind << endl;
      double error2 = 0;
      vector<vector<double> > tempvec(N);
      for(int i = 0; i < N; ++i){ tempvec[i].resize(N/2); };
      for(int vec = 0; vec < N; ++vec)
	{
	  double norm = 0.0;
	  //if(vec == 0 || vec == 1 || vec == 2){for(int i = 0; i < N/2; ++i){ std::cout << HF.configs[vec][i] << " "; }; std::cout << endl; };
	  for(int index = 0; index < N/2; ++index)
	    {
	      //if(HF.configs[vec][index] > 1){ std::cout << ind << " " << vec << " " << index << " " << HF.configs[vec][index] << endl; };
	      double line = Hartree_Fock_Line(vec, index, Space, HF, ME);
	      //std::cout << "line = " << line << ", ";
	      tempvec[vec][index] = line;
	      //HF.configs[vec][index] = line/HF.energies[vec];
	      norm += line*line;
	      //std::cout << "norm = " << norm << endl;
	    }
	  for(int index = 0; index < N/2; ++index)
	    {
	      tempvec[vec][index] = tempvec[vec][index] / sqrt(norm);
	    }
	  error2 += abs(sqrt(norm) - HF.energies[vec]);
	  HF.energies[vec] = sqrt(norm);
	}
      error1 = error2;
      HF.configs = tempvec;
      //std::cout << "error: " << error2 << endl;
    }
  
  return HF;
  
};

int main(int argc, char* argv[])
{ 
  clock_t t1, t2;    	//initialize program clock
  t1 = clock();

  string inputfile = "input.dat";
  Input_Parameters Parameters = Get_Input_Parameters(inputfile);
  Model_Space Space = Build_Model_Space(Parameters);
  Matrix_Elements ME = Get_Matrix_Elements(Parameters, Space);
  Single_Particle_States States = Build_Single_Particle_States(Parameters, Space, ME);
  Single_Particle_States HFStates = Hartree_Fock_States(States, Space, ME);

  for(int i = 0; i < int(HFStates.configs.size()); ++i)
    {
      for(int j = 0; j < int(HFStates.configs[i].size()); ++j)
	{
	  std::cout << HFStates.configs[i][j] << " ";
	}
      std::cout << ": " << HFStates.energies[i] << endl;
    }

  t2 = clock();
  float diff((float)t2 - (float)t1);
  std::cout << diff / CLOCKS_PER_SEC << "sec" << endl;

  int a;
  std::cin >> a;

  return 0;

}
