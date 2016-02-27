#include "CGC.hpp"
#include "HFfunctions.hpp"

int main(int argc, char* argv[])
{ 
  //clock_t t1, t2;    	//initialize program clock
  //t1 = clock();

  std::string inputfile = "input.dat";
  Input_Parameters Parameters = Get_Input_Parameters(inputfile);
  Model_Space Space = Build_Model_Space(Parameters);
  J_Matrix_Elements JME = Get_J_Matrix_Elements(Parameters, Space);
  Single_Particle_States States = Build_Single_Particle_States(Parameters, Space);
  Single_Particle_States HFStates = Hartree_Fock_States2(Parameters, States, Space, JME);
  J_Matrix_Elements HF_ME = Convert_To_HF_Matrix_Elements(Parameters.MatrixElements, HFStates, Space, JME);
  
  double hf_energy = 0;
  double factor;
  for(int i = 0; i < int(HFStates.holes.size()); ++i){
    hf_energy += (2 * HFStates.h_j[i] + 1) * HFStates.h_energies[i];
    std::cout << i << " " << (2 * HFStates.h_j[i] + 1) * HFStates.h_energies[i] << ", " << hf_energy << std::endl;
  }
  for(int J = 0; J <= Space.max2J; ++J){
    for(int i = 0; i < Space.Pocc; ++i){
      for(int j = 0; j < Space.Pocc; ++j){
	if(i == j){ factor = 2.0; }
	else{ factor = 1.0; }
	hf_energy -= 0.5 * factor * (2 * J + 1) * HF_ME.get_ppJME(i, j, i, j, J);
	std::cout << i << " " << j << " " << J << ", " << -0.5 * factor * (2 * J + 1) * HF_ME.get_ppJME(i, j, i, j, J) << ", " << hf_energy << std::endl;
      }
    }
    std::cout << std::endl;
    //std::cout << hf_energy << endl;
    for(int i = 0; i < Space.Nocc; ++i){
      for(int j = 0; j < Space.Nocc; ++j){
	if(i == j){ factor = 2.0; }
	else{ factor = 1.0; }
	hf_energy -= 0.5 * factor * (2 * J + 1) * HF_ME.get_nnJME(i, j, i, j, J);
	std::cout << i+Space.Pocc << " " << j+Space.Pocc << " " << J << ", " << -0.5 * factor * (2 * J + 1) * HF_ME.get_nnJME(i, j, i, j, J) << ", " << hf_energy << std::endl;
      }
    }
    std::cout << std::endl;
    //std::cout << hf_energy << endl;
    for(int i = 0; i < Space.Pocc; ++i){
      for(int j = 0; j < Space.Nocc; ++j){
	hf_energy -= 0.5 * (2 * J + 1) * HF_ME.get_pnJME(i, j, i, j, J);
	std::cout << i << " " << j+Space.Pocc << " " << J << ", " << -0.5 * (2 * J + 1) * HF_ME.get_pnJME(i, j, i, j, J) << ", " << hf_energy << std::endl;
      }
    }
    std::cout << std::endl;
    //std::cout << hf_energy << endl;
    for(int i = 0; i < Space.Nocc; ++i){
      for(int j = 0; j < Space.Pocc; ++j){
	hf_energy -= 0.5 * (2 * J + 1) * HF_ME.get_pnJME(j, i, j, i, J);
	std::cout << j << " " << i+Space.Pocc << " " << J << ", " << -0.5 * (2 * J + 1) * HF_ME.get_pnJME(j, i, j, i, J) << ", " << hf_energy << std::endl;
      }
    }
  }

  std::cout << "HF Energy = " << hf_energy << std::endl;
		  	  
  //t2 = clock();
  //float diff((float)t2 - (float)t1);
  //std::cout << diff / CLOCKS_PER_SEC << "sec" << endl;

  std::cout << "END!!!" << std::endl;

  return 0;

}
