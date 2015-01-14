#include <iostream>
#include <cmath>
#include <algorithm>
using namespace std;

long long factorial(int n)
{
  long long intfactorial = 1;
  for (int a = 1; a <= n; a++)
    {
      intfactorial = intfactorial*a;
    }
  return intfactorial;
}

long long factorial(double n)
{
  long long intfactorial;
  if ( int(n + 0.01) != int(n) ) //check if input is close to integer
    { cerr << "Factorial intput should be integer" << endl; exit(1); }
  
  intfactorial = 1;
  for (int a = 1; a <= int(n + 0.01); a++)
    {
      intfactorial = intfactorial*a;
    }
  return intfactorial;
}

double CGC(double j1, double m1, double j2, double m2, double jtot, double mtot)
{
  if (abs(m1 + m2 - mtot) > 0.001){ return 0.0; } //projections must add correctly
  
  double num1, den1, fac1, num2_1, num2_2, num2_3, fac2, den3_2, den3_1, den3_3, den3_4, den3_5, fac3; //numerators, denominators, and factors for CGC
  int change1 = 0, change2 = 0; //flags to change from general formula
  double maxk1, maxk2; //variables to find maximum sum
  double tempj, tempm; //variables to change from general formula
  double CGC; //clebsch-gordon coefficient
  if (j1 < j2){ tempj = j1; tempm = m1; j1 = j2; m1 = m2;  j2 = tempj; m2 = tempm; change1 = 1; };
  
  if (mtot < 0){ m1 = -m1; m2 = -m2; change2 = 1; };
  mtot = abs(mtot);
  
  num1 = (2 * jtot + 1) * factorial(jtot + j1 - j2) * factorial(jtot - j1 + j2) * factorial(j1 + j2 - jtot);
  den1 = double(factorial(j1 + j2 + jtot + 1));
  fac1 = sqrt(num1 / den1);
  
  num2_1 = double(factorial(jtot + mtot) * factorial(jtot - mtot));
  num2_2 = double(factorial(j1 - m1) * factorial(j1 + m1));
  num2_3 = double(factorial(j2 - m2) * factorial(j2 + m2));
  fac2 = sqrt(num2_1 * num2_2 * num2_3);
  
  maxk1 = min(j1 + j2 - jtot, j1 - m1);
  maxk2 = min(maxk1, j2 + m2);
  fac3 = 0.0;
  for (int k = 0; k <= maxk2; ++k)
    {
      den3_1 = j1 + j2 - jtot - k;
      den3_2 = j1 - m1 - k;
      den3_3 = j2 + m2 - k;
      den3_4 = jtot - j2 + m1 + k;
      den3_5 = jtot - j1 - m2 + k;
      if (den3_1 >= 0 && den3_2 >= 0 && den3_3 >= 0 && den3_4 >= 0 && den3_5 >= 0)
	{
	  fac3 = fac3 + pow(-1.0, k) / (factorial(k)*factorial(den3_1)*factorial(den3_2)*factorial(den3_3)*factorial(den3_4)*factorial(den3_5));
	};
    };
  
  CGC = fac1*fac2*fac3;
  if (change1 == 1){ CGC = CGC*pow(-1.0, -jtot + j1 + j2); };
  if (change2 == 1){ CGC = CGC*pow(-1.0, -jtot + j1 + j2); };
  
  return CGC;
}

double CGC3(double j1, double m1, double j2, double m2, double jtot, double mtot)
{
  double threej;
  
  if (m1 + m2 + mtot != 0 || abs(m1) > j1 || abs(m2) > j2 || abs(mtot) > jtot || abs(j1 - j2) > jtot || j1 + j2 < jtot)
    {
      threej = 0.0;
    }
  else
    {
      threej = (pow(-1.0, int(j1 - j2 - mtot)) / sqrt(2 * jtot + 1)) * CGC(j1, m1, j2, m2, jtot, -mtot);
    };
  
  return threej;
}

double CGC6(double j1, double j2, double j3, double j4, double j5, double j6)
{
  double sixj = 0.0;
  int S;
  
  for (double m1 = -j1; m1 <= j1; m1 = m1 + 1.0)
    {
      for (double m2 = -j2; m2 <= j2; m2 = m2 + 1.0)
	{
	  for (double m4 = -j4; m4 <= j4; m4 = m4 + 1.0)
	    {
	      for (double m5 = -j5; m5 <= j5; m5 = m5 + 1.0)
		{
		  double m3 = m2 + m1, m6 = m1 - m5;
		  if (m3 != m5 - m4 || m6 != -m2 - m4){ continue; }
		  else{
		    S = int((j1 - m1) + (j2 - m2) + (j3 - m3) + (j4 - m4) + (j5 - m5) + (j6 - m6) + 0.01);
		    sixj = sixj + pow(-1.0, S) * CGC3(j1, m1, j2, m2, j3, -m3) * CGC3(j1, -m1, j5, m5, j6, m6) * CGC3(j4, m4, j5, -m5, j3, m3) * CGC3(j4, -m4, j2, -m2, j6, -m6);
		  };
		};
	    };
	};
    };
  
  return sixj;
}
